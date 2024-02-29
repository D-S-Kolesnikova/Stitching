// this header should be at the beginning of the cpp
// because of a wrong include of needed headers
#include <boost/asio.hpp>

#include "AnalyzerTensorRT.h"
#include "GenericChannelState.h"
#include "InferenceHelperFunctions.h"

#include "TensorRT/Logger.h"
#include "TensorRT/ErrorRecorder.h"
#include "TensorRT/Buffers.h"
#include "TensorRT/CacheHelpers.h"
#include "TensorRT/Int8Calibrator.h"
#include "TensorRT/ReusableRequest.h"

#include <InferenceWrapper/InferenceEngineParams.h>

#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/Unicode.h>
#include <ItvCvUtils/Uuid.h>

#include <cuda_runtime.h>

#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <NvCaffeParser.h>

#include <fmt/format.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#ifndef __aarch64__
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

#include <boost/xpressive/xpressive.hpp>
#include <boost/xpressive/regex_actions.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/detail/utf8_codecvt_facet.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/lock_types.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <map>
#include <vector>
#include <exception>
#include <list>
#include <fstream>

// The cache size shouldn't be too big
// it can grow only if there are too many simultaneously completed inference requests which is very unlikely
constexpr int MAX_CACHED_REQUESTS = 32;
constexpr int MAX_WORKSPACE_SIZE = 256 * (1 << 20);

namespace InferenceWrapper
{
using TensorRTChannelState = GenericChannelState;
std::once_flag g_pluginsInitFlag{};
std::once_flag g_trtEngineMutexesFlag{};
static std::vector<boost::shared_mutex> g_trtEngineMutexes;

int GetGpuCount()
{
    int gpuCount{0};
    auto err = cudaGetDeviceCount(&gpuCount);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            fmt::format("{}; Code: {}; Name: {}; Msg: {}",
                "Cuda: Couldn't request available GPU count.",
                err,
                cudaGetErrorName(err),
                cudaGetErrorString(err)));
    }
    return gpuCount;
}

// ##############################
// >>>>>>>> TrtEngineParameters
// TRT engine configuration parameters
struct TrtEngineParameters
{
    int batchSize{1}; //!< Number of inputs in a batch
    bool int8{false}; //!< Allow runnning the network in Int8 mode.
    bool fp16{true};  //!< Allow running the network in FP16 mode.
    bool allowGPUFallback{true};
    std::vector<std::string> inputTensorNames{"data"};
    std::vector<std::string> outputTensorNames{
        "prob",
        "detection_out",
        "out",
        "fc5",
        "Mconv7_stage2_L1",
        "Mconv7_stage2_L2",
        "keep_count"
    };
};

// ##############################
// >>>>>>>> Analyzer
template<itvcvAnalyzerType analyzerType>
class AnalyzerTensorRT : public IAnalyzer<analyzerType>
{
private:
    // Holds request's necessary information that will be used in result's callback function.
    // - It is used once for every request and be destroyed after.
    struct AsyncInferRequestInfo
    {
        AsyncInferRequestInfo(
            const std::string& uuid,
            InferenceCounter&& counter,
            AnalyzerTensorRT* analyzer,
            std::unique_ptr<ReusableRequest>&& request,
            PreprocessContext preprocessContext,
            boost::promise<typename InferenceResultTraits<analyzerType>::InferenceResult_t>&& promise)
            : uuid(uuid)
            , counter(std::move(counter))
            , analyzer(analyzer)
            , request(std::move(request))
            , preprocessContext(preprocessContext)
            , promise(std::move(promise))
        {
        }

        std::string uuid;
        InferenceCounter counter;
        AnalyzerTensorRT* analyzer;

        std::unique_ptr<ReusableRequest> request;
        PreprocessContext preprocessContext;
        boost::promise<typename InferenceResultTraits<analyzerType>::InferenceResult_t> promise;
        std::unique_ptr<std::atomic<bool>> done = std::make_unique<std::atomic<bool>>(false);
    };

public:
    AnalyzerTensorRT(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

    ~AnalyzerTensorRT();

    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer(const ItvCv::Frame& frame, InferenceCounter) override;
    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer(const std::vector<ItvCv::Frame>& imgs, InferenceCounter) override;

    cv::Size GetInputGeometry() const override;

    std::shared_ptr<IChannelState> CreateChannelState(
        std::uint32_t max_active_requests,
        ITV8::Statistics::ISink* sink,
        std::chrono::milliseconds ttl,
        std::initializer_list<std::pair<const char*, const char*>> labels) override;

private:
    bool Build();

    bool CaffeParser(
        std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network,
        const std::string &model, const std::string &weights);

    bool ONNXParser(
        std::unique_ptr<nvonnxparser::IParser>& parser,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network,
        const std::string &weights);

    template<class T>
    typename std::enable_if<
        (std::is_same<T, cv::Mat>::value) ||
        (std::is_same<T, cv::cuda::GpuMat>::value),
        std::vector<T>>::type
    WrapOneSampleInBuffer(void* buffer, const int batchInd)
    {
        std::vector<T> inputChannels;

        const int channels = m_numChannels;
        const int height = m_inputGeometry.height;
        const int width = m_inputGeometry.width;

        auto data = static_cast<float*>(buffer) + (batchInd * width * height * channels);
        for (int i=0; i < channels; ++i)
        {
            inputChannels.emplace_back(height, width, CV_32FC1, data);
            data += (width * height);
        }
        return inputChannels;
    }

    void PreprocessHostFrame(const ItvCv::Frame& frame, void* inputBuffer, PreprocessContext& ctx, const int batchInd = 0)
    {
        auto cvFrame = cv::Mat(frame.height, frame.width, CV_8UC3, (void*) frame.data, frame.stride);
        cv::Mat tmpFrame;

        if (cvFrame.size() == m_inputGeometry)
        {
            tmpFrame = cvFrame;
        }
        else
        {
            tmpFrame = cv::Mat(m_inputGeometry, cvFrame.type());
            m_sampleResizerFunc(cvFrame, tmpFrame, ctx);
        }

        tmpFrame.convertTo(tmpFrame, CV_32FC3);

        if (m_engineParameters->net->inputParams.pixelFormat == ItvCv::PixelFormat::RGB)
        {
            cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2RGB);
        }
        if (m_engineParameters->net->inputParams.pixelFormat == ItvCv::PixelFormat::GRAY)
        {
            cv::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2GRAY);
        }

        auto inputChannels = WrapOneSampleInBuffer<cv::Mat>(inputBuffer, batchInd);
        cv::split(tmpFrame, inputChannels);

        if(m_engineParameters->net->inputParams.normalizationValues.empty())
        {
            throw std::runtime_error("Empty normalization values for GPU preprocessing.");
        }
        const auto& normalizationValues = m_engineParameters->net->inputParams.normalizationValues;
        if (normalizationValues.size() != inputChannels.size())
        {
            throw std::runtime_error("Different size of normalization values and image channels.");
        }
        // The position in the normalization value must match the pixel format(channel ordering)
        for (auto planeNum = 0; planeNum < inputChannels.size(); ++planeNum)
        {
            const auto normalizationValue = normalizationValues[planeNum];
            inputChannels[planeNum].convertTo(
                inputChannels[planeNum],
                CV_32FC1,
                normalizationValue.scale,
                -normalizationValue.mean);
        }
    }

#ifndef __aarch64__
    void PreprocessGpuFrame(const ItvCv::Frame& frame, void* inputBuffer, PreprocessContext& ctx, cudaStream_t stream)
    {
        auto cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
        auto cvFrame = cv::cuda::GpuMat(frame.height, frame.width, CV_8UC3, (void*) frame.data, frame.stride);
        cv::cuda::GpuMat tmpFrame;

        if (cvFrame.size() == m_inputGeometry)
        {
            tmpFrame = cvFrame;
        }
        else
        {
            tmpFrame = cv::cuda::GpuMat(m_inputGeometry, cvFrame.type());
            m_gpuSampleResizerFunc(cvFrame, tmpFrame, ctx, cvStream);
        }

        tmpFrame.convertTo(tmpFrame, CV_32FC3);

        if (m_engineParameters->net->inputParams.pixelFormat == ItvCv::PixelFormat::RGB)
        {
            cv::cuda::cvtColor(tmpFrame, tmpFrame, cv::COLOR_BGR2RGB);
        }

        auto inputChannels = WrapOneSampleInBuffer<cv::cuda::GpuMat>(inputBuffer, 0);
        cv::cuda::split(tmpFrame, inputChannels, cvStream);

        if(m_engineParameters->net->inputParams.normalizationValues.empty())
        {
            throw std::runtime_error("Empty normalization values for GPU preprocessing.");
        }
        const auto& normalizationValues = m_engineParameters->net->inputParams.normalizationValues;
        if (normalizationValues.size() != inputChannels.size())
        {
            throw std::runtime_error("Different size of normalization values and image channels.");
        }
        // The position in the normalization value must match the pixel format(channel ordering)
        for (auto planeNum = 0; planeNum < inputChannels.size(); ++planeNum)
        {
            const auto normalizationValue = normalizationValues[planeNum];
            inputChannels[planeNum].convertTo(inputChannels[planeNum],
                CV_32FC1,
                normalizationValue.scale,
                -normalizationValue.mean,
                cvStream);
        }
    }
#endif

    // not thread safe, use mutex (m_availableRequestsMutex) before calling
    std::unique_ptr<ReusableRequest> GetReusableRequest();

    bool EnqueueInference(std::unique_ptr<InferenceWrapper::ReusableRequest>& request);

    // not thread safe, use mutexes (m_availableRequestsMutex, m_asyncInferRequestsMutex) before calling
    void CleanupCompletedRequests();

    static void InferenceResultCallback(cudaStream_t stream, cudaError_t status, void* userData);

    std::string AdoptCaffeModelToTensorRT(const char* modelData, size_t size);

private:
    ITV8::ILogger* m_logger;
    std::shared_ptr<EngineCreationParams> m_engineParameters;
    SampleResizerFunc m_sampleResizerFunc;
#ifndef __aarch64__
    GpuSampleResizerFunc m_gpuSampleResizerFunc;
#endif

    int m_numChannels{0};
    cv::Size m_inputGeometry{};
    std::vector<cv::Mat> m_meanVector{};
    std::vector<float> m_scales{};

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine{nullptr};
    std::unique_ptr<ErrorRecorder> m_errorRecorder{nullptr};
    FileCacheHelper m_cacheHelper;

    std::vector<std::unique_ptr<ReusableRequest>> m_availableRequests;//todo Посмотреть размер
    std::mutex m_availableRequestsMutex;
    /**
     * @brief protects enqueue from data race inside tensorrt
     * bug report:
     * https://forums.developer.nvidia.com/t/error-run-2-context-parallel-in-tensorrt7/111236
     */
    std::mutex m_enqueueMutex;

    TrtEngineParameters m_trtEngineParameters;

    std::map<std::string, AsyncInferRequestInfo> m_asyncInferRequests;
    // TODO consider using of size_t instead of uuid, generating uuid takes a lot of time +7% cpu time, while size_t takes 0.03%
    std::mutex m_asyncInferRequestsMutex;

    // allows to ignore all async callbacks:
    // - set to true by destructor.
    // - FIXME set to true when an engine reset is required (critical error)
    std::atomic<bool> m_ignoreAsyncCallbacks{false};
};

template<itvcvAnalyzerType analyzerType>
AnalyzerTensorRT<analyzerType>::AnalyzerTensorRT(
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters)
    : m_logger(parameters->logger)
    , m_engineParameters(parameters)
    , m_sampleResizerFunc(DefineHostResizeFunction(parameters->net->inputParams.resizePolicy))
#ifndef __aarch64__
    , m_gpuSampleResizerFunc(DefineGpuResizeFunction(parameters->net->inputParams.resizePolicy))
#endif
    , m_cacheHelper(parameters->logger)
{
    ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "this=" << this << " Entered Analyzer's constructor");

    // build network
    if (!Build())
    {
        error = itvcvErrorOther;
        throw std::runtime_error("Error building the network.");
    }

    ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "this=" << this << " Leaving Analyzer's constructor");
}

template<itvcvAnalyzerType analyzerType>
AnalyzerTensorRT<analyzerType>::~AnalyzerTensorRT()
{
    ITVCV_LOG(m_logger, ITV8::LOG_DEBUG,
        "this=" << this << " Entered Analyzer's destructor");
    //! DON'T CALL nvcaffeparser1::shutdownProtobufLibrary
    //! TensorRT recommended to shutdown this library. However this causes problems when we want to parse
    //! a new network after the first time. Since we can't use any caffe parser anymore.
    // nvcaffeparser1::shutdownProtobufLibrary();
    m_ignoreAsyncCallbacks.store(true);

    // wait for async inferences to complete
    // to avoid being called when *this destroyed
    while (true)
    {
        {
            std::lock_guard<std::mutex> g1(m_availableRequestsMutex);
            std::lock_guard<std::mutex> g2(m_asyncInferRequestsMutex);
            CleanupCompletedRequests();
            if (m_asyncInferRequests.empty())
            {
                ITVCV_LOG(m_logger, ITV8::LOG_DEBUG,
                    "this=" << this << " Leaving Analyzer's destructor");
                return;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AnalyzerTensorRT<analyzerType>::AsyncInfer(const ItvCv::Frame& frame, InferenceCounter counter)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;

    if (frame.channels != 3)
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorFrame, {}));
    }

    boost::shared_lock<boost::shared_mutex> l(g_trtEngineMutexes[m_engineParameters->gpuDeviceNumToUse], boost::defer_lock);
    if (!l.try_lock())
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorGpuIsBusyBuildingNewEngine, {}));
    }

    // cudaSetDevice binds current thread to specific device.
    // After that all device operations performed from that thread
    // will use the specified device.
    // It's always necesarry to call cudaSetDevice
    // as this function can be called from arbitrary threads each time
    cv::cuda::setDevice(m_engineParameters->gpuDeviceNumToUse);
    auto returnCode = cudaSetDevice(m_engineParameters->gpuDeviceNumToUse);
    if (returnCode != cudaSuccess)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Cuda: unable to set proper cuda device: "
            << m_engineParameters->gpuDeviceNumToUse
            << "; Code: " << returnCode
            << "; Name: " << cudaGetErrorName(returnCode)
            << "; Msg: " << cudaGetErrorString(returnCode));
        return boost::make_ready_future(InferenceResult_t(itvcvError::itvcvErrorChoosingGPU, {}));
    }

    std::unique_ptr<ReusableRequest> request;
    {
        std::lock_guard<std::mutex> g1(m_availableRequestsMutex);
        {
            std::lock_guard<std::mutex> g2(m_asyncInferRequestsMutex);
            CleanupCompletedRequests();
        }
        // todo modify buffer manager so it allocate host buffer if needed.
        request = GetReusableRequest();
    }

    PreprocessContext ctx{};
#ifndef __aarch64__
    if (frame.memType == ItvCv::MemType::GPU)
    {
        PreprocessGpuFrame(frame, request->bufferManager.GetDeviceBuffer(m_trtEngineParameters.inputTensorNames[0]), ctx, request->stream);
    }
    else
#endif
    if (frame.memType == ItvCv::MemType::Host)
    {
        PreprocessHostFrame(frame, request->bufferManager.GetHostBuffer(m_trtEngineParameters.inputTensorNames[0]), ctx);
        request->bufferManager.CopyInputToDeviceAsync(request->stream);
    }
    else
    {
        throw std::logic_error("Unsupported memory type");
    }

    ctx.widths.emplace_back(frame.width);
    ctx.heights.emplace_back(frame.height);
    if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)
    {
        ctx.hpeParams = boost::get<ItvCv::PoseNetworkParams>(m_engineParameters->net->networkParams);
    }

    if (!EnqueueInference(request))
    {
        auto err = cudaGetLastError();
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Couldn't enqueue inference"
            << "; Code: " << err
            << "; Name: " << cudaGetErrorName(err)
            << "; Msg: " << cudaGetErrorString(err));
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
    request->bufferManager.CopyOutputToHostAsync(request->stream);

    auto uuid = ItvCv::Utils::ConvertUuidToString(ItvCv::Utils::GenerateUuid());
    auto promise = boost::promise<InferenceResult_t>{};
    auto future = promise.get_future();
    AsyncInferRequestInfo* requestInfo = nullptr;
    {
        std::lock_guard<std::mutex> g(m_asyncInferRequestsMutex);
        requestInfo = &m_asyncInferRequests.emplace(
            uuid,
            AsyncInferRequestInfo(
                uuid,
                std::move(counter),
                this,
                std::move(request),
                ctx,
                std::move(promise))).first->second;
    }

    if (auto err = cudaStreamAddCallback(requestInfo->request->stream, &AnalyzerTensorRT::InferenceResultCallback, requestInfo, 0))
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Unable to set cuda stream callback."
            << "; Code: " << err
            << "; Name: " << cudaGetErrorName(err)
            << "; Msg: " << cudaGetErrorString(err));
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
    return std::move(future);
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AnalyzerTensorRT<analyzerType>::AsyncInfer(
    const std::vector<ItvCv::Frame>& frames, InferenceCounter counter)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;

    boost::shared_lock<boost::shared_mutex> l(g_trtEngineMutexes[m_engineParameters->gpuDeviceNumToUse], boost::defer_lock);
    if (!l.try_lock())
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorGpuIsBusyBuildingNewEngine, {}));
    }

    // cudaSetDevice binds current thread to specific device.
    // After that all device operations performed from that thread
    // will use the specified device.
    // It's always necesarry to call cudaSetDevice
    // as this function can be called from arbitrary threads each time
    cv::cuda::setDevice(m_engineParameters->gpuDeviceNumToUse);
    auto returnCode = cudaSetDevice(m_engineParameters->gpuDeviceNumToUse);
    if (returnCode != cudaSuccess)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Cuda: unable to set proper cuda device: "
            << m_engineParameters->gpuDeviceNumToUse
            << "; Code: " << returnCode
            << "; Name: " << cudaGetErrorName(returnCode)
            << "; Msg: " << cudaGetErrorString(returnCode));
        return boost::make_ready_future(InferenceResult_t(itvcvError::itvcvErrorChoosingGPU, {}));
    }

    std::unique_ptr<ReusableRequest> request;
    {
        std::lock_guard<std::mutex> g1(m_availableRequestsMutex);
        {
            std::lock_guard<std::mutex> g2(m_asyncInferRequestsMutex);
            CleanupCompletedRequests();
        }
        // todo modify buffer manager so it allocate host buffer if needed.
        request = GetReusableRequest();
    }
    PreprocessContext ctx{};
    for(auto i =0; i < frames.size(); i++)
    {
#ifndef __aarch64__
        if (frames[i].memType == ItvCv::MemType::GPU)
        {
            PreprocessGpuFrame(frames[i], request->bufferManager.GetDeviceBuffer(m_trtEngineParameters.inputTensorNames[0]), ctx, request->stream);
        }
        else
#endif
        if (frames[i].memType == ItvCv::MemType::Host)
        {
            PreprocessHostFrame(frames[i], request->bufferManager.GetHostBuffer(m_trtEngineParameters.inputTensorNames[0]), ctx, i);
            request->bufferManager.CopyInputToDeviceAsync(request->stream);
        }
        else
        {
            throw std::logic_error("Unsupported memory type");
        }

        ctx.widths.emplace_back(frames[i].width);
        ctx.heights.emplace_back(frames[i].height);
        if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)
        {
            ctx.hpeParams = boost::get<ItvCv::PoseNetworkParams>(m_engineParameters->net->networkParams);
        }

    }
    if (!EnqueueInference(request))
    {
        auto err = cudaGetLastError();
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Couldn't enqueue inference"
            << "; Code: " << err
            << "; Name: " << cudaGetErrorName(err)
            << "; Msg: " << cudaGetErrorString(err));
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
    request->bufferManager.CopyOutputToHostAsync(request->stream);

    auto uuid = ItvCv::Utils::ConvertUuidToString(ItvCv::Utils::GenerateUuid());
    auto promise = boost::promise<InferenceResult_t>{};
    auto future = promise.get_future();
    AsyncInferRequestInfo* requestInfo = nullptr;
    {
        std::lock_guard<std::mutex> g(m_asyncInferRequestsMutex);
        requestInfo = &m_asyncInferRequests.emplace(
            uuid,
            AsyncInferRequestInfo(
                uuid,
                std::move(counter),
                this,
                std::move(request),
                ctx,
                std::move(promise))).first->second;
    }

    if (auto err = cudaStreamAddCallback(requestInfo->request->stream, &AnalyzerTensorRT::InferenceResultCallback, requestInfo, 0))
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Unable to set cuda stream callback."
            << "; Code: " << err
            << "; Name: " << cudaGetErrorName(err)
            << "; Msg: " << cudaGetErrorString(err));
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
    return std::move(future);
}

template<itvcvAnalyzerType analyzerType>
cv::Size AnalyzerTensorRT<analyzerType>::GetInputGeometry() const
{
    return m_inputGeometry;
}

template<itvcvAnalyzerType analyzerType>
std::shared_ptr<IChannelState> AnalyzerTensorRT<analyzerType>::CreateChannelState(
    std::uint32_t max_active_requests,
    ITV8::Statistics::ISink* sink,
    std::chrono::milliseconds ttl,
    std::initializer_list<std::pair<const char*, const char*>> labels)
{
    return std::make_shared<TensorRTChannelState>(max_active_requests, sink, ttl, labels);
}

template<itvcvAnalyzerType analyzerType>
bool AnalyzerTensorRT<analyzerType>::Build()
{
    std::call_once(g_pluginsInitFlag, initLibNvInferPlugins, &TrtLogger::GetInstance(m_logger), "");
    std::call_once(g_trtEngineMutexesFlag, [](){ g_trtEngineMutexes = decltype(g_trtEngineMutexes)(GetGpuCount()); });

    // define custom input dims
    m_trtEngineParameters.batchSize = 1;
    // TODO: this should be done by net information and not by every single framework
    // find custom size for input layer (w,h)

    m_inputGeometry = cv::Size(m_engineParameters->net->inputParams.inputWidth, m_engineParameters->net->inputParams.inputHeight);

    if (analyzerType == itvcvAnalyzerMaskSegments)
    {
        if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::onnx)
        {
            m_inputGeometry =
                cv::Size(m_engineParameters->net->inputParams.inputWidth, m_engineParameters->net->inputParams.inputHeight);
        }
    }
    else if (analyzerType == itvcvAnalyzerSSD)
    {
        if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::onnx)
        {
            if (m_engineParameters->net->commonParams.architecture == ItvCv::ArchitectureType::Yolo)
            {
                m_inputGeometry = cv::Size(320, 320);
            }
        }
    }

    std::string& model = m_engineParameters->net->modelData;
    std::string weights = ItvCv::ConsumeWeightsData(m_engineParameters->net);

    // activate proper CUcontext that matches used CUDA device.
    auto returnCode = cudaSetDevice(m_engineParameters->gpuDeviceNumToUse);
    if (returnCode != cudaSuccess)
    {
        throw std::runtime_error(fmt::format(
            "Cuda: unable to set proper cuda device: {}; Code: {}; Name: {}; Msg: {}",
            m_engineParameters->gpuDeviceNumToUse,
            returnCode,
            cudaGetErrorName(returnCode),
            cudaGetErrorString(returnCode)));
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(TrtLogger::GetInstance(m_logger)));
    if (!builder)
    {
        throw std::runtime_error("Failed to create engine builder.");
    }

    const auto calibrationDataPath = [this, trtSupportsInt8=builder->platformHasFastInt8()]() -> boost::filesystem::path
        {
            const auto tmp = boost::filesystem::path(
                m_engineParameters->weightsFilePath,
                boost::filesystem::detail::utf8_codecvt_facet()).replace_extension(".info");
            if (m_engineParameters->int8 && trtSupportsInt8 && boost::filesystem::exists(tmp))
            {
                return tmp;
            }
            return {};
        }();

    // Try to load cached engine otherwise build a new one from scratch
    const auto cachedEngineFileName = m_cacheHelper.GetFileName(
        ITV8::Size(m_inputGeometry.width, m_inputGeometry.height),
        model,
        weights,
        boost::filesystem::path(m_engineParameters->weightsFilePath).stem().string(),
        !calibrationDataPath.empty());

    m_engine = m_cacheHelper.Load(cachedEngineFileName);
    if (m_engine)
    {
        m_trtEngineParameters.inputTensorNames.clear();
        m_trtEngineParameters.outputTensorNames.clear();
        for (auto i = 0; i < m_engine->getNbBindings(); ++i)
        {
            if (m_engine->bindingIsInput(i))
            {
                m_trtEngineParameters.inputTensorNames.emplace_back(m_engine->getBindingName(i));
            }
            else
            {
                m_trtEngineParameters.outputTensorNames.emplace_back(m_engine->getBindingName(i));
            }
        }
    }
    else
    {
        std::unique_ptr<Int8EntropyCalibrator2> calibrator;
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            throw std::runtime_error("Failed to create IBuilderConfig.");
        }

        // build configuration
#ifdef __aarch64__
        config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
#else
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, MAX_WORKSPACE_SIZE);
#endif

        if (builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "nvinfer1::BuilderFlag::kFP16 enabled.");
        }

        if (!calibrationDataPath.empty())
        {
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "INT8 flag is enabled and file with calibration information is founded.");
            cv::Scalar meanValues;
            double scaleFactor;

            if (m_engineParameters->net->inputParams.normalizationValues.empty())
            {
                throw std::runtime_error("Empty normalization values for GPU preprocessing.");
            }

            const auto& normalizationValues = m_engineParameters->net->inputParams.normalizationValues;
            meanValues = cv::Scalar(normalizationValues[0].mean, normalizationValues[1].mean, normalizationValues[2].mean);
            scaleFactor = static_cast<double>(normalizationValues[0].scale);

            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            calibrator = std::make_unique<Int8EntropyCalibrator2>(
                m_logger,
                m_trtEngineParameters.batchSize,
                m_inputGeometry,
                meanValues,
                scaleFactor,
                m_engineParameters->net->inputParams.pixelFormat,
                calibrationDataPath.string());

            config->setInt8Calibrator(calibrator.get());
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "nvinfer1::BuilderFlag::kINT8 enabled.");
        }
        else if (m_engineParameters->int8)
        {
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "No calibration data or your GPU cannot use INT8");
        }

        // parse buffers to network
        /// NOTE caffeparser should be in scope while building the engine
        /// NOTE caffeparser can not support explicit batch sizing properly (many ANN parsing fails)
        std::unique_ptr<nvinfer1::INetworkDefinition> network{ nullptr };
        std::unique_ptr<nvcaffeparser1::ICaffeParser> caffeParser{ nullptr };
        std::unique_ptr<nvonnxparser::IParser> onnxParser{ nullptr };
        if(m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::onnx)
        {
            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
            if (!network)
            {
                throw std::runtime_error("Failed to create INetworkDefinition.");
            }
            onnxParser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, TrtLogger::GetInstance(m_logger)));
            if (!onnxParser)
            {
                throw std::runtime_error("Failed to create nvonnxparser.");
            }
            ONNXParser(onnxParser, network, weights);
        }
        else if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::caffe)
        {
            builder->setMaxBatchSize(m_trtEngineParameters.batchSize);
            network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
            if (!network)
            {
                throw std::runtime_error("Failed to create INetworkDefinition.");
            }
            caffeParser = std::unique_ptr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
            if (!caffeParser)
            {
                throw std::runtime_error("Failed to create ICaffeParser.");
            }
            CaffeParser(caffeParser, network, model, weights);
        }
        else
        {
            throw std::logic_error("Failed to create INetworkDefinition. Unknown model representation");
        }
        if (!network->getNbLayers())
        {
            ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
                "Failed to parse network."
                << " Path:" << m_engineParameters->weightsFilePath
                << " Mode:" << m_engineParameters->mode);
            return false;
        }
        // apply custom input size
        if (m_inputGeometry.area())
        {
            if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::caffe)
            {
                network->getInput(0)->setDimensions(nvinfer1::Dims3(3, m_inputGeometry.height, m_inputGeometry.width));
            }
            else if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::onnx)
            {
                m_trtEngineParameters.batchSize = m_engine->getBindingDimensions(0).d[0];
                network->getInput(0)->setDimensions(nvinfer1::Dims4(m_trtEngineParameters.batchSize, 3, m_inputGeometry.height, m_inputGeometry.width));
            }
            else
            {
                throw std::logic_error("Failed to set the dimensions of a tensor. Unknown model representation");
            }
        }
        // build the engine
        {
            // NOTE Only one network could be initialized in one CUcontext at the same time.
            boost::unique_lock<boost::shared_mutex> lock(g_trtEngineMutexes[m_engineParameters->gpuDeviceNumToUse]);
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "ANN optimization on GPU has started, frames will be skipped.");
            m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "ANN optimization on GPU is done, back to normal.");
        }
        if (!m_engine)
        {
            throw std::runtime_error("Failed to build ICudeEngine; Network: " + m_engineParameters->weightsFilePath);
        }
        if (!m_cacheHelper.Save(cachedEngineFileName, m_engine))
        {
            ITVCV_LOG(m_logger, ITV8::LOG_INFO, "Cached engine file was not created.");
        }
    }

    m_errorRecorder = std::make_unique<ErrorRecorder>(m_logger);
    m_engine->setErrorRecorder(m_errorRecorder.get());

    /// join output layers names into one string for caffe networks which have more than 1 output.
    //! NOTE bad architecture and should be replaced.
    //! Reason to keep these names is caffe models for which we should know the input and output layers names in order to mark and access them.
    // TODO consider adding these input/output names into ann file as 3rd file.
    // TODO pass outputnames to ProcessTensorRTOutput as a vector instead of a joined string
    if (
        analyzerType == itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator
        || analyzerType == itvcvAnalyzerType::itvcvAnalyzerSiamese
        || analyzerType == itvcvAnalyzerType::itvcvAnalyzerPointDetection)
    {
        m_trtEngineParameters.outputTensorNames = { fmt::format("{}", fmt::join(m_trtEngineParameters.outputTensorNames, ",")) };
    }
    // update meta info about network's input.
    auto inputDims = m_engine->getBindingDimensions(0);
    /**
     * @brief used to allow proper reading of CHW values ignoring N if exist.
     * Dims will be in CHW if caffe parser has been used (implicit batch sizing) otherwise NCHW.
     */
    const auto offset = m_engine->hasImplicitBatchDimension() ? 0 : 1;
    m_numChannels = inputDims.d[0 + offset];

    auto inputHeight = inputDims.d[1 + offset];
    auto inputWidth = inputDims.d[2 + offset];
    m_inputGeometry = cv::Size(inputWidth, inputHeight);
    return true;
}

template<itvcvAnalyzerType analyzerType>
bool AnalyzerTensorRT<analyzerType>::CaffeParser(
    std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    const std::string& model,
    const std::string& weights)
{
    if (!parser)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Failed to create parser.");
        return false;
    }
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor;

    std::string modelStr;
    modelStr = AdoptCaffeModelToTensorRT(
        model.data(),
        model.size());

    blobNameToTensor = parser->parseBuffers(
        reinterpret_cast<const uint8_t*>(modelStr.data()),
        modelStr.size(),
        reinterpret_cast<const uint8_t*>(weights.data()),
        weights.size(),
        *network,
        nvinfer1::DataType::kFLOAT);
    if (!blobNameToTensor)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Error parsing model's file!");
        return false;
    }

    // mark output and remove unwanted output layer names
    auto& b = m_trtEngineParameters.outputTensorNames;
    for (auto i = b.begin(); i != b.end();)
    {
        const auto tensor = blobNameToTensor->find((*i).c_str());
        if (tensor)
        {
            network->markOutput(*tensor);
            ++i;
        }
        else
        {
            i = b.erase(i);
        }
    }

    return true;
}

template<itvcvAnalyzerType analyzerType>
bool AnalyzerTensorRT<analyzerType>::ONNXParser(
    std::unique_ptr<nvonnxparser::IParser>& parser,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    const std::string& weights)
{
    if (!parser)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Failed to create ONNX parser.");
        return false;
    }
    auto isParsed = parser->parse(weights.data(), weights.size());
    if (!network->getInput(0))
    {
        return false;
    }
    m_trtEngineParameters.inputTensorNames = { network->getInput(0)->getName() };
    m_trtEngineParameters.outputTensorNames.clear();
    for (int i = 0; i < network->getNbOutputs(); ++i)
    {
        m_trtEngineParameters.outputTensorNames.emplace_back(network->getOutput(i)->getName());
    }
    return isParsed;
}

template<itvcvAnalyzerType analyzerType>
std::unique_ptr<ReusableRequest> AnalyzerTensorRT<analyzerType>::GetReusableRequest()
{
    if (!m_availableRequests.empty())
    {
        auto h = std::move(m_availableRequests.back());
        m_availableRequests.pop_back();
        return h;
    }
    else
    {
        return std::make_unique<ReusableRequest>(m_engine, m_logger);
    }
}

template<itvcvAnalyzerType analyzerType>
bool AnalyzerTensorRT<analyzerType>::EnqueueInference(std::unique_ptr<InferenceWrapper::ReusableRequest> &request)
{
    if (NV_TENSORRT_MAJOR < 7 || m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::caffe)
    {
        std::lock_guard<std::mutex> lock(m_enqueueMutex);
        return request->context->enqueue(
            BATCHSIZE_ASYNC,
            request->bufferManager.GetDeviceBindings().data(),
            request->stream,
            nullptr);
    }
    else if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::onnx)
    {
        return request->context->enqueueV2(
            request->bufferManager.GetDeviceBindings().data(),
            request->stream,
            nullptr);
    }
    else
    {
        throw std::logic_error("Failed to Asynchronously execute inference. Unknown model representation");
    }
}

template<itvcvAnalyzerType analyzerType>
void AnalyzerTensorRT<analyzerType>::CleanupCompletedRequests()
{
    for (auto it = m_asyncInferRequests.begin(); it != m_asyncInferRequests.end();)
    {
        if (it->second.done->load())
        {
            if (m_availableRequests.size() < MAX_CACHED_REQUESTS)
            {
                m_availableRequests.emplace_back(std::move(it->second.request));
            }
            it = m_asyncInferRequests.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

template<itvcvAnalyzerType analyzerType>
void AnalyzerTensorRT<analyzerType>::InferenceResultCallback(cudaStream_t stream, cudaError_t status, void* userData)
{
    // WARNING: don't call any cuda functions from this callback
    // it is CUDA requirement

    auto requestInfo = reinterpret_cast<AsyncInferRequestInfo*>(userData);
    auto counter = std::move(requestInfo->counter);
    auto self = requestInfo->analyzer;

    if (self->m_ignoreAsyncCallbacks.load())
    {
        requestInfo->promise.set_value({itvcvErrorOther, {}});
        return;
    }

    if (status != cudaSuccess)
    {
        ITVCV_LOG(self->m_logger, ITV8::LOG_ERROR,
            "Error occured processing result async"
            << "; Code: " << status
            << "; Name: " << cudaGetErrorName(status)
            << "; Msg: " << cudaGetErrorString(status));
        requestInfo->promise.set_value({itvcvErrorOther, {}});
    }
    else
    {
        auto result = ProcessTensorRTOutput<analyzerType>(
            requestInfo->request->bufferManager,
            BATCHSIZE_ASYNC,
            self->m_trtEngineParameters.outputTensorNames[0],
            requestInfo->preprocessContext,
            self->m_engineParameters->net);
        counter.template OnInferenceSucceeded<TensorRTChannelState>();
        requestInfo->promise.set_value({itvcvErrorSuccess, result[0]});
    }

    requestInfo->done->store(true);
}

template<itvcvAnalyzerType analyzerType>
std::string AnalyzerTensorRT<analyzerType>::AdoptCaffeModelToTensorRT(
    const char* modelData,
    size_t size)
{
    std::string newModel;
    std::map<std::string, std::string> replaceMap
    {
        {"relu6", "\"ReLU\""},
        {"DepthwiseConvolution", "type: \"Convolution\""},
        {"flatten", "type: \"Reshape\""},
        {
            "flattenParam",
            "reshape_param { shape { dim:0 dim:-1 dim:1 dim:1 } }"
        },
        {"output", "top:\"detection_out\" top:\"keep_count\""},
    };

    using namespace boost::xpressive;
    std::string inputParamRegStr = "(net_parametrs:{(name:\"\\w*\"\\s*value:[0-9\\.]*)}\\s*){7}";
    std::string result;

    int ind = 0;
    mark_tag relu6(++ind);
    mark_tag depthwiseConvolution(++ind);
    mark_tag flatten(++ind);
    mark_tag flattenParam(++ind);
    mark_tag output(++ind);
    cregex re = (relu6 = "\"ReLU6\"")
            | (flatten = "type: \"Flatten\"")
            | (depthwiseConvolution = "type: \"DepthwiseConvolution\"")
            | (flattenParam = "flatten_param" >> +~as_xpr('}') >> '}')
            | (output = "top: \"detection_out\"");
    int lastPos = 0;
    for (auto it = cregex_iterator(modelData, modelData + size, re); it !=
         cregex_iterator(); ++it)
    {
        newModel.append(modelData + lastPos, it->position(0) - lastPos);
        lastPos = it->position(0) + it->length();
        if ((*it)[relu6].matched)
        {
            newModel.append(replaceMap["relu6"]);
        }
        else if ((*it)[depthwiseConvolution].matched)
        {
            newModel.append(replaceMap["DepthwiseConvolution"]);
        }
        else if ((*it)[flatten].matched)
        {
            newModel.append(replaceMap["flatten"]);
        }
        else if ((*it)[flattenParam].matched)
        {
            newModel.append(replaceMap["flattenParam"]);
        }
        else if ((*it)[output].matched)
        {
            newModel.append(replaceMap["output"]);
        }
    }
    newModel.append(modelData + lastPos, size - lastPos);
    return newModel;
}

template<itvcvAnalyzerType analyzerType>
std::unique_ptr<IAnalyzer<analyzerType>> CreateTensorRTAnalyzer(
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters)
{
    return std::make_unique<AnalyzerTensorRT<analyzerType>>(error, parameters);
}

/// Instantiate all possible implementations
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerClassification>> CreateTensorRTAnalyzer<itvcvAnalyzerClassification>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerSSD>> CreateTensorRTAnalyzer<itvcvAnalyzerSSD>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerHumanPoseEstimator>> CreateTensorRTAnalyzer<itvcvAnalyzerHumanPoseEstimator>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerMaskSegments>> CreateTensorRTAnalyzer<itvcvAnalyzerMaskSegments>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerSiamese>> CreateTensorRTAnalyzer<itvcvAnalyzerSiamese>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerPointDetection>> CreateTensorRTAnalyzer<itvcvAnalyzerPointDetection>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

template
typename InferenceResultTraits<itvcvAnalyzerPointDetection>::AsyncInferenceResult_t AnalyzerTensorRT<itvcvAnalyzerPointDetection>::AsyncInfer(
    const std::vector<ItvCv::Frame>& imgs, InferenceCounter counter);

} // InferenceWrapper
