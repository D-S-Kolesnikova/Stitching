#include "AnalyzerIntelIE.h"
#include "GenericChannelState.h"
#include "InferenceHelperFunctions.h"

#include <InferenceWrapper/InferenceWrapperLibCommon.h>

#include <ItvCvUtils/Uuid.h>
#include <ItvCvUtils/Cpuid.h>
#include <ItvCvUtils/Log.h>

#include <openvino/openvino.hpp>
#include <ie/ie_blob.h>

#include <boost/lexical_cast.hpp>
#include <boost/locale/encoding_utf.hpp>

#include <opencv2/opencv.hpp>

#include <fmt/format.h>

#include <vector>
#include <mutex>
#include <exception>
#include <chrono>
#include <thread>


namespace
{

////// Важно, функция LoadNetwork не потокобезопасна, для этого ввел mutex во время инициализации сети,
    static std::mutex IE_CREATE_ENGINE_MUTEX;
/////////////
const std::map<itvcvModeType, std::string> g_modeToNameMap
{
    { itvcvModeCPU, "CPU" },
    { itvcvModeFPGAMovidius, "MYRIAD" },
    { itvcvModeIntelGPU, "GPU" },
    { itvcvModeHetero, "HETERO" },
    { itvcvModeHDDL, "HDDL" },
    { itvcvModeMULTI, "MULTI" },
};
std::mutex g_coreMutex;
std::mutex g_coreWeakMutex;
std::weak_ptr<ov::Core> g_coreWeak;

//probably need to add into NetworkInformation
const ov::Layout TENSOR_ORDER{ "NCHW" };
}

namespace InferenceWrapper
{

using IntelIEChannelState = GenericChannelState;

// map for monitoring active requests uuid -> (IntelIE::InferRequst::Ptr, done)
struct SAsyncRequest
{
    std::shared_ptr<ov::InferRequest> req;
    bool done {false};
};

namespace IntelIE = InferenceEngine;

template<itvcvAnalyzerType analyzerType>
class CAnalyzerIntelIE: public IAnalyzer<analyzerType>
{
public:
    CAnalyzerIntelIE(itvcvError& error, std::shared_ptr<EngineCreationParams> parameters);

    ~CAnalyzerIntelIE()
    {
        while (true)
        {
            using namespace std::chrono_literals;

            DeleteCompletedRequests();
            {
                std::lock_guard<std::mutex> lock(m_activeRequestsMutex);
                if (m_activeRequests.empty())
                {
                    return;
                }
            }
            std::this_thread::sleep_for(1s);
        }
    }

    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer(const ItvCv::Frame& frame, InferenceCounter) override;
    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer(const std::vector<ItvCv::Frame>& imgs, InferenceCounter counter) override;

    cv::Size GetInputGeometry() const override
    {
        return m_inputGeometry;
    }

    std::shared_ptr<IChannelState> CreateChannelState(
            std::uint32_t max_active_requests,
            ITV8::Statistics::ISink* sink,
            std::chrono::milliseconds ttl,
            std::initializer_list<std::pair<const char*, const char*>> labels) override
    {
        return std::make_shared<IntelIEChannelState>(max_active_requests, sink, ttl, labels);
    }

    std::string GetAppropriateTargetDevice(itvcvModeType mode);

private:
    //change preprocessing for input Tensor only onnx nets
    void InitPreprocessingOnnxModel(ov::preprocess::InputInfo& input) const;
    void initNetwork();
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels, PreprocessContext& ctx);
    void WrapInputLayer(std::vector<cv::Mat>& input_channels, int n, ov::InferRequest req);
    void DeleteCompletedRequests();

private:
    ITV8::ILogger* m_logger;
    std::shared_ptr<EngineCreationParams> m_engineParameters;
    SampleResizerFunc m_sampleResizeFunc;

    int m_numChannels {0};
    cv::Size m_inputGeometry {};
    cv::Mat m_mean {};

    size_t m_batchSize {1};

    std::shared_ptr<ov::Model> m_network;

    ov::CompiledModel m_executableNetwork;
    std::mutex m_executableNetworkMutex;

    std::map<std::string, SAsyncRequest> m_activeRequests;
    std::mutex m_activeRequestsMutex;
    std::shared_ptr<ov::Core> m_core;
};

template<itvcvAnalyzerType analyzerType>
CAnalyzerIntelIE<analyzerType>::CAnalyzerIntelIE(itvcvError& error, std::shared_ptr<EngineCreationParams> parameters)
    : m_logger(parameters->logger)
    , m_engineParameters(parameters)
    , m_sampleResizeFunc(DefineHostResizeFunction(parameters->net->inputParams.resizePolicy))
{
    {
        const std::lock_guard<std::mutex> lock(g_coreWeakMutex);
        m_core = g_coreWeak.lock();
        if (!m_core)
        {
            m_core = std::make_shared<ov::Core>();
            g_coreWeak = m_core;
        }
    }

    if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerSSD
        && m_engineParameters->net->commonParams.architecture == ItvCv::ArchitectureType::Yolo)
    {
        error = itvcvErrorOther;
        throw std::runtime_error("Yolo arch is not supported on Openvino");
    }

    initNetwork();
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t CAnalyzerIntelIE<analyzerType>::AsyncInfer(const ItvCv::Frame& frame, InferenceCounter counter)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;
    if (m_batchSize != BATCHSIZE_ASYNC)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "For async inference batch size of 1 is expected.");
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }

    auto reqPtr = std::make_shared<ov::InferRequest>();
    {
        std::lock_guard<std::mutex> lock1(m_executableNetworkMutex);
        *reqPtr = m_executableNetwork.create_infer_request();
    }
    if (!reqPtr)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Doesnt create  inferRequestPtr");
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }

    cv::Mat cvFrame(
        frame.height,
        frame.width,
        m_engineParameters->colorScheme == 8 ? CV_8UC1 : CV_8UC3,
        (void*) frame.data,
        frame.stride);

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(input_channels, 0, *reqPtr);
    PreprocessContext ctx {};
    Preprocess(cvFrame, input_channels, ctx);
    ctx.widths.emplace_back(cvFrame.cols);
    ctx.heights.emplace_back(cvFrame.rows);

    if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)
    {
        ctx.hpeParams = boost::get<ItvCv::PoseNetworkParams>(m_engineParameters->net->networkParams);
    }

    struct MovableContext
    {
        InferenceCounter counter;
        boost::promise<InferenceResult_t> promise;
        MovableContext(InferenceCounter&& counter) : counter(std::move(counter)) {}
    };
    auto movable_context = std::make_shared<MovableContext>(std::move(counter));
    auto future = movable_context->promise.get_future();

    const auto uuid = ItvCv::Utils::ConvertUuidToString(ItvCv::Utils::GenerateUuid());
    reqPtr->set_callback(
        [
            this,
            movable_context = std::move(movable_context),
            uuid,
            ctx,
            weakReq = std::weak_ptr<ov::InferRequest>(reqPtr)
        ]
        (std::exception_ptr ex) mutable
        {
            if (auto req = weakReq.lock())
            {
                if (ex)
                {
                    ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Openvino: Got exception:");
                    movable_context->promise.set_value({ itvcvErrorInference, {} });
                }
                else
                {
                    auto result = ProcessIntelIEOutput<analyzerType>(*req, BATCHSIZE_ASYNC, ctx, m_engineParameters->net);
                    movable_context->counter.template OnInferenceSucceeded<IntelIEChannelState>();
                    movable_context->promise.set_value({ itvcvErrorSuccess, result[0] });
                }
            }
            else
            {
                ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Openvino: request ptr is empty");
                movable_context->promise.set_value({ itvcvErrorInference, {} });
            }

            {
                std::lock_guard<std::mutex> l(m_activeRequestsMutex);
                m_activeRequests[uuid].done = true;
            }
        });

    {
        std::lock_guard<std::mutex> lock2(m_activeRequestsMutex);
        m_activeRequests[uuid].req = reqPtr;
    }

    reqPtr->start_async();

    DeleteCompletedRequests();
    return std::move(future);
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t CAnalyzerIntelIE<analyzerType>::AsyncInfer(const std::vector<ItvCv::Frame>& imgs, InferenceCounter counter)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;
    return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
}

template<itvcvAnalyzerType analyzerType>
std::string CAnalyzerIntelIE<analyzerType>::GetAppropriateTargetDevice(itvcvModeType mode)
{
    auto devName = g_modeToNameMap.find(mode)->second;
    auto cpuid = ItvCvUtils::GetCpuid();
    if (cpuid->AVX2() || cpuid->AVX512F() || cpuid->AVX512PF()  || cpuid->AVX512ER() || cpuid->AVX512CD())
    {
        return devName;
    }
    return "N/A";
}

template <itvcvAnalyzerType analyzerType>
void CAnalyzerIntelIE<analyzerType>::InitPreprocessingOnnxModel(ov::preprocess::InputInfo& input) const
{
    //need if nets is onnx, because mean and scale
    input.tensor().set_element_type(ov::element::f32);

    switch (m_engineParameters->net->inputParams.pixelFormat)
    {
    case ItvCv::PixelFormat::RGB:
        input.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
        break;
    case ItvCv::PixelFormat::NV12:
        input.preprocess().convert_color(ov::preprocess::ColorFormat::NV12_SINGLE_PLANE);
        break;
    case ItvCv::PixelFormat::BGR:
        break;
    default:
        throw std::runtime_error(fmt::format("Unsupported PixelFormat:{}", m_engineParameters->net->inputParams.pixelFormat));
    }

    if (m_engineParameters->net->inputParams.normalizationValues.size() != m_engineParameters->net->inputParams.numChannels)
    {
        throw std::runtime_error("Model doesn\'t contain normalization values");
    }
    std::vector<float> mean;
    std::vector<float> scale;
    // mean and scale values are different when using IntelIE and TensorRT for preprocess
    // current values from metaData are specialized for TensorRT
    for (const auto& normValue : m_engineParameters->net->inputParams.normalizationValues)
    {
        // transform values for IntelIE
        const auto scaleValue = 1.f / normValue.scale;
        const auto meanValue = normValue.mean * scaleValue;
        mean.emplace_back(meanValue);
        scale.emplace_back(scaleValue);
    }

    input.preprocess().mean(mean);
    input.preprocess().scale(scale);
}

template<itvcvAnalyzerType analyzerType>
void CAnalyzerIntelIE<analyzerType>::initNetwork()
{
    const auto targetDevice = GetAppropriateTargetDevice(m_engineParameters->mode);
    if (targetDevice == "N/A")
    {
        throw std::runtime_error("This device is not supported in this platform. mode: " + std::to_string(static_cast<int>(m_engineParameters->mode)));
    }

    auto weightsData = ItvCv::ConsumeWeightsData(m_engineParameters->net);
    if (m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::openvino)
    {

        ov::Tensor weightsTensor(ov::element::u8, ov::Shape({ size_t(weightsData.size()) }));
        std::memcpy(weightsTensor.data(), weightsData.data(), weightsData.size());
        m_network = m_core->read_model(m_engineParameters->net->modelData, weightsTensor);
    }
    else
    {
        /// for onnx/ascent weights data is model
        m_network = m_core->read_model(weightsData, ov::Tensor{});
    }
    ov::preprocess::PrePostProcessor inputOutputInfo(m_network);
    auto& inputInfo = inputOutputInfo.input();
    inputInfo.model().set_layout(TENSOR_ORDER);
    inputInfo.tensor().set_layout(TENSOR_ORDER).set_element_type(ov::element::u8);
    inputInfo.tensor().set_color_format(ov::preprocess::ColorFormat::BGR);
    if(m_engineParameters->net->commonParams.modelRepresentation == ItvCv::ModelRepresentation::onnx)
    {
        InitPreprocessingOnnxModel(inputInfo);
    }
    m_network = inputOutputInfo.build();
    ov::set_batch(m_network, m_batchSize);

    auto inputShapes = m_network->input().get_shape();

    m_numChannels = m_engineParameters->net->inputParams.numChannels;
    const int width = m_engineParameters->net->inputParams.inputWidth;
    const int height = m_engineParameters->net->inputParams.inputHeight;
    m_inputGeometry = cv::Size(width, height);

    if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)
    {
        auto poseParameters = boost::get<ItvCv::PoseNetworkParams>(m_engineParameters->net->networkParams);

        inputShapes[ov::layout::height_idx(TENSOR_ORDER)] = m_inputGeometry.height;
        inputShapes[ov::layout::width_idx(TENSOR_ORDER)] = m_inputGeometry.width;
        m_network->reshape(inputShapes);


    }
    else if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerMaskSegments)
    {
        auto& netWidth = inputShapes[ov::layout::width_idx(TENSOR_ORDER)];
        auto& netHeight = inputShapes[ov::layout::height_idx(TENSOR_ORDER)];
        if (width != netWidth || height != netHeight)
        {
            netWidth = width;
            netHeight = height;
            m_network->reshape(inputShapes);
        }
    }


    // без mutex может быть ошибка NOT found Input layer
    {
        std::lock_guard<std::mutex> l(g_coreMutex);
        m_executableNetwork = m_core->compile_model(m_network, targetDevice, ov::inference_num_threads(std::thread::hardware_concurrency()));
    }
}

template<itvcvAnalyzerType analyzerType>
void CAnalyzerIntelIE<analyzerType>::Preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels, PreprocessContext& ctx)
{
    cv::Mat sample;

    if (img.channels() == 3 && m_numChannels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && m_numChannels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && m_numChannels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && m_numChannels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat resultImg;
    if (sample.size() == m_inputGeometry)
    {
        resultImg = sample;
    }
    else
    {
        resultImg = cv::Mat(m_inputGeometry, sample.type());
        m_sampleResizeFunc(sample, resultImg, ctx);
    }

    auto matType = CV_8UC(m_numChannels);
    if(m_network->input().get_element_type() == ov::element::f32)
    {
        matType = CV_32FC(m_numChannels);
    }

    resultImg.convertTo(resultImg, matType);

    cv::split(resultImg, input_channels);
}

template<itvcvAnalyzerType analyzerType>
void CAnalyzerIntelIE<analyzerType>::WrapInputLayer(std::vector<cv::Mat>& input_channels, int n, ov::InferRequest req)
{
    const int channels = m_numChannels;
    const int height = m_inputGeometry.height;
    const int width = m_inputGeometry.width;

    auto inputTensor = req.get_input_tensor();

    auto offset = 0;
    for (int i = 0; i < channels; ++i)
    {
        if (m_network->input().get_element_type() == ov::element::f32)
        {
            input_channels.emplace_back(height, width, CV_32FC1, inputTensor.data<float>() + (n * width * height * channels) + offset);
        }
        else
        {
           input_channels.emplace_back(height, width, CV_8UC1, inputTensor.data<uint8_t>() + (n * width * height * channels) + offset);
        }
        offset += (width * height);
    }
}

template<itvcvAnalyzerType analyzerType>
void CAnalyzerIntelIE<analyzerType>::DeleteCompletedRequests()
{
    std::lock_guard<std::mutex> l(m_activeRequestsMutex);
    for (auto it = m_activeRequests.begin(); it != m_activeRequests.end();)
    {
        if (it->second.done)
        {
            it = m_activeRequests.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

template<itvcvAnalyzerType analyzerType>
std::unique_ptr<IAnalyzer<analyzerType>> CreateIntelIEAnalyzer(
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters)
{
    return std::make_unique<CAnalyzerIntelIE<analyzerType>>(error, parameters);
}

/// Instantiate all possible implementations
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerClassification>> CreateIntelIEAnalyzer<itvcvAnalyzerClassification>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

template
std::unique_ptr<IAnalyzer<itvcvAnalyzerSSD>> CreateIntelIEAnalyzer<itvcvAnalyzerSSD>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

template
std::unique_ptr<IAnalyzer<itvcvAnalyzerHumanPoseEstimator>> CreateIntelIEAnalyzer<itvcvAnalyzerHumanPoseEstimator>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

template
std::unique_ptr<IAnalyzer<itvcvAnalyzerMaskSegments>> CreateIntelIEAnalyzer<itvcvAnalyzerMaskSegments>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

template
std::unique_ptr<IAnalyzer<itvcvAnalyzerSiamese>> CreateIntelIEAnalyzer<itvcvAnalyzerSiamese>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

template
std::unique_ptr<IAnalyzer<itvcvAnalyzerPointMatcher>> CreateIntelIEAnalyzer<itvcvAnalyzerPointMatcher>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

} // InferenceWrapper
