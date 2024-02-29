#include "AnalyzerAtlas300.h"

#include "GenericChannelState.h"
#include "InferenceHelperFunctions.h"

#include <HuaweiAscend/LibraryConvenience.h>

#include <ItvCvUtils/Envar.h>
#include <ItvCvUtils/Log.h>

#include <boost/filesystem.hpp>
#include <boost/dll/shared_library.hpp>
#include <boost/dll/import_mangled.hpp>
#include <boost/utility/in_place_factory.hpp>

#include <future>
#include <fstream>
#include <vector>
#include <codecvt>
#include <string>
#include <locale>
#include <mutex>

namespace fs = boost::filesystem;

namespace InferenceWrapper
{
class AscendChannelState : public GenericChannelState
{
    std::shared_ptr<Ascend::Stream> m_stream;

public:
    template <typename ...Args>
    AscendChannelState(std::shared_ptr<Ascend::Stream>&& stream, Args&&... args)
        : GenericChannelState(std::forward<Args>(args)...)
        , m_stream(std::move(stream))
    {}

    std::shared_ptr<Ascend::Stream> const& Stream() const { return m_stream; }

    cv::Mat m_resizeBufferCache;
};

namespace {

std::once_flag g_loadHuaweiAscendOnce;
boost::dll::shared_library g_libHuaweiAscend;
NAscend::APIv2 g_ascendAPI;
bool g_AIResizeInputOnDVPPWhenPossible;

void LoadHuaweiAscend()
{
    TRY_LOAD_HUAWEI_ASCEND(g_libHuaweiAscend, g_ascendAPI, nullptr, CreateAscend);
    g_ascendAPI.fInitLibrary({
        std::make_pair(NAscend::LibraryParam::AIScheduler, ItvCvUtils::CEnvar::CvAscendScheduler()),
        std::make_pair(NAscend::LibraryParam::AIDumpInput, ItvCvUtils::CEnvar::CvDumpInferenceInput()),
        std::make_pair(NAscend::LibraryParam::AIDumpOutput, ItvCvUtils::CEnvar::CvDumpInferenceOutput()),
        std::make_pair(NAscend::LibraryParam::AIDumpLayers, ItvCvUtils::CEnvar::CvDumpInferenceLayers()),
        std::make_pair(NAscend::LibraryParam::DumpDir, ItvCvUtils::CEnvar::CvDumpDir()),
        std::make_pair(NAscend::LibraryParam::RuntimeDirectoryPrivate, ItvCvUtils::CEnvar::RuntimeDirectoryPrivate()),
        std::make_pair(NAscend::LibraryParam::RuntimeDirectoryShared, ItvCvUtils::CEnvar::RuntimeDirectoryShared())
    });
    g_AIResizeInputOnDVPPWhenPossible = ItvCvUtils::CEnvar::CvAscendResizeInputOnHW();
}

} // anonymous namespace

template<itvcvAnalyzerType analyzerType>
class AnalyzerAtlas300 : public IAnalyzer<analyzerType>
{
public:
    AnalyzerAtlas300 (
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters
    );

    ~AnalyzerAtlas300();

    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer (
        const ItvCv::Frame& frame, InferenceCounter
    ) override;

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
        return std::make_shared<AscendChannelState>(m_atlas300->CreateStream(max_active_requests), max_active_requests, sink, ttl, labels);
    }

private:
    ITV8::ILogger* m_logger;
    std::shared_ptr<EngineCreationParams> m_engineParameters;
    const SampleResizerFunc m_sampleResizeFunc;
    unsigned m_numChannels;
    cv::Size m_inputGeometry{};
    bool m_canResizeOnDevice;
    bool m_usePP;

    std::shared_ptr<Ascend> m_atlas300;

    void Preprocess (
        const cv::Mat& img,
        std::shared_ptr<Ascend::InputData> inferRequest,
        PreprocessContext& ctx
    );
};

template<itvcvAnalyzerType analyzerType>
AnalyzerAtlas300<analyzerType>::AnalyzerAtlas300 (
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters)
    : m_logger(parameters->logger)
    , m_engineParameters(parameters)
    , m_sampleResizeFunc(DefineHostResizeFunction(parameters->net->inputParams.resizePolicy))
    , m_canResizeOnDevice(false)
    , m_usePP(false)
{
    std::call_once(g_loadHuaweiAscendOnce, LoadHuaweiAscend);
    auto const& inputParams = parameters->net->inputParams;
    m_numChannels = inputParams.numChannels;
    m_inputGeometry = cv::Size(inputParams.inputWidth, inputParams.inputHeight);

    std::string const model_name = fs::path(m_engineParameters->weightsFilePath).stem().string();
    Ascend::InputDims shape { 1u, m_numChannels, m_inputGeometry.width, m_inputGeometry.height };

    auto ppMode = Ascend::PreProcessorMode::Off;
    switch (parameters->net->inputParams.pixelFormat)
    {
    case ItvCv::PixelFormat::NV12:
    case ItvCv::PixelFormat::Unspecified:
        ppMode = Ascend::PreProcessorMode::ConvertToNV12AndScale;
        m_usePP = true;
        switch (parameters->net->inputParams.resizePolicy)
        {
        case ItvCv::ResizePolicy::AsIs:
        case ItvCv::ResizePolicy::Unspecified:
            m_canResizeOnDevice = g_AIResizeInputOnDVPPWhenPossible;
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
    if (analyzerType == itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)
    {
        auto poseParameters = boost::get<ItvCv::PoseNetworkParams>(parameters->net->networkParams);
        /// @note we don't support input tensor reshape for now, so input shape change must be
        /// applied during model conversion
        // PrepareInputSizeHPE(HPE_MAX_INPUT_SIZE, m_inputGeometry, poseParameters);
    }
    std::string const passcode_is_not_used;
    Ascend::DynamicAIPPSettings const * dynamic_aipp_is_not_used = nullptr;
    m_atlas300 = g_ascendAPI.fCreateAscend(
        model_name,
        passcode_is_not_used,
        ItvCv::ConsumeWeightsData(parameters->net),
        &shape,
        ppMode,
        dynamic_aipp_is_not_used);
}

template<itvcvAnalyzerType analyzerType>
AnalyzerAtlas300<analyzerType>::~AnalyzerAtlas300()
{
}

template<itvcvAnalyzerType analyzerType>
void AnalyzerAtlas300<analyzerType>::Preprocess (
    const cv::Mat& img,
    std::shared_ptr<Ascend::InputData> inferRequest,
    PreprocessContext& ctx)
{
    cv::Mat sample;

    /// @todo CSC on device by AIPP and/or PreProcess
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

    cv::Mat infer_data;
    auto infer_data_size = m_inputGeometry;
    if (!m_usePP)
    {
        auto buffer = inferRequest->AddRaw(m_inputGeometry.width, m_inputGeometry.height, m_inputGeometry.area() * sample.elemSize());
        infer_data = cv::Mat(m_inputGeometry, sample.type(), buffer);
    }
    else
    {
        if (m_canResizeOnDevice)
            infer_data_size = sample.size();
        auto aligned_buffer = 3 == sample.channels()
            ? inferRequest->AddPackedBGR(infer_data_size.width, infer_data_size.height)
            : inferRequest->AddGray(infer_data_size.width, infer_data_size.height);
        infer_data = cv::Mat(infer_data_size, sample.type(), aligned_buffer.first, aligned_buffer.second.stride);
    }
    if (infer_data_size != sample.size())
        m_sampleResizeFunc(sample, infer_data, ctx);
    else
        sample.copyTo(infer_data);
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AnalyzerAtlas300<analyzerType>::AsyncInfer(const ItvCv::Frame& frame, InferenceCounter counter)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;
    try
    {
        cv::Mat img(
            frame.height,
            frame.width,
            m_engineParameters->colorScheme == 8 ? CV_8UC1 : CV_8UC3,
            (void*) frame.data,
            frame.stride);

        auto* channel = counter.state<AscendChannelState>();
        auto inferRequest = m_atlas300->CreateInferRequest(channel->Stream());

        PreprocessContext ctx{};
        ctx.resizeBufferCache = &channel->m_resizeBufferCache;

        Preprocess(img, inferRequest, ctx);
        ctx.widths.emplace_back(img.cols);
        ctx.heights.emplace_back(img.rows);

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
        m_atlas300->ProcessInfer(
                inferRequest,
    [this, ctx, movable_context{std::move(movable_context)}](const ResultData& res) mutable
        {
            auto& promise = movable_context->promise;
            try
            {
                auto result = ProcessAtlas300Output<analyzerType>(res, ctx, m_engineParameters->net);
                if (result.size() != 1)
                {
                    promise.set_value({itvcvErrorOther, {}});
                }
                else
                {
                    movable_context->counter.template OnInferenceSucceeded<AscendChannelState>();
                    promise.set_value({itvcvErrorSuccess, result[0]});
                }
            }
            catch (const std::logic_error&)
            {
                ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
                          "The analyzer " << analyzerType << " is not implemented for Atlas300");
                promise.set_value({itvcvErrorOther, {}});
            }
        });
        return std::move(future);
    }
    catch(std::exception const& e)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, e.what());
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
}

template<itvcvAnalyzerType analyzerType>
std::unique_ptr<IAnalyzer<analyzerType>> CreateAtlas300Analyzer(
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters)
{
    try
    {
        auto ptr = std::make_unique<AnalyzerAtlas300<analyzerType>>(
                error,
                parameters
            );
        return ptr;
    }
    catch (std::exception const& msg)
    {
        ITVCV_LOG(parameters->logger, ITV8::LOG_ERROR, msg.what());
        error = itvcvErrorOther;
        return nullptr;
    }
}

/// Instantiate all possible implementations
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerClassification>> CreateAtlas300Analyzer<itvcvAnalyzerClassification>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerSSD>> CreateAtlas300Analyzer<itvcvAnalyzerSSD>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerHumanPoseEstimator>> CreateAtlas300Analyzer<itvcvAnalyzerHumanPoseEstimator>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerMaskSegments>> CreateAtlas300Analyzer<itvcvAnalyzerMaskSegments>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerSiamese>> CreateAtlas300Analyzer<itvcvAnalyzerSiamese>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);
template
std::unique_ptr<IAnalyzer<itvcvAnalyzerPointMatcher>> CreateAtlas300Analyzer<itvcvAnalyzerPointMatcher>(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters);

}

