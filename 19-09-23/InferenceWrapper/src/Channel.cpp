#include "Channel.h"

#include "AnalyzerPool.h"

#include "InferenceHelperFunctions.h"
#include <opencv2/opencv.hpp>

#include <boost/predef.h>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/thread/sync_bounded_queue.hpp>
#include <boost/thread/executors/inline_executor.hpp>

#include <fmt/format.h>
#include <string>

#if BOOST_OS_WINDOWS
#include <Windows.h>

namespace
{
bool sm_environmentPathSet{false};
};
#endif

namespace InferenceWrapper
{
constexpr int MAX_ENV_VARIABLE_SIZE = 32767;

std::string GetDeviceName(itvcvModeType mode, int gpuDeviceNumToUse)
{
    switch (mode)
    {
    case itvcvModeGPU: return "GPU" + std::to_string(gpuDeviceNumToUse);
    case itvcvModeCPU: return "CPU";
    case itvcvModeHDDL: return "IntelHDDL";
    case itvcvModeMULTI: return "IntelMulti";
    case itvcvModeHetero: return "IntelHetero";
    case itvcvModeBalanced: return "IntelBalanced";
    case itvcvModeFPGAMovidius: return "Movidius";
    case itvcvModeIntelGPU: return "IntelGPU";
    case itvcvModeHuaweiNPU: return "HuaweiNPU";
    }
    return std::string();
}

bool CheckNetworkAndDeviceMismatch(ItvCv::ModelRepresentation modelRepresentation, ItvCv::DataType weightType, itvcvModeType mode)
{
    switch (mode)
    {
    case itvcvModeType::itvcvModeGPU:
        return !(modelRepresentation == ItvCv::ModelRepresentation::onnx
              || modelRepresentation == ItvCv::ModelRepresentation::caffe);
    case itvcvModeType::itvcvModeCPU:
    case itvcvModeType::itvcvModeIntelGPU:
        return !(modelRepresentation == ItvCv::ModelRepresentation::onnx
            || modelRepresentation == ItvCv::ModelRepresentation::openvino);
    case itvcvModeType::itvcvModeFPGAMovidius:
    case itvcvModeType::itvcvModeHDDL:
        return
        !((modelRepresentation == ItvCv::ModelRepresentation::openvino && weightType == ItvCv::DataType::FP16)
        || (modelRepresentation == ItvCv::ModelRepresentation::onnx));
    case itvcvModeType::itvcvModeHuaweiNPU:
        return modelRepresentation != ItvCv::ModelRepresentation::ascend;
    case itvcvModeHetero:
    case itvcvModeMULTI:
    case itvcvModeBalanced:
    default:
        throw std::logic_error("unsupported device");
    }
}

template<itvcvAnalyzerType analyzerType>
Channel<analyzerType>::Channel(
    itvcvError& error,
    const InferenceChannelParams& channelParams,
    const EngineCreationParams& engineParams)
    : m_channelParams(channelParams)
    , m_engineParams(std::make_shared<EngineCreationParams>(engineParams))
{
    ITVCV_THIS_LOG(
        m_engineParams->logger,
        ITV8::LOG_DEBUG,
        fmt::format("Async inference max queue length: {}", m_channelParams.maxAsyncQueueLength));

#if !defined(USE_INFERENCE_ENGINE_BACKEND) && \
    !defined(USE_TENSORRT_BACKEND) && \
    !defined(USE_ATLAS300_BACKEND)
    error = itvcvErrorOther;
    throw std::logic_error("No available analyzer backend");
#endif

#if BOOST_OS_WINDOWS
    if (!sm_environmentPathSet)
    {
        char originalPath[MAX_ENV_VARIABLE_SIZE];
        GetEnvironmentVariable("PATH", originalPath, MAX_ENV_VARIABLE_SIZE);
        std::string editedPath = m_engineParams->pluginDir + ";" + originalPath;
        SetEnvironmentVariable("PATH", editedPath.c_str());
        sm_environmentPathSet = true;
    }
#endif

    if (m_engineParams->colorScheme != 8 && m_engineParams->colorScheme != 16)
    {
        error = itvcvErrorColorScheme;
        throw std::runtime_error("Invalid color scheme");
    }

    m_analyzer = GetSharedAnalyzer<analyzerType>(
        *m_engineParams,
        [&error, engineParams=m_engineParams]()
        {
            return AnalyzerFactory::Create<analyzerType>(error, engineParams);
        });

    if (error != itvcvErrorSuccess)
    {
        throw std::runtime_error("Unable to create analyzer");
    }

    auto device = GetDeviceName(m_engineParams->mode, m_engineParams->gpuDeviceNumToUse);
    m_state = m_analyzer->CreateChannelState(
        channelParams.maxAsyncQueueLength,
        channelParams.statSink,
        channelParams.statTTL,
        {
            {"weights_path", m_engineParams->weightsFilePath.c_str()},
            {"analyzer_type", AnalyzerTraits<analyzerType>::Name()},
            {"device", device.c_str()}
        });
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t Channel<analyzerType>::AsyncProcessFrame(
    const ItvCv::Frame& frame)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;

    auto counter = TryStartInference(m_state);
    if (!counter)
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorFullInferenceQueue, {}));
    }

    if (frame.width < 0 || frame.height < 0 || frame.stride < frame.width)
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorFrameSize, {}));
    }

    if (frame.data == 0)
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorFrame, {}));
    }

    auto const frameChannels = frame.stride / frame.width;
    if (!( (frameChannels == 1 && m_engineParams->colorScheme == 8)
        || (frameChannels == 3 && m_engineParams->colorScheme == 16) ))
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorColorScheme, {}));
    }

    try
    {
        return m_analyzer->AsyncInfer(frame, std::move(*counter));
    }
    catch(std::runtime_error err)
    {
        ITVCV_THIS_LOG(m_engineParams->logger, ITV8::LOG_ERROR, "Fail in AsyncInfer: error " << err.what());
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
}

template<itvcvAnalyzerType analyzerType>
typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t Channel<analyzerType>::AsyncProcessFrame(
    const std::vector<ItvCv::Frame>& frames)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;

    auto counter = TryStartInference(m_state);
    if (!counter)
    {
        return boost::make_ready_future(InferenceResult_t(itvcvErrorFullInferenceQueue, {}));
    }
    for (const auto& frame : frames)
    {
        if (frame.width < 0 || frame.height < 0 || frame.stride < frame.width)
        {
            return boost::make_ready_future(InferenceResult_t(itvcvErrorFrameSize, {}));
        }

        if (frame.data == 0)
        {
            return boost::make_ready_future(InferenceResult_t(itvcvErrorFrame, {}));
        }

        auto const frameChannels = frame.stride / frame.width;
        if (!( (frameChannels == 1 && m_engineParams->colorScheme == 8)
            || (frameChannels == 3 && m_engineParams->colorScheme == 16) ))
        {
            return boost::make_ready_future(InferenceResult_t(itvcvErrorColorScheme, {}));
        }
    }
    try
    {
        return m_analyzer->AsyncInfer(frames, std::move(*counter));
    }
    catch(std::runtime_error err)
    {
        ITVCV_THIS_LOG(m_engineParams->logger, ITV8::LOG_ERROR, "Fail in AsyncInfer: error " << err.what());
        return boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {}));
    }
}

template<itvcvAnalyzerType analyzerType>
std::vector< typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t >  Channel<analyzerType>::AsyncProcessSubFrame(
    const ItvCv::Frame& frame, const std::pair<ITV8::int32_t, ITV8::int32_t>& window, const std::pair<ITV8::int32_t, ITV8::int32_t>& steps)
{
    using InferenceResult_t = typename InferenceResultTraits<analyzerType>::InferenceResult_t;
    std::vector< typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t > result;

    if (frame.width < 0 || frame.height < 0 || frame.stride < frame.width)
    {
        result.push_back(boost::make_ready_future(InferenceResult_t(itvcvErrorFrameSize, {})));
        return result;
    }

    if (frame.data == 0)
    {
        result.push_back(boost::make_ready_future(InferenceResult_t(itvcvErrorFrame, {})));
        return result;
    }

    if (window.first > frame.width || window.second > frame.height)
    {
        result.push_back(boost::make_ready_future(InferenceResult_t(itvcvErrorSubFrameWrongSize, {})));
        return result;
    }

    int nScanCols = (frame.width - window.first) / steps.first + 1;
    if ((frame.width - window.first) % steps.first != 0) nScanCols += 1;

    int nScanRows = (frame.height - window.second) / steps.second + 1;
    if ((frame.height - window.second) % steps.second != 0) nScanRows += 1;

    result.reserve(nScanRows * nScanCols);

    try
    {
        int32_t shiftX = 0, shiftY = 0;
        for (auto i = 0; i < nScanRows; ++i)
        {
            shiftX = 0;
            if (shiftY + window.second > frame.height) shiftY = std::max<int32_t>(frame.height - window.second, 0);
            for (auto j = 0; j < nScanCols; ++j)
            {
                if (shiftX + window.first > frame.width) shiftX = std::max<int32_t>(frame.width - window.first, 0);

                auto counter = TryStartInference(m_state);
                if (!counter)
                {
                    result.clear();
                    for (auto k = 0; k < nScanRows * nScanCols; ++k)
                        result.push_back(boost::make_ready_future(InferenceResult_t(itvcvErrorFullInferenceQueue, {})));

                    return result;
                }
                auto offsetedData = frame.data + shiftY * frame.stride + shiftX;
                ItvCv::Frame croppedFrame(window.first, window.second, frame.stride, offsetedData);
                result.push_back(m_analyzer->AsyncInfer(croppedFrame, std::move(*counter)));

                shiftX += steps.first;
            }
            shiftY += steps.second;
        }
        return result;
    }
    catch (std::runtime_error err)
    {
        ITVCV_THIS_LOG(m_engineParams->logger, ITV8::LOG_ERROR, "Fail in AsyncInfer: error " << err.what());

        for (auto i = 0; i < nScanRows * nScanCols; ++i)
            result.push_back(boost::make_ready_future(InferenceResult_t(itvcvErrorOther, {})));

        return result;
    }
}

template<itvcvAnalyzerType analyzerType>
void Channel<analyzerType>::TakeStats(std::chrono::milliseconds period)
{
    m_state->CollectStats(1000. / period.count());
    m_state->PushStats();
    ITVCV_THIS_LOG(m_engineParams->logger, ITV8::LOG_INFO, "inference statistics for " << m_engineParams->weightsFilePath << " [ " << *m_state << " ]" );
}

template class Channel<itvcvAnalyzerClassification>;
template class Channel<itvcvAnalyzerSSD>;
template class Channel<itvcvAnalyzerSiamese>;
template class Channel<itvcvAnalyzerHumanPoseEstimator>;
template class Channel<itvcvAnalyzerMaskSegments>;
template class Channel<itvcvAnalyzerPointDetection>;

template<itvcvAnalyzerType analyzerType>
std::unique_ptr<IInferenceEngine<analyzerType>> IInferenceEngine<analyzerType>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams)
{
    error = itvcvErrorSuccess;
    if (analyzerType != engineParams.net->commonParams.analyzerType)
    {
        error = itvcvError::itvcvErrorAnalyzerTypeMismatch;
        return nullptr;
    }

    if (CheckNetworkAndDeviceMismatch(
            engineParams.net->commonParams.modelRepresentation,
            engineParams.net->commonParams.weightType,
            engineParams.mode))
    {
        error = itvcvError::itvcvErrorNetworkAndDeviceMismatch;
        return nullptr;
    }

    auto analyzer = std::make_unique<Channel<analyzerType>>(
        error,
        channelParams,
        engineParams);
    return std::move(analyzer);
}

template
std::unique_ptr<IInferenceEngine<itvcvAnalyzerClassification>> IInferenceEngine<itvcvAnalyzerClassification>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams);
template
std::unique_ptr<IInferenceEngine<itvcvAnalyzerSSD>> IInferenceEngine<itvcvAnalyzerSSD>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams);
template
std::unique_ptr<IInferenceEngine<itvcvAnalyzerSiamese>> IInferenceEngine<itvcvAnalyzerSiamese>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams);
template
std::unique_ptr<IInferenceEngine<itvcvAnalyzerHumanPoseEstimator>> IInferenceEngine<itvcvAnalyzerHumanPoseEstimator>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams);
template
std::unique_ptr<IInferenceEngine<itvcvAnalyzerMaskSegments>> IInferenceEngine<itvcvAnalyzerMaskSegments>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams);
template
std::unique_ptr<IInferenceEngine<itvcvAnalyzerPointDetection>> IInferenceEngine<itvcvAnalyzerPointDetection>::Create(
    itvcvError& error,
    const EngineCreationParams& engineParams,
    const InferenceChannelParams& channelParams);

template
typename InferenceResultTraits<itvcvAnalyzerPointDetection>::AsyncInferenceResult_t Channel<itvcvAnalyzerPointDetection>::AsyncProcessFrame(
    const std::vector<ItvCv::Frame>& frames);

} // InferenceWrapper
