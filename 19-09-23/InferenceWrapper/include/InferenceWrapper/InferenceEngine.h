#ifndef COMPUTERVISION_INFERENCEENGINE_H
#define COMPUTERVISION_INFERENCEENGINE_H

#include <InferenceWrapper/InferenceChannelParams.h>
#include <InferenceWrapper/InferenceEngineParams.h>
#include <InferenceWrapper/AnalyzerTraits.h>

#include <ItvSdk/include/IErrorService.h>
#include <ItvCvUtils/ItvCvDefs.h>
#include <ItvCvUtils/Frame.h>
#include <ItvCvUtils/DynamicThreadPool.h>

#include <chrono>

#ifdef ITVCV_INFERENCEWRAPPER_EXPORT
#define IW_API ITVCV_API_EXPORT
#else
#define IW_API ITVCV_API_IMPORT
#endif

namespace InferenceWrapper
{

class IW_API IInferenceEngineBase
{
protected:
    virtual ~IInferenceEngineBase() {}

public:
    virtual void TakeStats(std::chrono::milliseconds period) = 0;
    virtual ITV8::Size GetInputGeometry() const = 0;
};

template<itvcvAnalyzerType analyzerType>
struct IW_API IInferenceEngine : public IInferenceEngineBase
{
    virtual ~IInferenceEngine() {}

    using AsyncInferenceResult_t = typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t;
    using PInferenceEngine = std::unique_ptr<IInferenceEngine<analyzerType>>;

    virtual AsyncInferenceResult_t AsyncProcessFrame(const ItvCv::Frame& bgrFrame) = 0;
    virtual AsyncInferenceResult_t AsyncProcessFrame(const std::vector<ItvCv::Frame>& batchFrames) = 0;
    virtual std::vector<AsyncInferenceResult_t> AsyncProcessSubFrame(const ItvCv::Frame& bgrFrame, const std::pair<int, int>& window, const std::pair<int, int>& steps) = 0;

    static PInferenceEngine Create(
        itvcvError& error,
        const EngineCreationParams& engineParams,
        const InferenceChannelParams& channelParams = {});
};

template<itvcvAnalyzerType analyzerType>
using PInferenceEngine = std::unique_ptr<IInferenceEngine<analyzerType>>;

template<itvcvAnalyzerType analyzerType>
inline boost::future<std::pair<itvcvError, PInferenceEngine<analyzerType>>> AsyncCreateInferenceEngine(
    ItvCvUtils::IDynamicThreadPool& threadPool,
    EngineCreationParams engineParams,
    InferenceChannelParams channelParams = {})
{
    return threadPool.PostTask([engineParams = std::move(engineParams), channelParams = std::move(channelParams)] {
        itvcvError error = itvcvErrorSuccess;
        auto engine = IInferenceEngine<analyzerType>::Create(error, engineParams, channelParams);
        return std::make_pair(error, std::move(engine));
    });
}

}

#endif // COMPUTERVISION_INFERENCEENGINE_H
