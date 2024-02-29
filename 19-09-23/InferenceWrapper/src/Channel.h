#ifndef _SAMPLEANALYZER_H_
#define _SAMPLEANALYZER_H_

#include "IAnalyzer.h"

#include <InferenceWrapper/InferenceEngine.h>
#include <NetworkInformation/NetworkInformationLib.h>

#include <ItvCvUtils/Log.h>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace InferenceWrapper
{

template<itvcvAnalyzerType analyzerType>
class Channel: public IInferenceEngine<analyzerType>
{
public:
    Channel(
        itvcvError& error,
        const InferenceChannelParams& channelParams,
        const EngineCreationParams& engineParams);

    ITV8::Size GetInputGeometry() const override
    {
        auto geometry = m_analyzer->GetInputGeometry();
        return ITV8::Size(geometry.width, geometry.height);
    }

    void TakeStats(std::chrono::milliseconds period) override;
    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncProcessFrame(const ItvCv::Frame& frame) override;
    typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncProcessFrame(const std::vector<ItvCv::Frame>& frames) override;

    std::vector< typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t > AsyncProcessSubFrame(
        const ItvCv::Frame& bgrFrame,
        const std::pair<ITV8::int32_t, ITV8::int32_t>& window,
        const std::pair<ITV8::int32_t, ITV8::int32_t>& steps) override;

private:
    const InferenceChannelParams m_channelParams;
    std::shared_ptr<EngineCreationParams> m_engineParams;
    std::shared_ptr<IAnalyzer<analyzerType>> m_analyzer;
    std::shared_ptr<IChannelState> m_state;
};
} // InferenceWrapper
#endif
