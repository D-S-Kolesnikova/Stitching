#ifndef _IANALYZER_H_
#define _IANALYZER_H_

#include <InferenceWrapper/InferenceChannelParams.h>
#include <InferenceWrapper/InferenceEngineParams.h>
#include <InferenceWrapper/AnalyzerTraits.h>
#include <ItvCvUtils/Frame.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>
#include <NetworkInformation/NetworkInformationLib.h>

#include <opencv2/opencv.hpp>

#include <functional>
#include <vector>
#include <chrono>
#include <iosfwd>
#include <memory>

#include <boost/optional.hpp>

namespace ITV8 { namespace Statistics { struct ISink; } }

namespace InferenceWrapper
{
class InferenceCounter;
struct IChannelState
{
public:
    virtual ~IChannelState(){}

    virtual void CollectStats(double period_factor) = 0;
    virtual void PushStats() const = 0;
    virtual void Print(std::ostream& os) const = 0;

    virtual void OnInferenceFailed() = 0;

    friend std::ostream& operator <<(std::ostream& os, IChannelState const& stats)
    {
        stats.Print(os);
        return os;
    }

    friend boost::optional<InferenceCounter> TryStartInference(std::shared_ptr<IChannelState> const&);

private:
    virtual bool TryStartInference() = 0;
};

class InferenceCounter
{
    std::shared_ptr<IChannelState> m_state;
public:
    InferenceCounter(std::shared_ptr<IChannelState> const& state) : m_state{state} {}
    InferenceCounter(InferenceCounter&&) = default;
    InferenceCounter& operator=(InferenceCounter&&) = default;
    InferenceCounter(InferenceCounter const&) = delete;
    InferenceCounter& operator=(InferenceCounter const&) = delete;

    ~InferenceCounter()
    {
        if (m_state)
            m_state->OnInferenceFailed();
    }

    template <typename StateImpl, typename ...Args>
    void OnInferenceSucceeded(Args&&... args)
    {
        if (auto impl = std::static_pointer_cast<StateImpl>(m_state))
        {
            m_state.reset();
            impl->OnInferenceSucceeded(std::forward<Args>(args)...);
        }
        else
            abort();
    }

    template <typename StateImpl>
    StateImpl* state() const
    {
        return static_cast<StateImpl*>(m_state.get());
    }
};

inline boost::optional<InferenceCounter> TryStartInference(std::shared_ptr<IChannelState> const& state)
{
    return state->TryStartInference() ? boost::make_optional(InferenceCounter(state)) : boost::none;
}

struct IAnalyzerBase
{
    virtual ~IAnalyzerBase(){}
    virtual cv::Size GetInputGeometry() const = 0;
    virtual std::shared_ptr<IChannelState> CreateChannelState(std::uint32_t max_active_requests, ITV8::Statistics::ISink* sink, std::chrono::milliseconds ttl, std::initializer_list<std::pair<const char*, const char*>> labels) = 0;
};

template<itvcvAnalyzerType analyzerType>
struct IAnalyzer : IAnalyzerBase
{
    virtual typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer(
        const ItvCv::Frame& img, InferenceCounter counter) = 0;
    virtual typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t AsyncInfer(
        const std::vector<ItvCv::Frame>& img, InferenceCounter counter) = 0;
};
} // InferenceWrapper
#endif
