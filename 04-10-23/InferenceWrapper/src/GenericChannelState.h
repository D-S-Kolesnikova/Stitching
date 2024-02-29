#pragma once

#include "IAnalyzer.h"
#include <mutex>

namespace ITV8 { namespace Statistics { struct IMetric; } }

namespace InferenceWrapper {

struct MetricDtor
{
    void operator()(ITV8::Statistics::IMetric* ptr) const;
};
using PMetric = std::unique_ptr<ITV8::Statistics::IMetric, MetricDtor>;

struct GenericStats
{
    uint64_t total_errors;
    double reject_rate;
    double ips;
    uint32_t queue_length;

    struct Accumulator
    {
        uint64_t nr_errors;
        uint32_t nr_rejects;
        uint32_t nr_inferences;
    };
};

class GenericChannelState : public IChannelState
{
    const std::uint32_t m_maxActiveRequests;
    std::uint32_t m_nrStartedInferences;
    GenericStats m_stats;
    GenericStats::Accumulator m_sum;
    PMetric m_metricIPS;
    PMetric m_metricErrors;
    PMetric m_metricRejectRate;
    PMetric m_metricQueueLength;

protected:
    std::mutex m_mutex;
    ITV8::Statistics::ISink* const m_sink;

protected:
    void OnInferenceSucceededLocked();
    virtual void CollectStatsLocked(double period_factor);

private:
    bool TryStartInference() final override;

public:
    GenericChannelState(std::uint32_t max_active_requests, ITV8::Statistics::ISink* sink, std::chrono::milliseconds ttl, std::initializer_list<std::pair<const char*, const char*>> labels);

    void OnInferenceSucceeded()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        OnInferenceSucceededLocked();
    }

    void OnInferenceFailed() final override;

    void CollectStats(double period_factor) final override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        CollectStatsLocked(period_factor);
    }

    void PushStats() const override;
    void Print(std::ostream& os) const override;
};

} // namespace InferenceWrapper

