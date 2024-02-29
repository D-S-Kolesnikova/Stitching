#include "GenericChannelState.h"

#include <ItvSdk/include/Statistics.h>
#include <ostream>

namespace InferenceWrapper {

namespace {
const char s_sIPS[] = "ips";
const char s_sERRORS[] = "errors";
const char s_sREJECT_RATE[] = "reject_rate";
const char s_sQUEUE_LENGTH[] = "queue_length";
}

void MetricDtor::operator()(ITV8::Statistics::IMetric* ptr) const
{
    if (ptr)
        ptr->Destroy();
}

GenericChannelState::GenericChannelState(std::uint32_t max_active_requests, ITV8::Statistics::ISink* sink, std::chrono::milliseconds ttl, std::initializer_list<std::pair<const char*, const char*>> labels)
    : m_maxActiveRequests(max_active_requests)
    , m_nrStartedInferences(0u)
    , m_stats{}
    , m_sum{}
    , m_sink(sink)
{
    if (sink)
    {
        m_metricIPS.reset(sink->CreateGaugeMetric(s_sIPS, ttl.count()));
        m_metricErrors.reset(sink->CreateGaugeMetric(s_sERRORS, ttl.count()));
        m_metricRejectRate.reset(sink->CreateGaugeMetric(s_sREJECT_RATE, ttl.count()));
        m_metricQueueLength.reset(sink->CreateGaugeMetric(s_sQUEUE_LENGTH, ttl.count()));
        for (auto const& l : labels)
        {
            m_metricIPS->SetLabel(l.first, l.second);
            m_metricErrors->SetLabel(l.first, l.second);
            m_metricRejectRate->SetLabel(l.first, l.second);
            m_metricQueueLength->SetLabel(l.first, l.second);
        }
    }
}

bool GenericChannelState::TryStartInference()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_nrStartedInferences >= m_maxActiveRequests)
    {
        ++m_sum.nr_rejects;
        return false;
    }
    ++m_nrStartedInferences;
    return true;
}

void GenericChannelState::OnInferenceSucceededLocked()
{
    --m_nrStartedInferences;
    ++m_sum.nr_inferences;
}

void GenericChannelState::OnInferenceFailed()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    --m_nrStartedInferences;
    ++m_sum.nr_errors;
}

void GenericChannelState::CollectStatsLocked(double period_factor)
{
    m_stats = GenericStats{ m_sum.nr_errors, m_sum.nr_rejects * period_factor, m_sum.nr_inferences * period_factor, m_nrStartedInferences };
    m_sum = GenericStats::Accumulator{ m_sum.nr_errors, 0, 0 };
}

void GenericChannelState::PushStats() const
{
    if (!m_sink)
        return;
    m_metricIPS->SetValue(m_stats.ips);
    m_sink->Push(m_metricIPS.get(), ITV8::Statistics::ISink::Reusable_DestroyedByCaller);

    m_metricErrors->SetValue(m_stats.total_errors);
    m_sink->Push(m_metricErrors.get(), ITV8::Statistics::ISink::Reusable_DestroyedByCaller);

    m_metricRejectRate->SetValue(m_stats.reject_rate);
    m_sink->Push(m_metricRejectRate.get(), ITV8::Statistics::ISink::Reusable_DestroyedByCaller);

    m_metricQueueLength->SetValue(m_stats.queue_length);
    m_sink->Push(m_metricQueueLength.get(), ITV8::Statistics::ISink::Reusable_DestroyedByCaller);
}

void GenericChannelState::Print(std::ostream& os) const
{
    os << s_sIPS << ": " << m_stats.ips << "; " << s_sREJECT_RATE << ": " << m_stats.reject_rate << "; "
       << s_sQUEUE_LENGTH << ": " << m_stats.queue_length << " of " << m_maxActiveRequests << "; " << s_sERRORS << ": " << m_stats.total_errors;
}

} // namespace InferenceWrapper
