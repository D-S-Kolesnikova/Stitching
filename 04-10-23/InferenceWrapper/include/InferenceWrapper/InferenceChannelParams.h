#ifndef INFERENCE_CHANNEL_PARAMS_H
#define INFERENCE_CHANNEL_PARAMS_H

#include <cstdint>
#include <chrono>

namespace ITV8 { namespace Statistics { struct ISink; } }

namespace InferenceWrapper
{

struct InferenceChannelParams
{
    InferenceChannelParams(
        std::uint32_t maxAsyncQueueLength = 10,
        ITV8::Statistics::ISink* stat_sink = nullptr,
        std::chrono::milliseconds stat_ttl = std::chrono::milliseconds{30000})
        : maxAsyncQueueLength(maxAsyncQueueLength)
        , statSink(stat_sink)
        , statTTL(stat_ttl)
    {
    }

    std::uint32_t maxAsyncQueueLength;
    ITV8::Statistics::ISink* statSink;
    std::chrono::milliseconds statTTL;
};

}

#endif