#pragma once
#include <chrono>

namespace ItvCvUtils {

class StopWatch
{
    std::chrono::steady_clock::time_point start;

public:
    StopWatch()
    {
        Reset();
    }

    void Reset()
    {
        start = std::chrono::steady_clock::now();
    }

    template <class Duration>
    Duration Elapsed() const
    {
        return std::chrono::duration_cast<Duration>(
                std::chrono::steady_clock::now() - start);
    }

    std::chrono::milliseconds ElapsedMs() const
    {
        return Elapsed<std::chrono::milliseconds>();
    }
};

}
