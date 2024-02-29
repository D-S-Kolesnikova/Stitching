#include <ItvCvUtils/FpsCounter.h>

#include <ItvSdk/include/IErrorService.h>

#include <boost/format.hpp>

#include <chrono>
#include <mutex>

namespace
{
class FpsCounter: public ItvCvUtils::IFpsCounter
{
public:
    FpsCounter(const std::string& name, void* id, ITV8::ILogger* logger): m_logger(logger)
    {
        if (m_logger)
        {
            if (id)
            {
                m_logPrefix = boost::str(boost::format("[%s:%08p]") % name % id);
            }
            else
            {
                m_logPrefix = std::string("[") + name + "]";
            }
        }
    }

    void Reset() override
    {
        std::lock_guard<std::mutex> l(m_mutex);
        m_count = 0;
        m_t0 = std::chrono::high_resolution_clock::now();
        m_fps = 0;
    }

    void Increment() override
    {
        std::lock_guard<std::mutex> l(m_mutex);
        ++m_count;
        if (m_count % REPORT_RATE == 0)
        {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - m_t0).count();
            m_fps = 1e3 * m_count / ms;
            if (m_logger)
            {
                m_logger->Log(ITV8::LOG_INFO, boost::str(boost::format("%s:  FPS=%04.2f") % m_logPrefix % m_fps).c_str());
            }
            m_t0 = t1;
            m_count = 0;
        }
    }

    double Fps() const override
    {
        std::lock_guard<std::mutex> l(m_mutex);
        return m_fps;
    }

    void ForceCalcFps() override
    {
        std::lock_guard<std::mutex> l(m_mutex);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - m_t0).count();
        if (ms != 0)
        {
            m_fps = 1e3 * m_count / ms;
        }
    }

    ~FpsCounter() override = default;

private:
    static constexpr auto REPORT_RATE = 300;

    ITV8::ILogger* m_logger;
    std::string m_logPrefix;

    mutable std::mutex m_mutex;
    std::chrono::high_resolution_clock::time_point m_t0;
    int m_count = 0;
    double m_fps = 0;
};
}

namespace ItvCvUtils
{
std::unique_ptr<IFpsCounter> CreateFpsCounter(const std::string& name, void* id, ITV8::ILogger* logger)
{
    return std::make_unique<FpsCounter>(name, id, logger);
}
}
