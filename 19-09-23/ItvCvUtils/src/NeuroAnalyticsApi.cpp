#include <ItvCvUtils/NeuroAnalyticsApi.h>

#include <boost/format.hpp>

#include <string>
#include <map>
#include <vector>
#include <algorithm>


namespace ItvCv
{
namespace Utils
{

uint32_t ItvCvLogLevelToITV8(itvcvLogSeverity severity)
{
    switch (severity)
    {
    case itvcvLogError:
        return ITV8::LOG_ERROR;
    case itvcvLogInfo:
        return ITV8::LOG_INFO;
    case itvcvLogWarning:
        return ITV8::LOG_WARNING;
    case itvcvLogDebug:
        return ITV8::LOG_DEBUG;
    default:
        throw std::logic_error("Unknown itvcvLogSeverity");
    }
}

itvcvLogSeverity ITV8LogLevelToItvCv(uint32_t level)
{
    switch (level)
    {
    case ITV8::LOG_ERROR:
        return itvcvLogError;
    case ITV8::LOG_INFO:
        return itvcvLogInfo;
    case ITV8::LOG_WARNING:
        return itvcvLogWarning;
    case ITV8::LOG_DEBUG:
        return itvcvLogDebug;
    default:
        throw std::logic_error("Unknown ITV8 log level");
    }
}

class LogCallbackWrapper: public ITV8::ILogger
{
    ITV8_BEGIN_CONTRACT_MAP()
        ITV8_CONTRACT_ENTRY(ITV8::IContract)
        ITV8_CONTRACT_ENTRY(ITV8::ILogger)
    ITV8_END_CONTRACT_MAP()

public:
    LogCallbackWrapper(itvcvLogCallback_t callback, void* userData, itvcvLogSeverity logSeverity)
        : m_callback(callback)
        , m_userData(userData)
        , m_logLevel(ItvCvLogLevelToITV8(logSeverity))
    {}

    uint32_t GetLogLevel() const override
    {
        return m_logLevel;
    }

    void Log(uint32_t level, const char* message) override
    {
        if (level != ITV8::LOG_ERROR && !CheckLevel(level))
        {
            return;
        }
        m_callback(m_userData, message, ITV8LogLevelToItvCv(level));
    }

    void Log(uint32_t level, const char* file, uint32_t line, const char* function, const char* message) override
    {
        if (level != ITV8::LOG_ERROR && !CheckLevel(level))
        {
            return;
        }
        auto formattedMessage = boost::str(
            (boost::format("%1%(%2%): %3% : %4%") % (!file ? "" : file) % line % (!function ? "" : function)
             % (!message ? "" : message)));
        Log(level, formattedMessage.c_str());
    }

    void Destroy()
    {
        delete this;
    }

private:
    bool CheckLevel(uint32_t level)
    {
        return m_logLevel <= level;
    }

private:
    itvcvLogCallback_t m_callback;
    void* m_userData;
    uint32_t m_logLevel;
};

ITVCV_UTILS_API std::shared_ptr<ITV8::ILogger> CreateLogCallbackWrapper(
    itvcvLogCallback_t callback,
    void* userData,
    itvcvLogSeverity logSeverity)
{
    return std::make_shared<LogCallbackWrapper>(callback, userData, logSeverity);
}
}
}
