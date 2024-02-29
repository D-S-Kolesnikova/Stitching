#ifndef ITVCVUTILS_LOG_H
#define ITVCVUTILS_LOG_H

#include <ItvCvUtils/ItvCvUtils.h>

#include <ItvSdk/include/IErrorService.h>

#include <memory>
#include <sstream>

namespace ItvCv
{
namespace Utils
{
ITVCV_UTILS_API std::shared_ptr<ITV8::ILogger> CreateStreamLogger(std::ostream& stream, uint32_t logLevel);
ITVCV_UTILS_API std::shared_ptr<ITV8::ILogger> CreateFileLogger(const std::string& filename, uint32_t logLevel);

inline const char* GetLogLevelName(uint32_t level)
{
    switch (level)
    {
    case ITV8::LOG_DEBUG:
        return "DEBUG";
    case ITV8::LOG_INFO:
        return "INFO";
    case ITV8::LOG_WARNING:
        return "WARNING";
    case ITV8::LOG_ERROR:
        return "ERROR";
    default:
        return "OFF";
    }
}

} // namespace Utils
} // namespace ItvCv

/// Logs a message at specified level to logger. This macro checks for
/// validity of passed logger pointer and level.
#define ITVCV_LOG(logger, level, formatString)                \
    do                                                        \
    {                                                         \
        ITV8::ILogger* const mylogger = logger;               \
        if (!mylogger || mylogger->GetLogLevel() > level)     \
            break;                                            \
        std::ostringstream s;                                 \
        s << formatString;                                    \
        mylogger->Log(level, ITV8_LINEINFO, s.str().c_str()); \
    } while (false)

/// Same as ITVCV_LOG with the difference that it catches std::exception
// objects during string formatting.
#define ITVCV_LOG_SAFE(logger, level, formatString)                                                   \
    do                                                                                                \
    {                                                                                                 \
        ITV8::ILogger* const mylogger = logger;                                                       \
        if (!mylogger || mylogger->GetLogLevel() > level)                                             \
            break;                                                                                    \
        try                                                                                           \
        {                                                                                             \
            std::ostringstream s;                                                                     \
            s << formatString;                                                                        \
            mylogger->Log(level, ITV8_LINEINFO, s.str().c_str());                                     \
        }                                                                                             \
        catch (const std::exception& e)                                                               \
        {                                                                                             \
            mylogger->Log(level, ITV8_LINEINFO, "Unexpected exception while formatting log message"); \
            mylogger->Log(level, ITV8_LINEINFO, e.what());                                            \
            mylogger->Log(level, ITV8_LINEINFO, #formatString);                                       \
        }                                                                                             \
    } while (false)

/// Same as ITVCV_LOG with the difference that it prepends "this=<this>" to the log message
#define ITVCV_THIS_LOG(logger, level, formatString) \
    do \
    { \
        ITV8::ILogger* const mylogger = logger; \
        if (!mylogger || mylogger->GetLogLevel() > level) \
            break; \
        std::ostringstream s; \
        s << "this=" << this << "; " << formatString; \
        mylogger->Log(level, ITV8_LINEINFO, s.str().c_str()); \
    } while(false)

/// Same as ITVCV_THIS_LOG with the difference that it catches std::exception
// objects during string formatting.
#define ITVCV_THIS_LOG_SAFE(logger, level, formatString) \
    do \
    { \
        ITV8::ILogger* const mylogger = logger; \
        if (!mylogger || mylogger->GetLogLevel() > level) \
            break; \
        try \
        { \
            std::ostringstream s; \
            s << "this=" << this << "; " << formatString; \
            mylogger->Log(level, ITV8_LINEINFO, s.str().c_str()); \
        } \
        catch (const std::exception& e) \
        { \
            mylogger->Log(level, ITV8_LINEINFO, "Unexpected exception while formatting log message"); \
            mylogger->Log(level, ITV8_LINEINFO, e.what()); \
            mylogger->Log(level, ITV8_LINEINFO, #formatString); \
        } \
    } while(false)

#endif // ITVCVUTILS_LOG_H
