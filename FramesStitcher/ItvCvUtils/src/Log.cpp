#include <ItvCvUtils/Log.h>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>

#include <fstream>
#include <mutex>
#include <thread>
#include <memory>

namespace
{
std::ostream& PutDate(std::ostream& os)
{
    namespace bpt = boost::posix_time;
    auto current = bpt::microsec_clock::local_time();
    auto cdate = current.date();
    os << cdate.year() << '-' << std::setfill('0')
       << std::setw(2) << cdate.month().as_number() << '-'
       << std::setfill('0') << std::setw(2) << cdate.day();

    return os;
}

std::ostream& PutTime(std::ostream& os)
{
    namespace bpt = boost::posix_time;
    auto current = bpt::microsec_clock::local_time();
    auto ctime = current.time_of_day();
    os << std::setfill('0') << std::setw(2) << ctime.hours() << ':'
       << std::setfill('0') << std::setw(2) << ctime.minutes() << ':'
       << std::setfill('0') << std::setw(2) << ctime.seconds() << '.'
       << std::setfill('0') << std::setw(3) << ctime.fractional_seconds() / (bpt::ptime::time_duration_type::traits_type::res_adjust() / 1000);

    return os;
}

const char* MapLogLevelToString(ITV8::int32_t logLevel)
{
    switch (logLevel)
    {
    case ITV8::LOG_ERROR:
        return "ERROR";
    case ITV8::LOG_WARNING:
        return "WARN";
    case ITV8::LOG_INFO:
        return "INFO";
    case ITV8::LOG_DEBUG:
        return "DEBUG";
    default:
        return "";
    }
}

std::ostream& PutBasicInfo(std::ostream& stream, uint32_t level)
{
    stream << '[' << std::this_thread::get_id() << "] ";
    PutDate(stream);
    stream << "; ";
    PutTime(stream);
    stream << "; ";
    stream << MapLogLevelToString(level) << "; ";
    return stream;
}

class StreamLogger: public ITV8::ILogger
{
    ITV8_BEGIN_CONTRACT_MAP()
        ITV8_CONTRACT_ENTRY(ITV8::IContract)
        ITV8_CONTRACT_ENTRY(ITV8::ILogger)
    ITV8_END_CONTRACT_MAP()

public:
    StreamLogger(std::ostream& stream, uint32_t logLevel)
        : m_stream(stream)
        , m_logLevel(logLevel)
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
        std::lock_guard<std::mutex> lock(m_mutex);
        PutBasicInfo(m_stream, level) << message << std::endl;
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
    std::mutex m_mutex;
    std::ostream &m_stream;
    uint32_t m_logLevel;
};

class FileLogger: public ITV8::ILogger
{
    ITV8_BEGIN_CONTRACT_MAP()
        ITV8_CONTRACT_ENTRY(ITV8::IContract)
        ITV8_CONTRACT_ENTRY(ITV8::ILogger)
    ITV8_END_CONTRACT_MAP()

public:
    FileLogger(const std::string& filename, uint32_t level)
        : m_file(filename, std::ios_base::app)
        , m_logger(m_file, level)
    {
        if (!m_file.is_open())
        {
            std::ostringstream oss;
            oss << "Couldn't open log file: " << filename;
            throw std::runtime_error(oss.str());
        }
    }

    uint32_t GetLogLevel() const override
    {
        return m_logger.GetLogLevel();
    }

    void Log(uint32_t level, const char* message) override
    {
        m_logger.Log(level, message);
    }

    void Log(uint32_t level, const char* file, uint32_t line, const char* function, const char* message) override
    {

        m_logger.Log(level, file, line, function, message);
    }

private:
    std::ofstream m_file;
    StreamLogger m_logger;
};

}

namespace ItvCv
{
namespace Utils
{
std::shared_ptr<ITV8::ILogger> CreateStreamLogger(std::ostream& stream, uint32_t logLevel)
{
    return std::make_shared<StreamLogger>(stream, logLevel);
}

std::shared_ptr<ITV8::ILogger> CreateFileLogger(const std::string& filename, uint32_t logLevel)
{
    return std::make_shared<FileLogger>(filename, logLevel);
}

}
}
