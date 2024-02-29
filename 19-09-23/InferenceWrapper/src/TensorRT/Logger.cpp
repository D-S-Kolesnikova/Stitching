#include "Logger.h"

#include <map>

namespace
{
const std::map<nvinfer1::ILogger::Severity, uint32_t> g_severityMap
{
    { nvinfer1::ILogger::Severity::kINTERNAL_ERROR, ITV8::LOG_ERROR },
    { nvinfer1::ILogger::Severity::kERROR, ITV8::LOG_ERROR },
    { nvinfer1::ILogger::Severity::kWARNING, ITV8::LOG_DEBUG },
    { nvinfer1::ILogger::Severity::kINFO, ITV8::LOG_DEBUG },
    { nvinfer1::ILogger::Severity::kVERBOSE, ITV8::LOG_DEBUG }
};
}

namespace InferenceWrapper
{
TrtLogger::TrtLogger(ITV8::ILogger* logger) : m_logger(logger) {}

TrtLogger::~TrtLogger() {}

TrtLogger& TrtLogger::GetInstance(ITV8::ILogger* logger)
{
    static TrtLogger trtLogger(logger);
    return trtLogger;
}

void TrtLogger::log(Severity severity, const char* msg) noexcept
{
    const auto itvcvSeverity = g_severityMap.find(severity)->second;
    ITVCV_LOG(m_logger, itvcvSeverity, msg);
}
}