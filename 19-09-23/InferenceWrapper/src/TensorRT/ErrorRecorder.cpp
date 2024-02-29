#include "ErrorRecorder.h"

namespace InferenceWrapper
{
ErrorRecorder::ErrorRecorder(ITV8::ILogger* logger)
    : m_logger(logger)
{
    clear();
}

bool ErrorRecorder::reportError(nvinfer1::ErrorCode val, ErrorDesc desc) noexcept
{
    if (val == nvinfer1::ErrorCode::kSUCCESS)
    {
        return false;
    }

    {
        std::lock_guard<std::mutex> l(m_errorsMutex);
        m_errors[m_totalErrors++ % ERROR_RECORDER_BUFFER_SIZE] = ErrorPair{val, desc};
    }

    ITVCV_THIS_LOG(m_logger, ITV8::LOG_ERROR,
        "Err: " << static_cast<int>(val) << "; Desc: " << desc);

    /**
     * TODO return true if it is unrecoverable error (if it is worth it).
     * possible use if and only if we could destroy CUcontext and create a new one 
     * then create all cuda objects once again inside of the new CUcontext
     */
    return false;
}

int32_t ErrorRecorder::getNbErrors() const noexcept
{
    std::lock_guard<std::mutex> l(m_errorsMutex);
    return m_totalErrors;
}

nvinfer1::ErrorCode ErrorRecorder::getErrorCode(int32_t errorIdx) const noexcept
{
    std::lock_guard<std::mutex> l(m_errorsMutex);
    return m_errors[errorIdx].first;
}

ErrorRecorder::ErrorDesc ErrorRecorder::getErrorDesc(int32_t errorIdx) const noexcept
{
    std::lock_guard<std::mutex> l(m_errorsMutex);
    return m_errors[errorIdx].second;
}

bool ErrorRecorder::hasOverflowed() const noexcept
{
    /**
     * NOTE ignored as it could cause extra thread blocking
    
        std::lock_guard<std::mutex> l(m_errorsMutex);
        return m_totalErrors > ERROR_RECORDER_BUFFER_SIZE;
     */
    return false;
}

ErrorRecorder::RefCount ErrorRecorder::incRefCount() noexcept
{
    std::lock_guard<std::mutex> l(m_errorsMutex);
    return ++m_refCount;
}

ErrorRecorder::RefCount ErrorRecorder::decRefCount() noexcept
{
    std::lock_guard<std::mutex> l(m_errorsMutex);
    return --m_refCount;
}

void ErrorRecorder::clear() noexcept
{
    std::lock_guard<std::mutex> l(m_errorsMutex);
    m_errors.fill(ErrorPair{ nvinfer1::ErrorCode::kSUCCESS, ""});
    m_totalErrors = 0;
}
}