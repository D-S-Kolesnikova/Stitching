#ifndef TRTERRORRECORDER_H
#define TRTERRORRECORDER_H

#include <ItvCvUtils/Log.h>

#include <NvInfer.h>

#include <array>
#include <mutex>

namespace InferenceWrapper
{
constexpr std::size_t ERROR_RECORDER_BUFFER_SIZE = 64;

class ErrorRecorder final : public nvinfer1::IErrorRecorder
{
private:
    using ErrorPair = std::pair<nvinfer1::ErrorCode, ErrorDesc>;

public:
    ErrorRecorder(ITV8::ILogger* logger);

    bool reportError(nvinfer1::ErrorCode val, ErrorDesc desc) noexcept override;

    int32_t getNbErrors() const noexcept override;
    nvinfer1::ErrorCode getErrorCode(int32_t errorIdx) const noexcept override;
    ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override;

    bool hasOverflowed() const noexcept override;

    RefCount incRefCount() noexcept override;
    RefCount decRefCount() noexcept override;

    void clear() noexcept override;

private:
    ITV8::ILogger* m_logger{ nullptr };
    RefCount m_refCount{ 0 };
    std::size_t m_totalErrors{ 0 };
    
    std::array<ErrorPair, ERROR_RECORDER_BUFFER_SIZE> m_errors;
    mutable std::mutex m_errorsMutex;
};
}

#endif