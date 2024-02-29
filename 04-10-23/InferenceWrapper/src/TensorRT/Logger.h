#ifndef TRTLOGGER_H
#define TRTLOGGER_H

#include <ItvCvUtils/Log.h>

#include <NvInfer.h>

namespace InferenceWrapper
{
class TrtLogger final : public nvinfer1::ILogger
{
private:
    TrtLogger(ITV8::ILogger* logger);
    ~TrtLogger();

public:
    TrtLogger(const TrtLogger&) = delete;
    void operator=(const TrtLogger&) = delete;

    static TrtLogger& GetInstance(ITV8::ILogger* logger);
    void log(Severity severity, const char* msg) noexcept override;

private:
    ITV8::ILogger* m_logger;
};
}

#endif