#ifndef REUSABLE_REQUEST_H
#define REUSABLE_REQUEST_H

#include "Buffers.h"

#include <ItvCvUtils/Log.h>

#include <NvInferRuntime.h>

#include <memory>

namespace InferenceWrapper
{

class ReusableRequest
{
public:
    ReusableRequest(
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        ITV8::ILogger* logger);

    ~ReusableRequest();

public:
    cudaStream_t stream;
    CBufferManager bufferManager;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

private:
    ITV8::ILogger* m_logger;
};

}

#endif