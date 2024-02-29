#include "ReusableRequest.h"

#include <fmt/format.h>

namespace InferenceWrapper
{

constexpr auto BATCH_SIZE = 1;

ReusableRequest::ReusableRequest(
    std::shared_ptr<nvinfer1::ICudaEngine> engine,
    ITV8::ILogger* logger)
    : bufferManager(engine, BATCH_SIZE)
    , context(engine->createExecutionContext())
    , m_logger(logger)
{
    if (!context)
    {
        throw std::runtime_error("TensorRT engine returned an empty context");
    }

    auto err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            fmt::format("{}; Code: {}; Name: {}; Msg: {}",
                "Cuda: cannot create cudaStream_t error",
                err,
                cudaGetErrorName(err),
                cudaGetErrorString(err)));
    }
}

ReusableRequest::~ReusableRequest()
{
    auto err = cudaStreamDestroy(stream);
    if (err != cudaSuccess)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
            "Cuda: cannot destroy cudaStream_t error"
            << "; Code: " << err
            << "; Name: " << cudaGetErrorName(err)
            << "; Msg: " << cudaGetErrorString(err));
    }
}

}