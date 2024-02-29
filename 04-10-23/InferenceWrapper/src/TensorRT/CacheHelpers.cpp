#include "CacheHelpers.h"

#include "TensorRT/Logger.h"

#include <ItvCvUtils/Envar.h>
#include <cryptoWrapper/cryptoWrapperLib.h>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include <boost/atomic/atomic.hpp>
#include <boost/format.hpp>
#include <boost/uuid/detail/md5.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/detail/utf8_codecvt_facet.hpp>
#include <boost/filesystem/operations.hpp>

#include <mutex>
#include <sstream>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <stdlib.h>
#endif

namespace
{

//! I did not made a variable to store this value because we can get a failure from a recoverable error
boost::filesystem::path GetCachingPath()
{
    boost::format cudaMsgFormat("%1%; Code:%2%; Name:%3%; Msg:%4%.");
    cudaError_t cudaError;

    int cudartVer{ 0 };
    cudaError = cudaRuntimeGetVersion(&cudartVer);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error(
            str(cudaMsgFormat
                % "Can not retrive cuda runtime version."
                % cudaError
                % cudaGetErrorName(cudaError)
                % cudaGetErrorString(cudaError)));
    }

    int cudaVer{ 0 };
    cudaError = cudaDriverGetVersion(&cudaVer);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error(
            str(cudaMsgFormat
                % "Can not retrive cuda driver version."
                % cudaError
                % cudaGetErrorName(cudaError)
                % cudaGetErrorString(cudaError)));
    }

    boost::filesystem::path path(ItvCvUtils::CEnvar::GpuCacheDir());
    if (!path.empty())
    {
        path = path.append("NeuroSDK").append(str(boost::format("%1%-%2%-%3%") % NV_TENSORRT_VERSION % cudartVer % cudaVer));
        if (boost::filesystem::is_directory(path) || boost::filesystem::create_directories(path))
        {
            return path;
        }
    }

    return {};
}
}

namespace InferenceWrapper
{
FileCacheHelper::FileCacheHelper(ITV8::ILogger* logger)
    : m_logger(logger)
{
}

bool FileCacheHelper::Save(
    const std::string& fileName,
    std::shared_ptr<nvinfer1::ICudaEngine> engine)
{
    auto cachingPath = GetCachingPath();
    if (cachingPath.empty())
    {
        ITVCV_LOG(m_logger, ITV8::LOG_INFO, "GPU cache path is not available.");
        return false;
    }

    auto trtBuffer = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
    if (!trtBuffer)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Failed to serialize engine to trtBuffer.");
        return false;
    }

    std::vector<std::string> userData{ "" };
    std::vector<std::string> buffers{ std::string((char*)trtBuffer->data(), trtBuffer->size()) };
    std::vector<std::uint64_t> bytesToEncrypt{ trtBuffer->size() };

    auto cw = ItvCv::CreateCryptoWrapper(m_logger);
    if (!cw->EncryptBuffers(userData, buffers, bytesToEncrypt, cachingPath.append(fileName).string(boost::filesystem::detail::utf8_codecvt_facet())))
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Failed to cache engine to output file.");
        return false;
    }

    return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> FileCacheHelper::Load(
    const std::string& fileName)
{
    auto cachingPath = GetCachingPath();
    if (cachingPath.empty())
    {
        ITVCV_LOG(m_logger, ITV8::LOG_INFO, "GPU cache path is not available.");
        return {};
    }

    cachingPath.append(fileName);
    if (!boost::filesystem::is_regular_file(cachingPath))
    {
        ITVCV_LOG(m_logger, ITV8::LOG_INFO, "Cached engine file does not exist.");
        return {};
    }

    auto cw = ItvCv::CreateCryptoWrapper(m_logger);
    auto buffers = cw->GetDecryptedContent(cachingPath.string(boost::filesystem::detail::utf8_codecvt_facet()));
    if (buffers.empty())
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Failed to read cached engine file.");
        return {};
    }

    auto trtRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(TrtLogger::GetInstance(m_logger)));
    if (!trtRuntime)
    {
        ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "Failed to create trtRuntime.");
        return {};
    }

    return std::shared_ptr<nvinfer1::ICudaEngine>(
        trtRuntime->deserializeCudaEngine((void*) buffers[0].second.data(), buffers[0].second.size()));
}

std::string InferenceWrapper::FileCacheHelper::GetFileName(
    const ITV8::Size& inputSize,
    const std::string& model,
    const std::string& weights,
    const std::string& originalFileName,
    const bool int8Usage) const
{
    boost::format cudaMsgFormat("%1%; Code:%2%; Name:%3%; Msg:%4%.");
    cudaError_t cudaError;

    // input dims str
    std::string inputSizeStr;
    if (inputSize.width != 0 && inputSize.height != 0)
    {
        inputSizeStr = fmt::format("{}x{}_", inputSize.width, inputSize.height);
    }

    // GPU ordinal
    int gpuOrdinal{ 0 };
    cudaError = cudaGetDevice(&gpuOrdinal);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error(
            str(cudaMsgFormat
                % "Unable to get gpu ordinal"
                % cudaError
                % cudaGetErrorName(cudaError)
                % cudaGetErrorString(cudaError)));
    }

    // GPU model name
    cudaDeviceProp prop;
    cudaError = cudaGetDeviceProperties(&prop, gpuOrdinal);
    if (cudaError != cudaSuccess)
    {
        throw std::runtime_error(
            str(cudaMsgFormat
                % "Unable to get cuda properites"
                % cudaError
                % cudaGetErrorName(cudaError)
                % cudaGetErrorString(cudaError)));
    }
    // get hash from key
    boost::uuids::detail::md5 hash;
    boost::uuids::detail::md5::digest_type digest;
    hash.process_bytes(static_cast<const void *>(&int8Usage), sizeof(int8Usage));
    hash.process_bytes(static_cast<const void *>(inputSizeStr.data()), inputSizeStr.size());
    hash.process_bytes(static_cast<const void *>(model.data()), model.size());
    hash.process_bytes(static_cast<const void *>(weights.data()), weights.size());
    hash.get_digest(digest);
    return fmt::format("{}_{}_{}{}{:08x}{:08x}.engine",
        prop.name,
        originalFileName,
        inputSizeStr,
        int8Usage ? "int8_" : "",
        digest[0],
        digest[1]);
}
}