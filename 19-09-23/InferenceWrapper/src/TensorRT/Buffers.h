#ifndef _BUFFERS_H_
#define _BUFFERS_H_

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <vector>
#include <memory>
#include <mutex>

namespace InferenceWrapper
{

template <typename AllocFunction, typename FreeFunction>
class GenericBuffer
{
public:
    GenericBuffer()
        : m_buffer(nullptr), m_nbBytes(0)
    {
    }

    GenericBuffer(std::size_t size)
        : m_nbBytes(size)
    {
        if (!AllocFn(&m_buffer, m_nbBytes))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : m_nbBytes(buf.size()), m_buffer(buf.m_buffer)
    {
        buf.m_nbBytes = 0;
        buf.m_buffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            FreeFn(m_buffer);
            m_nbBytes = buf.m_nbBytes;
            m_buffer = buf.m_buffer;
            buf.m_nbBytes = 0;
            buf.m_buffer = nullptr;
        }
        return *this;
    }

    ~GenericBuffer()
    {
        FreeFn(m_buffer);
    }

    void* data() { return m_buffer; }

    const void* data() const { return m_buffer; }

    std::size_t size() const { return m_nbBytes; }

private:
    void* m_buffer = nullptr;
    std::size_t m_nbBytes = 0;
    AllocFunction AllocFn;
    FreeFunction FreeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const { free(ptr); }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class CManagedBuffer
{
public:
    DeviceBuffer m_deviceBuffer;
    HostBuffer m_hostBuffer;
};

class CBufferManager
{
public:
    CBufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const std::size_t& batchSize);

    std::vector<void *>& GetDeviceBindings();

    const std::vector<void *>& GetDeviceBindings() const;

    void* GetHostBuffer(const std::string& tensorName) const;
    void* GetHostBufferByID(const int indx) const;
    void* GetDeviceBuffer(const std::string& tensorName) const;

    std::size_t sizeInBytes(const std::string tensorName) const;

    std::size_t size(const std::string tensorName) const;

    void CopyInputToDevice();

    void CopyOutputToHost();

    void CopyInputToDeviceAsync(const cudaStream_t& stream = 0);

    void CopyOutputToHostAsync(const cudaStream_t& stream = 0);
    nvinfer1::Dims GetDims(const std::string &tensorName)const;
private:
    inline unsigned int GetElementSize(nvinfer1::DataType t) const;

    inline int64_t Volume(const nvinfer1::Dims& d) const;

    void* getBuffer(const bool isHost, const std::string& tensorName) const;
    void* getBuffer(const bool isHost, const int& index) const;
    void memcpyBuffers(bool hostToDevice, bool async = false, const cudaStream_t& stream = 0);

private:
    int m_batchSize;
    std::shared_ptr<const nvinfer1::ICudaEngine> m_engine;
    std::vector<void *> m_deviceBindings;
    std::vector<std::unique_ptr<CManagedBuffer>> m_managedBuffers;
};
} // InferenceWrapper
#endif
