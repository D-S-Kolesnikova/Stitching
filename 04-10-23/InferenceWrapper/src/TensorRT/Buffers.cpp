#include "Buffers.h"

#include <numeric>
#include <iostream>

namespace InferenceWrapper
{

CBufferManager::CBufferManager(
    std::shared_ptr<nvinfer1::ICudaEngine> engine,
    const size_t& batchSize)
    : m_batchSize(batchSize), m_engine(engine)
{
    for (int i = 0; i < m_engine->getNbBindings(); ++i)
    {
        auto vol = Volume(m_engine->getBindingDimensions(i));
        auto elementSize = GetElementSize(m_engine->getBindingDataType(i));
        auto allocSize = m_batchSize * vol * elementSize;
        std::unique_ptr<CManagedBuffer> manBuf{new CManagedBuffer};
        manBuf->m_hostBuffer = HostBuffer(allocSize);
        manBuf->m_deviceBuffer = DeviceBuffer(allocSize);
        m_deviceBindings.emplace_back(manBuf->m_deviceBuffer.data());
        m_managedBuffers.emplace_back(std::move(manBuf));
    }
}

std::vector<void *>& CBufferManager::GetDeviceBindings()
{
    return m_deviceBindings;
}

const std::vector<void *>& CBufferManager::GetDeviceBindings() const
{
    return m_deviceBindings;
}

void* CBufferManager::GetHostBuffer(const std::string& tensorName) const
{
    return getBuffer(true, tensorName);
}
void* CBufferManager::GetHostBufferByID(const int indx) const
{
    return getBuffer(true, indx);
}

void* CBufferManager::GetDeviceBuffer(const std::string& tensorName) const
{
    return getBuffer(false, tensorName);
}

size_t CBufferManager::sizeInBytes(const std::string tensorName) const
{
    auto index = m_engine->getBindingIndex(tensorName.c_str());

    if (index == -1)
        return ~std::size_t(0);

    return m_managedBuffers[index]->m_hostBuffer.size();
}

size_t CBufferManager::size(const std::string tensorName) const
{
    auto index = m_engine->getBindingIndex(tensorName.c_str());

    if (index == -1)
        return ~std::size_t(0);

    return m_managedBuffers[index]->m_hostBuffer.size() / GetElementSize(
        m_engine->getBindingDataType(index));
}

void CBufferManager::CopyInputToDevice()
{
    memcpyBuffers(true);
}

void CBufferManager::CopyOutputToHost()
{
    memcpyBuffers(false);
}

void CBufferManager::CopyInputToDeviceAsync(const cudaStream_t& stream)
{
    memcpyBuffers(true, true, stream);
}

void CBufferManager::CopyOutputToHostAsync(const cudaStream_t& stream)
{
    memcpyBuffers(false, true, stream);
}

nvinfer1::Dims CBufferManager::GetDims(const std::string& tensorName) const
{
    auto index = m_engine->getBindingIndex(tensorName.c_str());

    return m_engine->getBindingDimensions(index);
}

unsigned int CBufferManager::GetElementSize(nvinfer1::DataType t) const
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            throw std::runtime_error("Invalid DataType.");
    }
    return 0;
}

int64_t CBufferManager::Volume(const nvinfer1::Dims& d) const
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

void* CBufferManager::getBuffer(const bool isHost, const std::string& tensorName) const
{
    int index = m_engine->getBindingIndex(tensorName.c_str());

    if (index == -1)
        return nullptr;

    return isHost
        ? m_managedBuffers[index]->m_hostBuffer.data()
        : m_managedBuffers[index]->m_deviceBuffer.data();
}

void* CBufferManager::getBuffer(const bool isHost, const int& index) const
{
    if (index == -1)
        return nullptr;

    return isHost
        ? m_managedBuffers[index]->m_hostBuffer.data()
        : m_managedBuffers[index]->m_deviceBuffer.data();
}
void CBufferManager::memcpyBuffers(bool hostToDevice, bool async, const cudaStream_t& stream)
{
    const cudaMemcpyKind memcpyType =
        hostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;

    for (int i = 0; i < m_engine->getNbBindings(); ++i)
    {
        // check for the binding is it input or output as required by user
        if (!((hostToDevice && m_engine->bindingIsInput(i))
            || (!hostToDevice && !m_engine->bindingIsInput(i))))
        {
            continue;
        }

        // choose dst and src
        const void* src = hostToDevice
            ? m_managedBuffers[i]->m_hostBuffer.data()
            : m_managedBuffers[i]->m_deviceBuffer.data();
        void* dst = hostToDevice
            ? m_managedBuffers[i]->m_deviceBuffer.data()
            : m_managedBuffers[i]->m_hostBuffer.data();

        const std::size_t byteSize = m_managedBuffers[i]->m_hostBuffer.size();

        if (async)
        {
            cudaMemcpyAsync(dst, src, byteSize, memcpyType, stream);
        }
        else
        {
            cudaMemcpy(dst, src, byteSize, memcpyType);
        }

    }
}
} // InferenceWrapper
