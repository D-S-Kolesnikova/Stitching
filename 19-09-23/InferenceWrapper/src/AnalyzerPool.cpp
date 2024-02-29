#include "AnalyzerPool.h"
#include "InferenceWrapper/InferenceEngine.h"

#include <boost/filesystem/path.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <functional>
#include <vector>
#include <map>
#include <mutex>

namespace
{

std::string GetHash(const InferenceWrapper::EngineCreationParams& parameters)
{
    return fmt::format(
        "{0}|{1}|{2}|{3}|{4}",
        parameters.netType,
        parameters.mode,
        parameters.gpuDeviceNumToUse,
        boost::filesystem::path(parameters.weightsFilePath).filename().string(),
        parameters.int8);
}

class PoolEntry
{
    std::mutex m_mutex;
    std::shared_ptr<void> m_handle = nullptr;

public:
    void* Get(const std::function<std::shared_ptr<void>()>& creationFunc)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_handle)
        {
            m_handle = creationFunc();
        }
        return m_handle.get();
    }
};
}

namespace InferenceWrapper
{
static std::mutex g_enginesPoolMutex;
static std::map<std::string, std::weak_ptr<PoolEntry>> g_enginesPool;

std::shared_ptr<void> GetSharedAnalyzer(
    const EngineCreationParams& parameters,
    const std::function<std::shared_ptr<void>()>& creationFunc)
{
    std::shared_ptr<PoolEntry> entry;

    const auto hash = GetHash(parameters);

    std::unique_lock<std::mutex> lock(g_enginesPoolMutex);
    auto& weak = g_enginesPool[hash];
    entry = weak.lock();

    if (!entry)
    {
        entry = std::make_shared<PoolEntry>();
        weak = entry;
    }
    lock.unlock();
    return std::shared_ptr<void>( entry, entry->Get(creationFunc) );
}

}
