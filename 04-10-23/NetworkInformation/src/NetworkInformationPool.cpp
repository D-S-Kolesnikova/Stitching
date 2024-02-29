#include "NetworkInformationPool.h"

#include <stdlib.h>

constexpr uint32_t make_magic(uint32_t o1, uint32_t o2, uint32_t o3, uint32_t o4) { return o1 | (o2 << 8) | (o3 << 16) | (o4 << 24); }

class NetworkInformationPool::PoolEntry
{
    ItvCv::NetworkInformation m_data;
    static constexpr uint32_t MAGIC = make_magic('P', 'O', 'O', 'L');
    uint32_t m_magic = 0u;
    std::mutex m_mutex;
    std::string m_weights;
    std::shared_ptr<NetworkInformationPool> m_mgr;
    storage_t::iterator m_iter;

    struct Deleter
    {
        void operator()(PoolEntry* ptr) const
        {
            delete ptr;
        }
    };

public:
    template <typename ...Args>
    static std::shared_ptr<PoolEntry> Create(Args&&... args)
    {
        return std::shared_ptr<PoolEntry>( new PoolEntry(std::forward<Args>(args)...), Deleter() );
    }

    static PoolEntry& FromNetworkInformation(ItvCv::PNetworkInformation const& info)
    {
        auto* d = std::get_deleter<PoolEntry::Deleter>(info);
        if (!d)
           abort();
        return *from_member(&PoolEntry::m_data, info.get());
    }

    static ItvCv::PNetworkInformation GetNetworkInformation(std::shared_ptr<PoolEntry> const& e)
    {
        e->initialize();
        return ItvCv::PNetworkInformation(e, &e->m_data);
    }

    std::string TakeWeights()
    {
        if (MAGIC != m_magic)
            abort();
        std::lock_guard<std::mutex> entryLock(m_mutex);
        if (m_weights.empty())
        {
            m_mgr->ReadNetwork(GetEncryptedModelPath(), m_weights);
        }
        return std::move(m_weights);
    }

private:
    PoolEntry(std::shared_ptr<NetworkInformationPool>&& pool, storage_t::iterator iter)
        : m_mgr(std::move(pool))
        , m_iter(iter)
    {}
    PoolEntry(PoolEntry const&) = delete;
    PoolEntry& operator=(PoolEntry const&) = delete;

    ~PoolEntry()
    {
        m_mgr->UnrefEntry(m_iter);
    }

    template <typename T> static
    PoolEntry* from_member(T PoolEntry::* member, T* ptr)
    {
        constexpr PoolEntry* e = nullptr;
        return reinterpret_cast<PoolEntry*>( reinterpret_cast<uint8_t*>(ptr) - reinterpret_cast<intptr_t>(&(e->*member)) );
    }

    void initialize()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (MAGIC != m_magic)
        {
            m_data = m_mgr->ReadNetwork(GetEncryptedModelPath(), m_weights);
            m_magic = MAGIC;
        }
    }

    std::string const& GetEncryptedModelPath() const
    {
        return m_iter->first;
    }
};

void NetworkInformationPool::UnrefEntry(storage_t::iterator iter)
{
    std::unique_lock<std::mutex> storageLock(m_mutex);
    auto entry = iter->second.lock();
    if (!entry)
        m_storage.erase(iter);
}

std::shared_ptr<NetworkInformationPool> NetworkInformationPool::Instance()
{
    static std::mutex mutex;
    static std::weak_ptr<NetworkInformationPool> weak;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto strong = weak.lock())
        return strong;
    auto strong = std::make_shared<NetworkInformationPool>(AccessToken());
    weak = strong;
    return strong;
}

std::shared_ptr<ItvCv::NetworkInformation> NetworkInformationPool::Lookup(const char* pathToEncryptedModel)
{
    std::unique_lock<std::mutex> storageLock(m_mutex);
    auto insertRes = m_storage.emplace(pathToEncryptedModel, std::weak_ptr<PoolEntry>());
    auto& weak = insertRes.first->second;
    auto entry = weak.lock();
    if (!entry)
    {
        entry = PoolEntry::Create(shared_from_this(), insertRes.first);
        weak = entry;
    }
    storageLock.unlock();
    return PoolEntry::GetNetworkInformation(entry);
}

std::string NetworkInformationPool::TakeWeights(const std::shared_ptr<ItvCv::NetworkInformation>& ptr)
{
    auto& entry = PoolEntry::FromNetworkInformation(ptr);
    return entry.TakeWeights();
}