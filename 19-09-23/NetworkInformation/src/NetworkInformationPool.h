#pragma once

#include <NetworkInformation/NetworkInformationLib.h>
#include <mutex>
#include <memory>
#include <map>

class NetworkInformationPool : public std::enable_shared_from_this<NetworkInformationPool>
{
    std::mutex m_mutex;

    class PoolEntry;

    using storage_t = std::map<std::string, std::weak_ptr<PoolEntry>>;
    storage_t m_storage;

    struct AccessToken {};

    void UnrefEntry(storage_t::iterator iter);
    static ItvCv::NetworkInformation ReadNetwork(const std::string& pathToEncryptedModel, std::string& weights);

public:
    NetworkInformationPool(AccessToken){}

    static std::shared_ptr<NetworkInformationPool> Instance();

    std::shared_ptr<ItvCv::NetworkInformation> Lookup(const char* pathToEncryptedModel);
    
    std::string TakeWeights(const std::shared_ptr<ItvCv::NetworkInformation>& ptr);
};
