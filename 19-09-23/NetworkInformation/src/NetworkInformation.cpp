#include "TagsConstant.h"
#include "MetadataJson.h"
#include "NetworkInformationPool.h"
#include "NetworkInformation/NetworkInformationLib.h"
#include "NetworkInformation/Utils.h"

#include <cryptoWrapper/cryptoWrapperLib.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/detail/utf8_codecvt_facet.hpp>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996 4459 4458)
#endif
#include <boost/property_tree/ptree.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

class NetworkParser
{
public:
    NetworkParser(const std::string& pathToEncryptedModel)
    {
        auto cryptoWrapper = ItvCv::CreateCryptoWrapper(nullptr);
        const auto& buffers = cryptoWrapper->GetDecryptedContent(pathToEncryptedModel);

        m_metaInfo = FindInDecryptedBuffersByTag(buffers, UserDataKey::METADATA_KEY);
        m_weights = FindInDecryptedBuffersByTag(buffers, UserDataKey::WEIGHTS_KEY);

        MetadataParser(m_metaInfo);

        if (m_networkInformation.commonParams.modelRepresentation != ItvCv::ModelRepresentation::onnx
            && m_networkInformation.commonParams.modelRepresentation != ItvCv::ModelRepresentation::ascend)
        {
            m_networkInformation.modelData = FindInDecryptedBuffersByTag(buffers, UserDataKey::MODEL_KEY);
        }
    }

    void MetadataParser(const std::string& metaInfoJson)
    {
        Json::Value metaInfoValue;
        ///Warning Json::Reader is deprecated
        Json::Reader reader;

        if (!reader.parse(metaInfoJson, metaInfoValue))
        {
            throw std::runtime_error("Bad json file. unable to parse.");
        }

        const auto version = MetadataJson<NodeType::Version>::Parse(metaInfoValue);
        m_networkInformation.description = MetadataJson<NodeType::ModelDescription>::Parse(metaInfoValue);
        if(!m_networkInformation.description)
        {
            m_networkInformation.description = {ItvCv::ModelDescription()};
        }
        m_networkInformation.description.get().metadataVersion = version;

        m_networkInformation.inputParams = MetadataJson<NodeType::Input>::Parse(metaInfoValue);
        m_networkInformation.commonParams = MetadataJson<NodeType::Common>::Parse(metaInfoValue);

        m_networkInformation.networkParams = MetadataJson<NodeType::NetworkParameters>::Parse(
            metaInfoValue,
            m_networkInformation.commonParams);
    }

    ItvCv::NetworkInformation TakeInformation()
    {
        return std::move(m_networkInformation);
    }

    std::string TakeWeights()
    {
        return std::move(m_weights);
    }

private:
    std::string FindInDecryptedBuffersByTag(
        const std::vector<std::pair<std::string, std::string>>& buffers,
        const std::string& tag)
    {
        auto data = std::find_if(
            buffers.begin(),
            buffers.end(),
            [&](std::pair<std::string, std::string> const & ref)
        {
            return ref.first == tag;
        });

        if (data == buffers.end())
        {
            throw std::runtime_error(
                fmt::format(
                    "Bad ANN file - A deprecated neural model format. {} not found. Please contact support team to update the model.",
                    tag));
        }
        return data->second;
    }

private:
    ItvCv::NetworkInformation m_networkInformation;
    std::string m_weights;
    std::string m_metaInfo;
};

namespace bPtree = boost::property_tree;
ItvCv::NetworkInformation NetworkInformationPool::ReadNetwork(const std::string& pathToEncryptedModel, std::string& weights)
{
    NetworkParser reader(pathToEncryptedModel);
    reader.TakeWeights().swap(weights);
    return reader.TakeInformation();
}


namespace ItvCv
{
ITVCV_NETINFO_API ItvCv::Version GetMetadataParserVersion()
{
    return NETWORK_INFORMATION_METADATA_VERSION;
}

ITVCV_NETINFO_API std::shared_ptr<NetworkInformation> GetNetworkInformation(
    const char* pathToEncryptedModel)
{

    return NetworkInformationPool::Instance()->Lookup(pathToEncryptedModel);
}

ITVCV_NETINFO_API std::string ConsumeWeightsData(std::shared_ptr<NetworkInformation> const& netInfo)
{
    return NetworkInformationPool::Instance()->TakeWeights(netInfo);
}

ITVCV_NETINFO_API std::string GenerateMetadata(const NetworkInformation& netInfo)
{
    Json::Value root;

    const Json::StreamWriterBuilder builder;

    root[Tags::JSON_VERSION] = MetadataJson<NodeType::Version>::Dump();
    root[Tags::JSON_INPUT] = MetadataJson<NodeType::Input>::Dump(netInfo);
    root[Tags::JSON_COMMON] = MetadataJson<NodeType::Common>::Dump(netInfo);
    root[Tags::JSON_NETWORK_PARAMETERS] = MetadataJson<NodeType::NetworkParameters>::Dump(netInfo);

    if (const auto& modelDescriptionNode = MetadataJson<NodeType::ModelDescription>::Dump(netInfo))
    {
        root[Tags::JSON_MODEL_DESCRIPTION] = modelDescriptionNode.get();
    }

    return Json::writeString(builder, root);
}

std::uint64_t GetEncryptSize(std::int64_t byteSize, size_t dataSize)
{
    const auto castDataSize = static_cast<uint64_t>(dataSize);
    return
        (byteSize > 0) && (byteSize <= castDataSize)
        ? static_cast<std::uint64_t>(byteSize)
        : castDataSize;
}

ITVCV_NETINFO_API bool DumpNetworkInformationToAnn(
    const std::string& weightsData,
    const NetworkInformation& netInfo,
    const std::string& pathOut,
    const std::int64_t byteSize)
{
    if(NetworkInfoUtils::ValidationParameters(netInfo).first != NetworkInfoUtils::ValidationError::NoError)
    {
        return false;
    }

    const auto metadataStr = GenerateMetadata(netInfo);

    if (
        netInfo.modelData.empty()
        && !(netInfo.commonParams.modelRepresentation == ModelRepresentation::onnx
            || netInfo.commonParams.modelRepresentation == ModelRepresentation::ascend))
    {
        throw std::runtime_error(
            fmt::format(
                "This type ModelRepresentation: {}, must contain model data",
                netInfo.commonParams.modelRepresentation));
    }

    auto cryptoWrapper = ItvCv::CreateCryptoWrapper(nullptr);
    auto isEncrypted = false;

    if (netInfo.modelData.empty())
    {
        isEncrypted = cryptoWrapper->EncryptBuffers(
            std::vector<std::string>{ UserDataKey::METADATA_KEY, UserDataKey::WEIGHTS_KEY },
            std::vector<std::string>{ metadataStr, weightsData },
            std::vector<std::uint64_t>{
                GetEncryptSize(byteSize, metadataStr.size()),
                GetEncryptSize(byteSize, weightsData.size())
            },
            pathOut);
    }
    else
    {
        isEncrypted = cryptoWrapper->EncryptBuffers(
            std::vector<std::string>{
                UserDataKey::MODEL_KEY,
                UserDataKey::METADATA_KEY,
                UserDataKey::WEIGHTS_KEY
            },
            std::vector<std::string>{ netInfo.modelData, metadataStr, weightsData},
            std::vector<std::uint64_t>{
                GetEncryptSize(byteSize, netInfo.modelData.size()),
                GetEncryptSize(byteSize, metadataStr.size()),
                GetEncryptSize(byteSize, weightsData.size())
            },
            pathOut);
    }
    //try to decrypt ann
    auto newNetInfo = GetNetworkInformation(pathOut.data());
    return isEncrypted;
}
}
