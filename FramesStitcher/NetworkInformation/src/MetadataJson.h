#ifndef NETWORKINFORMATION_MATADATA_PARSER_H
#define NETWORKINFORMATION_MATADATA_PARSER_H

#include "NetworkInformation/NetworkInformationLib.h"

#include <fmt/format.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996 4459 4458)
#endif
#include <boost/spirit/include/qi.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <json/json.h>

enum class NodeType
{
    Version,
    Input,
    Common,
    NetworkParameters,
    ComplexLabel,
    PoseParameters,
    ReIdParameters,
    LabelParameters,
    SemanticSegmentationParameters,
    OpenPoseParameters,
    PointDetectionParameters,
    AEParameters,
    ModelDescription
};

const Json::Value& GetRequiredNodeByTag(const Json::Value& jsonNode, const std::string& tag);

template<NodeType NodeType_t>
struct MetadataJson
{
   static void Parse(const Json::Value& root)
   {
       throw std::runtime_error(fmt::format("Not implemented parser fot this NodeType:", NodeType_t));
   }
   static Json::Value Dump()
   {
       throw std::runtime_error(fmt::format("Not implemented dump function fot this NodeType:", NodeType_t));
   }
};

template<>
struct MetadataJson<NodeType::Version>
{
    static ItvCv::Version Parse(const Json::Value& root);
    static Json::Value Dump();
};

template<>
struct MetadataJson<NodeType::Input>
{
    static ItvCv::InputParams Parse(const Json::Value& root);
    static Json::Value Dump(const ItvCv::NetworkInformation& netInfo);
};

template<>
struct MetadataJson<NodeType::Common>
{
    static ItvCv::CommonParams Parse(const Json::Value& root);
    static Json::Value Dump(const ItvCv::NetworkInformation& netInfo);
};

template<>
struct MetadataJson<NodeType::ComplexLabel>
{
    static ItvCv::ComplexLabelType Parse(const Json::Value& complexLabelNode);
    static Json::Value Dump(const ItvCv::ComplexLabelType& complexData);
private:
    static boost::optional<Json::Value> DumpSubtype(const ItvCv::ComplexLabelType& complexData);
    static ItvCv::ComplexLabelType::SubTypes_t ParseSubtype(const Json::Value& complexLabelNode);
};

template<>
struct MetadataJson<NodeType::LabelParameters>
{
    static ItvCv::NetworkInformation::Labels_t Parse(const Json::Value& networkParametersNode);
    static Json::Value Dump(const ItvCv::NetworkInformation::Labels_t& labels);
};

template<>
struct MetadataJson<NodeType::SemanticSegmentationParameters>
{
    static ItvCv::SemanticSegmentationParameters Parse(const Json::Value& networkParametersNode);
    static Json::Value Dump(const ItvCv::NetworkInformation& netInfo);
};

template<>
struct MetadataJson<NodeType::ReIdParameters>
{
    static ItvCv::ReidParams Parse(const Json::Value& networkParametersNode);
    static Json::Value Dump(const ItvCv::NetworkInformation& netInfo);
};

template<>
struct MetadataJson<NodeType::OpenPoseParameters>
{
    static ItvCv::OpenPoseParams Parse(const Json::Value& poseNetworkParams);
    static Json::Value Dump(const ItvCv::PoseNetworkParams& poseParams);
};

template<>
struct MetadataJson<NodeType::AEParameters>
{
    static ItvCv::AEPoseParams Parse(const Json::Value& poseNetworkParams);
    static Json::Value Dump(const ItvCv::PoseNetworkParams& poseParams);
};

template<>
struct MetadataJson<NodeType::PointDetectionParameters>
{
    static ItvCv::PointDetectionParams Parse(const Json::Value& networkParametersNode);
    static Json::Value Dump(const ItvCv::PointDetectionParams& pointDetectionParams);
};

template<>
struct MetadataJson<NodeType::PoseParameters>
{
    static ItvCv::PoseNetworkParams Parse(
        const Json::Value& networkParametersNode,
        const ItvCv::CommonParams& commonParams);

    static Json::Value Dump(const ItvCv::NetworkInformation& netInfo);

private:
    static std::vector<ItvCv::PoseLinker::SPafElement> PafParser(const Json::Value& linkerNode);
    static Json::Value DumpPaf(const ItvCv::PoseNetworkParams& poseParams);

    template<ObjectType Type_t>
    static std::vector<int> HeatmapParser(const Json::Value& linkerNode);
    template<ObjectType Type_t>
    static Json::Value DumpHeatmap(const ItvCv::PoseNetworkParams& poseParams);

};

template<>
struct MetadataJson<NodeType::NetworkParameters>
{
    static ItvCv::NetworkInformation::NetParams_t Parse(
        const Json::Value& root,
        const ItvCv::CommonParams& commonParams);

    static Json::Value Dump(const ItvCv::NetworkInformation& netInfo);
};

template<>
struct MetadataJson<NodeType::ModelDescription>
{
    static boost::optional<ItvCv::ModelDescription> Parse(const Json::Value& root);

    static boost::optional<Json::Value> Dump(const ItvCv::NetworkInformation& netInfo);
};
#endif
