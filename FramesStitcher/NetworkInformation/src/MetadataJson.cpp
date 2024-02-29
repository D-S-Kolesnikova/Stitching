#include "TagsConstant.h"
#include "MetadataJson.h"

#include <cryptoWrapper/cryptoWrapperLib.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996 4459 4458)
#endif
#include <boost/lexical_cast.hpp>
#include <boost/xpressive/xpressive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace
{
std::string VersionToString(const ItvCv::Version& version)
{
    return fmt::format("{}.{}.{}", version.major, version.minor, version.patch);
}
}

const Json::Value& GetRequiredNodeByTag(const Json::Value& jsonNode, const std::string& tag)
{
    const auto& resultNode = jsonNode[tag];
    if (!resultNode)
    {
        throw std::runtime_error(fmt::format("Metadata-json: doesn\'t contain required tag : {}", tag));
    }
    return resultNode;
}

ItvCv::Version MetadataJson<NodeType::Version>::Parse(const Json::Value& root)
{
    std::vector<std::string> versionStr;
    versionStr.reserve(3);
    const auto& data = GetRequiredNodeByTag(root, Tags::JSON_VERSION).asString();

    boost::split(versionStr, data, boost::is_any_of("."));
    if(versionStr.size() != 3)
    {
        throw std::runtime_error("Metadata-json has incorrect version");
    }
    std::vector<int> versionInt(3, 0);
    for(auto i = 0; i < static_cast<int>(versionStr.size()); ++i)
    {
        const auto& element = versionStr[i];
        if(!boost::all(element, boost::is_digit()))
        {
            throw std::runtime_error(fmt::format("Metadata-json: version element {} is not int", element));
        }
        versionInt[i] = boost::lexical_cast<int>(element);
    }

    if(NETWORK_INFORMATION_METADATA_VERSION.major != versionInt[0])
    {
        throw std::runtime_error(
            fmt::format(
                "Metadata-json: major version:{} != supported version:{}",
                versionInt[0],
                NETWORK_INFORMATION_METADATA_VERSION.major));
    }

    return {versionInt[0], versionInt[1], versionInt[2]};
}

Json::Value MetadataJson<NodeType::Version>::Dump()
{
    return VersionToString(NETWORK_INFORMATION_METADATA_VERSION);
}

ItvCv::InputParams MetadataJson<NodeType::Input>::Parse(
    const Json::Value& root)
{
    ItvCv::InputParams result;
    const auto& inputNodeJsonData = GetRequiredNodeByTag(root, Tags::JSON_INPUT);

    result.inputWidth = GetRequiredNodeByTag(inputNodeJsonData, Tags::JSON_WIDTH).asInt();
    result.inputHeight = GetRequiredNodeByTag(inputNodeJsonData, Tags::JSON_HEIGHT).asInt();

    result.numChannels = GetRequiredNodeByTag(
        inputNodeJsonData,
        Tags::JSON_NUM_CHANNELS).asInt();

    result.pixelFormat = ConvertStringToEnum(
        GetRequiredNodeByTag(inputNodeJsonData, Tags::JSON_PIXEL_FORMAT).asString(),
        TagToEnumMaps::PIXEL_FORMAT_MAP);

    result.supportedDynamicBatch = GetRequiredNodeByTag(inputNodeJsonData, Tags::JSON_DYNAMIC_BATCH).asBool();

    result.resizePolicy = ConvertStringToEnum(
        inputNodeJsonData[Tags::JSON_RESIZE_POLICY].asString(),
        TagToEnumMaps::RESIZE_POLICY_MAP);

    const auto& normalizationNode = GetRequiredNodeByTag(
        inputNodeJsonData,
        Tags::JSON_NORMALIZATION_VALUES);

    const auto normalizationArraySize = static_cast<int>(normalizationNode.size());
    if (result.numChannels != normalizationArraySize)
    {
        throw std::runtime_error(
            fmt::format(
                "Metadata-json: count normalization values: {} != num channel: {}",
                normalizationArraySize,
                result.numChannels));
    }

    std::vector<ItvCv::NormalizationValue> normalizationValues(normalizationArraySize);

    for (auto i = 0; i != normalizationArraySize; i++)
    {
        normalizationValues[i].mean =
            GetRequiredNodeByTag(normalizationNode[i], Tags::JSON_MEAN).asFloat();

        normalizationValues[i].scale =
            GetRequiredNodeByTag(normalizationNode[i], Tags::JSON_SCALE).asFloat();
    }
    result.normalizationValues = std::move(normalizationValues);
    return result;
}

Json::Value MetadataJson<NodeType::Input>::Dump(
    const ItvCv::NetworkInformation& netInfo)
{
    Json::Value inputNode;
    inputNode[Tags::JSON_NUM_CHANNELS] = netInfo.inputParams.numChannels;
    inputNode[Tags::JSON_WIDTH] = netInfo.inputParams.inputWidth;
    inputNode[Tags::JSON_HEIGHT] = netInfo.inputParams.inputHeight;
    inputNode[Tags::JSON_DYNAMIC_BATCH] = netInfo.inputParams.supportedDynamicBatch;
    inputNode[Tags::JSON_RESIZE_POLICY] =
        ReverseMap(TagToEnumMaps::RESIZE_POLICY_MAP)[netInfo.inputParams.resizePolicy];
    inputNode[Tags::JSON_PIXEL_FORMAT] =
        ReverseMap(TagToEnumMaps::PIXEL_FORMAT_MAP)[netInfo.inputParams.pixelFormat];

    Json::Value normalizationNode;
    for(int i = 0; i < netInfo.inputParams.numChannels; ++i)
    {
        const auto& normalizationData = netInfo.inputParams.normalizationValues[i];
        normalizationNode[i][Tags::JSON_MEAN] = normalizationData.mean;
        normalizationNode[i][Tags::JSON_SCALE] = normalizationData.scale;
    }

    inputNode[Tags::JSON_NORMALIZATION_VALUES] = std::move(normalizationNode);

    return inputNode;
}

ItvCv::CommonParams MetadataJson<NodeType::Common>::Parse(
    const Json::Value& root)
{
    const auto& commonNode = GetRequiredNodeByTag(root, Tags::JSON_COMMON);
    return {
    ConvertStringToEnum(
        GetRequiredNodeByTag(commonNode, Tags::JSON_ARCHITECTURE).asString(),
        TagToEnumMaps::ARCHITECTURE_MAP),
    ConvertStringToEnum(
        GetRequiredNodeByTag(commonNode, Tags::JSON_NET_TYPE).asString(),
        TagToEnumMaps::NET_TYPES_MAP),
    ConvertStringToEnum(
        GetRequiredNodeByTag(commonNode, Tags::JSON_MODEL_REPRESENTATION).asString(),
        TagToEnumMaps::MODELS_REPRESENTATIONS_MAP),
    ConvertStringToEnum(
        GetRequiredNodeByTag(commonNode, Tags::JSON_WEIGHT_TYPE).asString(),
        TagToEnumMaps::WEIGHT_TYPE_MAP)
    };
}

Json::Value MetadataJson<NodeType::Common>::Dump(
    const ItvCv::NetworkInformation& netInfo)
{
    Json::Value commonParams;
    commonParams[Tags::JSON_ARCHITECTURE] =
        ReverseMap(TagToEnumMaps::ARCHITECTURE_MAP)[netInfo.commonParams.architecture];
    commonParams[Tags::JSON_NET_TYPE] =
        ReverseMap(TagToEnumMaps::NET_TYPES_MAP)[netInfo.commonParams.analyzerType];
    commonParams[Tags::JSON_MODEL_REPRESENTATION] =
        ReverseMap(TagToEnumMaps::MODELS_REPRESENTATIONS_MAP)[netInfo.commonParams.modelRepresentation];
    commonParams[Tags::JSON_WEIGHT_TYPE] =
        ReverseMap(TagToEnumMaps::WEIGHT_TYPE_MAP)[netInfo.commonParams.weightType];
    return commonParams;
}

ItvCv::ComplexLabelType MetadataJson<NodeType::ComplexLabel>::Parse(
    const Json::Value& complexLabelNode)
{
    const auto objectType = ConvertStringToEnum(
        GetRequiredNodeByTag(complexLabelNode, Tags::JSON_OBJECT_DETECTION).asString(),
        TagToEnumMaps::OBJECT_DETECTION_MAP);

    return ItvCv::ComplexLabelType(objectType, ParseSubtype(complexLabelNode));
}

Json::Value MetadataJson<NodeType::ComplexLabel>::Dump(
    const ItvCv::ComplexLabelType& complexData)
{
    Json::Value complexLabelNode;
    complexLabelNode[Tags::JSON_OBJECT_DETECTION] = ReverseMap(TagToEnumMaps::OBJECT_DETECTION_MAP)[complexData.objectType];
    const auto& subtypeNode = DumpSubtype(complexData);
    if(subtypeNode)
    {
        complexLabelNode[Tags::JSON_SUBTYPE] = subtypeNode.get();
    }
    return complexLabelNode;
}

boost::optional<Json::Value> MetadataJson<NodeType::ComplexLabel>::DumpSubtype(
    const ItvCv::ComplexLabelType& complexData)
{
    Json::Value subtypeNode;
    const auto subtype = complexData.WichSubtype();
    switch (subtype)
    {
    case ItvCv::SubTypes::BodySegments:
    {
        subtypeNode[Tags::JSON_TYPE] = ReverseMap(TagToEnumMaps::SUB_TYPES_MAP)[subtype];
        const auto& segments = complexData.GetSubtype<ItvCv::SubTypes::BodySegments>();
        int i = 0;
        if (!segments)
        {
            throw std::runtime_error(fmt::format("Subtype: {}, dosen\'t contain data", subtype));
        }
        const auto& reverseMapSegmentsType = ReverseMap(TagToEnumMaps::BODY_SEGMENTS_MAP);
        for (const Segments::BodySegment& segment : segments.get())
        {
            subtypeNode[Tags::JSON_VALUES][i] = reverseMapSegmentsType.at(segment);
            ++i;
        }
        return subtypeNode;
    }
    default:
        return {};
    }
}

ItvCv::NetworkInformation::Labels_t MetadataJson<NodeType::LabelParameters>::Parse(
    const Json::Value& networkParametersNode)
{
    const auto& labelsNode = GetRequiredNodeByTag(
        networkParametersNode,
        Tags::JSON_LABELS);

    std::vector<ItvCv::Label> labels;
    std::set<int> collisionsPositionSet;

    for (const auto& label : labelsNode)
    {
        ItvCv::Label value;
        value.name = GetRequiredNodeByTag(label, Tags::JSON_NAME).asString();
        value.position = GetRequiredNodeByTag(label, Tags::JSON_POSITION).asInt();
        if(collisionsPositionSet.find(value.position) != collisionsPositionSet.end())
        {
            throw std::runtime_error(
                fmt::format(
                    "Metadata-json: collision label position, position:{} is already using",
                    value.position));
        }
        collisionsPositionSet.emplace(value.position);

        value.type = MetadataJson<NodeType::ComplexLabel>::Parse(
            GetRequiredNodeByTag(label, Tags::JSON_COMPLEX_TYPE));

        value.reportType = value.type.objectType != ObjectType::itvcvObjectUndefined
                ? ConvertStringToEnum(
                    GetRequiredNodeByTag(label, Tags::JSON_REPORT_TYPE).asString(),
                    TagToEnumMaps::CLASS_REPORT_TYPE_MAP)
                : ClassReportType::itvcvNeedSkip;
        labels.emplace_back(std::move(value));
    }

    return labels;
}

ItvCv::ComplexLabelType::SubTypes_t MetadataJson<NodeType::ComplexLabel>::ParseSubtype(
    const Json::Value& complexLabelNode)
{
    const auto& subType = complexLabelNode[Tags::JSON_SUBTYPE];
    if (!subType)
    {
        return boost::blank();
    }

    const auto& subTypeNode = GetRequiredNodeByTag(subType, Tags::JSON_TYPE);
    auto typeOfSubType = ConvertStringToEnum(
        subTypeNode.asString(),
        TagToEnumMaps::SUB_TYPES_MAP);

    if (typeOfSubType == ItvCv::SubTypes::BodySegments)
    {
        std::vector<Segments::BodySegment> result;
        const auto& valuesNode = GetRequiredNodeByTag(subType, Tags::JSON_VALUES);
        for (const auto& value : valuesNode)
        {
            result.emplace_back(
                ConvertStringToEnum(value.asString(), TagToEnumMaps::BODY_SEGMENTS_MAP));
        }
        return result;
    }
    return boost::blank();
}

Json::Value MetadataJson<NodeType::LabelParameters>::Dump(
    const ItvCv::NetworkInformation::Labels_t& labels)
{
    Json::Value labelsNode;
    int i = 0;
    for (const auto& label : labels)
    {
        labelsNode[i][Tags::JSON_NAME] = label.name;
        labelsNode[i][Tags::JSON_POSITION] = label.position;
        labelsNode[i][Tags::JSON_REPORT_TYPE] = ReverseMap(TagToEnumMaps::CLASS_REPORT_TYPE_MAP)[label.reportType];
        labelsNode[i][Tags::JSON_COMPLEX_TYPE] = MetadataJson<NodeType::ComplexLabel>::Dump(label.type);
        ++i;
    }

    return labelsNode;
}

ItvCv::SemanticSegmentationParameters MetadataJson<NodeType::SemanticSegmentationParameters>::Parse(
    const Json::Value& networkParametersNode)
{
    ItvCv::SemanticSegmentationParameters result;
    const auto& semanticSegmentationNode = GetRequiredNodeByTag(
        networkParametersNode,
        Tags::JSON_SEMANTIC_SEGMENTATION);

    result.isSingleChannel = GetRequiredNodeByTag(semanticSegmentationNode, Tags::JSON_IS_SINGLE_CHANNEL).asBool();
    result.labels = MetadataJson<NodeType::LabelParameters>::Parse(semanticSegmentationNode);
    return result;
}

Json::Value MetadataJson<NodeType::SemanticSegmentationParameters>::Dump(
    const ItvCv::NetworkInformation& netInfo)
{
    Json::Value semanticSegmentationNode;
    const auto semanticSegmentationData = boost::get<ItvCv::SemanticSegmentationParameters>(netInfo.networkParams);
    semanticSegmentationNode[Tags::JSON_IS_SINGLE_CHANNEL] = semanticSegmentationData.isSingleChannel;
    semanticSegmentationNode[Tags::JSON_LABELS] = MetadataJson<NodeType::LabelParameters>::Dump(semanticSegmentationData.labels);

    return semanticSegmentationNode;
}

ItvCv::ReidParams MetadataJson<NodeType::ReIdParameters>::Parse(
    const Json::Value& networkParametersNode)
{
    const auto& reidNode = GetRequiredNodeByTag(
        networkParametersNode,
        Tags::JSON_REID_PARAMETERS);

    ItvCv::ReidParams result;
    result.type = MetadataJson<NodeType::ComplexLabel>::Parse(
        GetRequiredNodeByTag(reidNode, Tags::JSON_COMPLEX_TYPE));

    result.vectorSize = GetRequiredNodeByTag(reidNode, Tags::JSON_VECTOR_SIZE).asInt();
    result.datasetGroup = GetRequiredNodeByTag(reidNode, Tags::JSON_DATASET_GROUP).asString();
    result.version = GetRequiredNodeByTag(reidNode,Tags::JSON_VERSION).asInt();

    return result;
}

Json::Value MetadataJson<NodeType::ReIdParameters>::Dump(
    const ItvCv::NetworkInformation& netInfo)
{
    Json::Value reIdNode;
    const auto reIdData = boost::get<ItvCv::ReidParams>(netInfo.networkParams);
    reIdNode[Tags::JSON_DATASET_GROUP] = reIdData.datasetGroup;
    reIdNode[Tags::JSON_VERSION] = reIdData.version;
    reIdNode[Tags::JSON_VECTOR_SIZE] = reIdData.vectorSize;
    reIdNode[Tags::JSON_COMPLEX_TYPE] = MetadataJson<NodeType::ComplexLabel>::Dump(reIdData.type);
    return reIdNode;
}

ItvCv::OpenPoseParams MetadataJson<NodeType::OpenPoseParameters>::Parse(
    const Json::Value& poseNetworkParams)
{
    const auto& openposeNode = GetRequiredNodeByTag(poseNetworkParams, Tags::JSON_OPENPOSE);

    return ItvCv::OpenPoseParams{
        GetRequiredNodeByTag(openposeNode, Tags::JSON_BOX_SIZE).asInt(),
        GetRequiredNodeByTag(openposeNode, Tags::JSON_STRIDE).asFloat(),
        GetRequiredNodeByTag(openposeNode, Tags::JSON_MIN_PEAK_DIST).asFloat(),
        GetRequiredNodeByTag(openposeNode, Tags::JSON_POINT_SCORE_THRESHOLD).asFloat(),
        GetRequiredNodeByTag(openposeNode, Tags::JSON_POINT_RATIO_THRESHOLD).asFloat(),
        GetRequiredNodeByTag(openposeNode, Tags::JSON_UP_SAMPLE).asFloat()
    };
}

Json::Value MetadataJson<NodeType::OpenPoseParameters>::Dump(
    const ItvCv::PoseNetworkParams& poseParams)
{
    const auto& openposeData = boost::get<ItvCv::OpenPoseParams>(poseParams.params);
    Json::Value openposeNode;
    openposeNode[Tags::JSON_BOX_SIZE] = openposeData.boxSize;
    openposeNode[Tags::JSON_STRIDE] = openposeData.stride;
    openposeNode[Tags::JSON_MIN_PEAK_DIST] = openposeData.minPeaksDistance;
    openposeNode[Tags::JSON_POINT_SCORE_THRESHOLD] = openposeData.midPointsScoreThreshold;
    openposeNode[Tags::JSON_POINT_RATIO_THRESHOLD] = openposeData.midPointsRatioThreshold;
    openposeNode[Tags::JSON_UP_SAMPLE] = openposeData.upSampleRatio;
    return openposeNode;
}

ItvCv::AEPoseParams MetadataJson<NodeType::AEParameters>::Parse(
    const Json::Value& poseNetworkParams)
{
    const auto& poseAeNode = GetRequiredNodeByTag(poseNetworkParams, Tags::JSON_POSE_AE);

    return ItvCv::AEPoseParams{
        GetRequiredNodeByTag(poseAeNode, Tags::JSON_DELTA).asFloat(),
        GetRequiredNodeByTag(poseAeNode, Tags::JSON_PEAK_THRESHOLD).asFloat(),
        GetRequiredNodeByTag(poseAeNode, Tags::JSON_TAGS_THRESHOLD).asFloat(),
        GetRequiredNodeByTag(poseAeNode, Tags::JSON_CONFIDENCE_THRESHOLD).asFloat(),
        GetRequiredNodeByTag(poseAeNode, Tags::JSON_MAX_OBJECTS).asInt(),
    };
}

Json::Value MetadataJson<NodeType::AEParameters>::Dump(
    const ItvCv::PoseNetworkParams& poseParams)
{
    const auto& poseAeData = boost::get<ItvCv::AEPoseParams>(poseParams.params);
    Json::Value poseAeNode;
    poseAeNode[Tags::JSON_TAGS_THRESHOLD] = poseAeData.tagsThreshold;
    poseAeNode[Tags::JSON_PEAK_THRESHOLD] = poseAeData.peakThreshold;
    poseAeNode[Tags::JSON_CONFIDENCE_THRESHOLD] = poseAeData.confidenceThreshold;
    poseAeNode[Tags::JSON_DELTA] = poseAeData.delta;
    poseAeNode[Tags::JSON_MAX_OBJECTS] = poseAeData.maxObjects;
    return poseAeNode;
}

ItvCv::PointDetectionParams MetadataJson<NodeType::PointDetectionParameters>::Parse(
    const Json::Value& networkParametersNode)
{
    ItvCv::PointDetectionParams result;
    const auto& PointDetectionNode = GetRequiredNodeByTag(
        networkParametersNode,
        Tags::JSON_POINT_DETECTION);

    result.maxNumKeypoints = GetRequiredNodeByTag(PointDetectionNode, Tags::JSON_NUM_KEYPOINTS).asInt();
    result.numDescriptors = GetRequiredNodeByTag(PointDetectionNode, Tags::JSON_NUM_DESCRIPTORS).asInt();
    return result;
}

Json::Value MetadataJson<NodeType::PointDetectionParameters>::Dump(const ItvCv::PointDetectionParams& pointDetectionParams)
{
    Json::Value PointDetectionNode;
    const auto pointDetectionData = boost::get<ItvCv::PointDetectionParams>(pointDetectionParams);
    PointDetectionNode[Tags::JSON_POSITION] = pointDetectionData.maxNumKeypoints;

    return PointDetectionNode;
}

ItvCv::PoseNetworkParams MetadataJson<NodeType::PoseParameters>::Parse(
    const Json::Value& networkParametersNode,
    const ItvCv::CommonParams& commonParams)
{
    ItvCv::PoseNetworkParams result;
    const auto& poseNetworkParams = GetRequiredNodeByTag(
        networkParametersNode,
        Tags::JSON_POSE_PARAMETERS);

    result.type = MetadataJson<NodeType::ComplexLabel>::Parse(
        GetRequiredNodeByTag(poseNetworkParams, Tags::JSON_COMPLEX_TYPE));
    const auto& linkerNode = GetRequiredNodeByTag(poseNetworkParams, Tags::JSON_POSE_LINKER);

    switch (result.type.objectType)
    {
    case ObjectType::itvcvObjectHuman:
        result.linker.heatmap = HeatmapParser<ObjectType::itvcvObjectHuman>(linkerNode);
        break;
    default:
        throw std::runtime_error("Unsupported object type");
    }

    result.minSubsetScore = GetRequiredNodeByTag(
        poseNetworkParams,
        Tags::JSON_MIN_SUBSET_SCORE).asFloat();

    switch (commonParams.architecture)
    {
    case ItvCv::ArchitectureType::Openpose18_MobileNet:
        result.linker.paf = PafParser(linkerNode);
        result.params = MetadataJson<NodeType::OpenPoseParameters>::Parse(poseNetworkParams);
        break;
    case ItvCv::ArchitectureType::HigherHRNet_AE:
        result.params = MetadataJson<NodeType::AEParameters>::Parse(poseNetworkParams);
        break;
    default:
        throw std::runtime_error(
            fmt::format(
                "Metadata-json: pose parameters doesn\'t support this type architecture :\"{}\"",
                commonParams.architecture));
    }
    return result;
}

Json::Value MetadataJson<NodeType::PoseParameters>::Dump(
    const ItvCv::NetworkInformation& netInfo)
{
    const auto& poseParamsData = boost::get<ItvCv::PoseNetworkParams>(netInfo.networkParams);
    Json::Value poseParamsNode;
    poseParamsNode[Tags::JSON_COMPLEX_TYPE] = MetadataJson<NodeType::ComplexLabel>::Dump(poseParamsData.type);
    poseParamsNode[Tags::JSON_MIN_SUBSET_SCORE] = poseParamsData.minSubsetScore;
    switch(poseParamsData.type.objectType)
    {
    case ObjectType::itvcvObjectHuman:
        poseParamsNode[Tags::JSON_POSE_LINKER][Tags::JSON_HEATMAP] = DumpHeatmap<ObjectType::itvcvObjectHuman>(poseParamsData);
        break;
    default:
        throw std::runtime_error(fmt::format("Unsupported object type:{}", poseParamsData.type.objectType));
    }

    switch (netInfo.commonParams.architecture)
    {
    case ItvCv::ArchitectureType::Openpose18_MobileNet:
        poseParamsNode[Tags::JSON_OPENPOSE] = MetadataJson<NodeType::OpenPoseParameters>::Dump(poseParamsData);
        poseParamsNode[Tags::JSON_POSE_LINKER][Tags::JSON_PAF] = DumpPaf(poseParamsData);
        break;
    case ItvCv::ArchitectureType::HigherHRNet_AE:
        poseParamsNode[Tags::JSON_POSE_AE] = MetadataJson<NodeType::AEParameters>::Dump(poseParamsData);
        break;
    default:
        throw std::runtime_error(fmt::format("Unexpected pose architecture:{}", netInfo.commonParams.architecture));
    }
    return poseParamsNode;
}

std::vector<ItvCv::PoseLinker::SPafElement> MetadataJson<NodeType::PoseParameters>::PafParser(
    const Json::Value& linkerNode)
{
    const auto& pafNode = GetRequiredNodeByTag(linkerNode, Tags::JSON_PAF);
    std::vector<ItvCv::PoseLinker::SPafElement> paf;
    paf.reserve(pafNode.size());
    for (const auto& element : pafNode)
    {
        paf.emplace_back(
            ItvCv::PoseLinker::SPafElement{
                GetRequiredNodeByTag(element, Tags::JSON_ID).asInt(),
                GetRequiredNodeByTag(element, Tags::JSON_POINT_ID_FROM).asInt(),
                GetRequiredNodeByTag(element, Tags::JSON_POINT_ID_TO).asInt()
            });
    }

    return paf;
}

Json::Value MetadataJson<NodeType::PoseParameters>::DumpPaf(
    const ItvCv::PoseNetworkParams& poseParams)
{
    Json::Value pafNode;
    int i = 0;

    for(const auto& pafElement : poseParams.linker.paf.get())
    {
        pafNode[i][Tags::JSON_POINT_ID_FROM] = pafElement.idPointFrom;
        pafNode[i][Tags::JSON_POINT_ID_TO] = pafElement.idPointTo;
        pafNode[i][Tags::JSON_ID] = pafElement.idChannel;
        ++i;
    }
    return pafNode;
}

template <ObjectType Type_t>
std::vector<int> MetadataJson<NodeType::PoseParameters>::HeatmapParser(
    const Json::Value& linkerNode)
{
    const auto& availableMap = GetAvailablePoseMapByObjectType<Type_t>();
    const auto& heatmapNode = GetRequiredNodeByTag(linkerNode, Tags::JSON_HEATMAP);
    std::vector<int> result(heatmapNode.size());

    for (const auto& element : heatmapNode)
    {
        const int fromId = GetRequiredNodeByTag(element, Tags::JSON_ID).asInt();

        const auto toId = static_cast<int>(
            ConvertStringToEnum(
                GetRequiredNodeByTag(element, Tags::JSON_NAME).asString(),
                availableMap));

        result[fromId] = toId;
    }
    return result;
}

template <ObjectType Type_t>
Json::Value MetadataJson<NodeType::PoseParameters>::DumpHeatmap(
    const ItvCv::PoseNetworkParams& poseParams)
{
    const auto& availableMap = ReverseMap(GetAvailablePoseMapByObjectType<Type_t>());
    Json::Value heatmapNode;
    for (int i = 0; i < static_cast<int>(poseParams.linker.heatmap.size()); ++i)
    {
        heatmapNode[i][Tags::JSON_ID] = i;
        heatmapNode[i][Tags::JSON_NAME] = ConvertEnumToString(poseParams.linker.heatmap[i], availableMap);
    }
    return heatmapNode;
}

ItvCv::NetworkInformation::NetParams_t MetadataJson<NodeType::NetworkParameters>::Parse(
    const Json::Value& root,
    const ItvCv::CommonParams& commonParams)
{
    const auto& networkParametersNode = GetRequiredNodeByTag(
        root,
        Tags::JSON_NETWORK_PARAMETERS);

    switch (commonParams.analyzerType)
    {
    case itvcvAnalyzerSSD:
    case itvcvAnalyzerClassification:
        return MetadataJson<NodeType::LabelParameters>::Parse(networkParametersNode);
    case itvcvAnalyzerMaskSegments:
        return MetadataJson<NodeType::SemanticSegmentationParameters>::Parse(networkParametersNode);
    case itvcvAnalyzerHumanPoseEstimator:
        return MetadataJson<NodeType::PoseParameters>::Parse(networkParametersNode, commonParams);
    case itvcvAnalyzerSiamese:
        return MetadataJson<NodeType::ReIdParameters>::Parse(networkParametersNode);
    case itvcvAnalyzerPointDetection:
        return MetadataJson<NodeType::PointDetectionParameters>::Parse(networkParametersNode);
    default:
        throw std::runtime_error(
            fmt::format("Metadata-json: doesn\'t support this type:\"{}\" analyzer", commonParams.analyzerType));
    }
}

Json::Value MetadataJson<NodeType::NetworkParameters>::Dump(
    const ItvCv::NetworkInformation& netInfo)
{
    Json::Value networkParamsNode;
    switch (netInfo.commonParams.analyzerType)
    {
    case itvcvAnalyzerSSD:
    case itvcvAnalyzerClassification:
    {
        const auto& labels = boost::get<ItvCv::NetworkInformation::Labels_t>(netInfo.networkParams);
        networkParamsNode[Tags::JSON_LABELS] = MetadataJson<NodeType::LabelParameters>::Dump(labels);
        break;
    }
    case itvcvAnalyzerMaskSegments:
        networkParamsNode[Tags::JSON_SEMANTIC_SEGMENTATION] = MetadataJson<NodeType::SemanticSegmentationParameters>::Dump(netInfo);
        break;
    case itvcvAnalyzerHumanPoseEstimator:
        networkParamsNode[Tags::JSON_POSE_PARAMETERS] = MetadataJson<NodeType::PoseParameters>::Dump(netInfo);
        break;
    case itvcvAnalyzerSiamese:
        networkParamsNode[Tags::JSON_REID_PARAMETERS] = MetadataJson<NodeType::ReIdParameters>::Dump(netInfo);
        break;
    default:
        throw std::runtime_error(
            fmt::format("In {}; Doesn\'t support this type:\"{}\" analyzer", __FUNCTION__, netInfo.commonParams.analyzerType));
    }
    return networkParamsNode;
}

boost::optional<ItvCv::ModelDescription> MetadataJson<NodeType::ModelDescription>::Parse(
    const Json::Value& root)
{
    const auto& modelDescriptionNode = root[Tags::JSON_MODEL_DESCRIPTION];
    if(!modelDescriptionNode)
    {
        return {};
    }
    ItvCv::ModelDescription modelDescriptionData;

    if(const auto& dataNode = modelDescriptionNode[Tags::JSON_AUTHOR])
    {
        modelDescriptionData.author = dataNode.asString();
    }

    if(const auto& dataNode = modelDescriptionNode[Tags::JSON_DESCRIPTION])
    {
        modelDescriptionData.info = dataNode.asString();
    }

    if(const auto& dataNode = modelDescriptionNode[Tags::JSON_TASK_NAME])
    {
        modelDescriptionData.task = dataNode.asString();
    }

    return { modelDescriptionData };
}

boost::optional<Json::Value> MetadataJson<NodeType::ModelDescription>::Dump(const ItvCv::NetworkInformation& netInfo)
{
    //version dump separately
    if(!netInfo.description)
    {
        return {};
    }
    const auto& data = netInfo.description.get();

    if(
        data.info.empty()
        && data.task.empty()
        && data.author.empty())
    {
        return {};
    }

    Json::Value modelDescriptionNode;
    modelDescriptionNode[Tags::JSON_DESCRIPTION] = data.info;
    modelDescriptionNode[Tags::JSON_TASK_NAME] = data.task;
    modelDescriptionNode[Tags::JSON_AUTHOR] = data.author;

    return { modelDescriptionNode };
}
