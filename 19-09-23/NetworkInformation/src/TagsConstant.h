#ifndef NETWORKINFORMATION_TAGS_CONSTANT_H
#define NETWORKINFORMATION_TAGS_CONSTANT_H

#include <NetworkInformation/NetworkInformationLib.h>

#include <fmt/format.h>

#include <map>

// increment major version if changed: tags name, old json shame
// increment minor version if changed: added new parameters, new future
// increment patch version if changed: fixed bug, code parser
const ItvCv::Version NETWORK_INFORMATION_METADATA_VERSION{0, 2, 1};

namespace UserDataKey
{
constexpr auto METADATA_KEY = "metadata";
constexpr auto MODEL_KEY = "model";
constexpr auto WEIGHTS_KEY = "weights";
}

namespace Tags
{
constexpr auto JSON_DATASET_GROUP = "datasetGroup";
constexpr auto JSON_INPUT = "input";
constexpr auto JSON_COMMON = "common";
constexpr auto JSON_VERSION = "version";
constexpr auto JSON_PIXEL_FORMAT = "pixelFormat";
constexpr auto JSON_NUM_CHANNELS = "numChannels";
constexpr auto JSON_WIDTH = "width";
constexpr auto JSON_HEIGHT = "height";
constexpr auto JSON_RESIZE_POLICY = "resizePolicy";
constexpr auto JSON_MODEL_REPRESENTATION = "modelRepresentation";
constexpr auto JSON_NORMALIZATION_VALUES = "normalizationValues";
constexpr auto JSON_WEIGHT_TYPE = "weightType";
constexpr auto JSON_MEAN = "mean";
constexpr auto JSON_SCALE = "scale";
constexpr auto JSON_ARCHITECTURE = "architecture";
constexpr auto JSON_NET_TYPE = "netType";
constexpr auto JSON_LABELS = "labels";
constexpr auto JSON_COMPLEX_TYPE = "complexType";
constexpr auto JSON_NAME = "name";
constexpr auto JSON_POSITION = "position";
constexpr auto JSON_REPORT_TYPE = "reportType";
constexpr auto JSON_OBJECT_DETECTION = "objectDetection";
constexpr auto JSON_SUBTYPE = "subType";
constexpr auto JSON_TYPE = "type";
constexpr auto JSON_VALUES = "values";
constexpr auto JSON_DYNAMIC_BATCH = "dynamicBatch";
constexpr auto JSON_NETWORK_PARAMETERS = "networkParams";
constexpr auto JSON_REID_PARAMETERS = "reId";
constexpr auto JSON_VECTOR_SIZE = "vectorSize";
constexpr auto JSON_SEMANTIC_SEGMENTATION = "semanticSegmentation";
constexpr auto JSON_IS_SINGLE_CHANNEL = "isSingleChannel";
constexpr auto JSON_POSE_PARAMETERS = "pose";
constexpr auto JSON_OPENPOSE = "openpose";
constexpr auto JSON_POSE_AE = "poseAE";
constexpr auto JSON_MAX_OBJECTS = "maxObjects";
constexpr auto JSON_TAGS_THRESHOLD = "tagsThreshold";
constexpr auto JSON_CONFIDENCE_THRESHOLD = "confidenceThreshold";
constexpr auto JSON_DELTA = "delta";
constexpr auto JSON_PEAK_THRESHOLD = "peakThreshold";
constexpr auto JSON_POSE_LINKER = "linker";
constexpr auto JSON_HEATMAP = "heatmap";
constexpr auto JSON_ID = "id";
constexpr auto JSON_PAF = "paf";
constexpr auto JSON_POINT_ID_TO = "pointIdTo";
constexpr auto JSON_POINT_ID_FROM = "pointIdFrom";
constexpr auto JSON_BOX_SIZE = "boxSize";
constexpr auto JSON_STRIDE = "stride";
constexpr auto JSON_MIN_PEAK_DIST = "minPeaksDistance";
constexpr auto JSON_POINT_SCORE_THRESHOLD = "midPointsScoreThreshold";
constexpr auto JSON_POINT_RATIO_THRESHOLD = "midPointsRatioThreshold";
constexpr auto JSON_MIN_SUBSET_SCORE = "minSubsetScore";
constexpr auto JSON_UP_SAMPLE = "upSampleRatio";
constexpr auto JSON_MODEL_DESCRIPTION = "modelDescription";
constexpr auto JSON_AUTHOR = "author";
constexpr auto JSON_DESCRIPTION = "description";
constexpr auto JSON_TASK_NAME = "task";
constexpr auto JSON_POINT_DETECTION = "PointDetection";
constexpr auto JSON_NUM_KEYPOINTS = "maxNumKeypoints";
constexpr auto JSON_NUM_DESCRIPTORS = "numDescriptors";
}

namespace TagToEnumMaps
{
constexpr auto UNKNOWN_KEY = "Unknown";

using HumanPointMap_t = std::map<std::string, Pose::HumanPosePointType>;
using ReportTypeMap_t = std::map<std::string, ClassReportType>;
using ArchitectureTypeMap_t = std::map<std::string, ItvCv::ArchitectureType>;
using ModelRepresentationMap_t = std::map<std::string, ItvCv::ModelRepresentation>;
using SubTypesMap_t = std::map<std::string, ItvCv::SubTypes>;
using BodySegmentMap_t = std::map<std::string, Segments::BodySegment>;
using ObjectTypeMap_t = std::map<std::string, ObjectType>;
using AnalyzerTypeMap_t = std::map<std::string, itvcvAnalyzerType>;
using PixelFormatMap_t = std::map<std::string, ItvCv::PixelFormat>;
using ResizePolicyMap_t = std::map<std::string, ItvCv::ResizePolicy>;
using DataTypeMap_t = std::map<std::string, ItvCv::DataType>;

const ReportTypeMap_t CLASS_REPORT_TYPE_MAP
{
    {"Skip", ClassReportType::itvcvNeedSkip},
    {"NotReport", ClassReportType::itvcvNeedNotReport},
    {"Report", ClassReportType::itvcvNeedReport},
    {UNKNOWN_KEY, ClassReportType::itvcvNeedSkip}
};

const ArchitectureTypeMap_t ARCHITECTURE_MAP
{
    {"GoogleNet", ItvCv::ArchitectureType::GoogleNet},
    {"EfficientNet", ItvCv::ArchitectureType::EfficientNet},
    {"Osnetfpn", ItvCv::ArchitectureType::Osnetfpn},
    {"CustomSegmentation9_v1", ItvCv::ArchitectureType::CustomSegmentation9_v1},
    {"DDRNet", ItvCv::ArchitectureType::DDRNet},
    {"Openpose18_MobileNet", ItvCv::ArchitectureType::Openpose18_MobileNet},
    {"HigherHRNet_AE", ItvCv::ArchitectureType::HigherHRNet_AE},
    {"SSD_MobileNetv2", ItvCv::ArchitectureType::SSD_MobileNetv2},
    {"SSD_PyTorch", ItvCv::ArchitectureType::SSD_PyTorch},
    {"SSD_ResNet34", ItvCv::ArchitectureType::SSD_ResNet34},
    {"Yolo", ItvCv::ArchitectureType::Yolo},
    {"SuperPoint", ItvCv::ArchitectureType::SuperPoint},
    {UNKNOWN_KEY, ItvCv::ArchitectureType::Unknown}
};

const ModelRepresentationMap_t MODELS_REPRESENTATIONS_MAP
{
    {"ascend", ItvCv::ModelRepresentation::ascend},
    {"caffe", ItvCv::ModelRepresentation::caffe},
    {"onnx", ItvCv::ModelRepresentation::onnx},
    {"openvino", ItvCv::ModelRepresentation::openvino},
    {UNKNOWN_KEY, ItvCv::ModelRepresentation::unknown}
};

const SubTypesMap_t SUB_TYPES_MAP
{
    {"BodySegments", ItvCv::SubTypes::BodySegments},
    {UNKNOWN_KEY, ItvCv::SubTypes::None}
};

const HumanPointMap_t HUMAN_POINT_TYPES_MAP
{
    {"nose", Pose::HumanPosePointType::humanPointNose},
    {"neck", Pose::HumanPosePointType::humanPointNeck},
    {"rShoulder", Pose::HumanPosePointType::humanPointRightShoulder},
    {"rElbow", Pose::HumanPosePointType::humanPointRightElbow},
    {"rWrist", Pose::HumanPosePointType::humanPointRightWrist},
    {"lShoulder", Pose::HumanPosePointType::humanPointLeftShoulder},
    {"lElbow", Pose::HumanPosePointType::humanPointLeftElbow},
    {"lWrist", Pose::HumanPosePointType::humanPointLeftWrist},
    {"rHip", Pose::HumanPosePointType::humanPointRightHip},
    {"rKnee", Pose::HumanPosePointType::humanPointRightKnee},
    {"rAnkle", Pose::HumanPosePointType::humanPointRightAnkle},
    {"lHip", Pose::HumanPosePointType::humanPointLeftHip},
    {"lKnee", Pose::HumanPosePointType::humanPointLeftKnee},
    {"lAnkle", Pose::HumanPosePointType::humanPointLeftAnkle},
    {"rEye", Pose::HumanPosePointType::humanPointRightEye},
    {"lEye", Pose::HumanPosePointType::humanPointLeftEye},
    {"rEar", Pose::HumanPosePointType::humanPointRightEar},
    {"lEar", Pose::HumanPosePointType::humanPointLeftEar},
    {UNKNOWN_KEY, Pose::HumanPosePointType::humanPointUnknown}
};

const BodySegmentMap_t BODY_SEGMENTS_MAP
{
    {"Head", Segments::BodySegment::Head},
    {"Face", Segments::BodySegment::Face},
    {"Shoulder", Segments::BodySegment::Shoulder},
    {"Forearm", Segments::BodySegment::Forearm},
    {"Hand", Segments::BodySegment::Hand},
    {"Body", Segments::BodySegment::Body},
    {"Hips", Segments::BodySegment::Hips},
    {"Shin", Segments::BodySegment::Shin},
    {"Foot", Segments::BodySegment::Foot},
    {"All", Segments::BodySegment::All},
    {UNKNOWN_KEY, Segments::BodySegment::Unknown}
};

const ObjectTypeMap_t OBJECT_DETECTION_MAP
{
    {"Animal", ObjectType::itvcvObjectAnimal},
    {"Bicyclist", ObjectType::itvcvObjectBicyclist},
    {"Box", ObjectType::itvcvObjectBox},
    {"Bus", ObjectType::itvcvObjectBus},
    {"Car", ObjectType::itvcvObjectCar},
    {"Cat", ObjectType::itvcvObjectCat},
    {"Child", ObjectType::itvcvObjectChild},
    {"Face", ObjectType::itvcvObjectFace},
    {"Fire", ObjectType::itvcvObjectFire},
    {"GasBottle", ObjectType::itvcvObjectGasBottle},
    {"GroupOfHumans", ObjectType::itvcvObjectGroupOfHumans},
    {"Human", ObjectType::itvcvObjectHuman},
    {"Motorcycle", ObjectType::itvcvObjectMotorcycle},
    {"Noise", ObjectType::itvcvObjectNoise},
    {"Reserved", ObjectType::itvcvObjectReserved},
    {"RobotDog", ObjectType::itvcvObjectRobotDog},
    {"Sack", ObjectType::itvcvObjectSack},
    {"Smoke", ObjectType::itvcvObjectSmoke},
    {"Other", ObjectType::itvcvObjectOther},
    {"Vehicle", ObjectType::itvcvObjectVehicle},
    {UNKNOWN_KEY, ObjectType::itvcvObjectUndefined}
};

const AnalyzerTypeMap_t NET_TYPES_MAP
{
    {"Detection", itvcvAnalyzerType::itvcvAnalyzerSSD},
    {"SemanticSegmentation", itvcvAnalyzerType::itvcvAnalyzerMaskSegments},
    {"Classification", itvcvAnalyzerType::itvcvAnalyzerClassification},
    {"ReId", itvcvAnalyzerType::itvcvAnalyzerSiamese},
    {"PoseEstimation", itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator},
    {"PointDetection", itvcvAnalyzerType::itvcvAnalyzerPointDetection},
    {UNKNOWN_KEY, itvcvAnalyzerType::itvcvAnalyzerUnknown}
};

const PixelFormatMap_t PIXEL_FORMAT_MAP
{
    {"BGR", ItvCv::PixelFormat::BGR},
    {"NV12", ItvCv::PixelFormat::NV12},
    {"RGB", ItvCv::PixelFormat::RGB},
    {"GRAY", ItvCv::PixelFormat::GRAY},
    {UNKNOWN_KEY, ItvCv::PixelFormat::Unspecified}
};

const ResizePolicyMap_t RESIZE_POLICY_MAP
{
    {"AsIs", ItvCv::ResizePolicy::AsIs},
    {"ProportionalWithPaddedBorders", ItvCv::ResizePolicy::ProportionalWithPaddedBorders},
    {"ProportionalWithRepeatedBorders", ItvCv::ResizePolicy::ProportionalWithRepeatedBorders},
    {UNKNOWN_KEY, ItvCv::ResizePolicy::Unspecified}
};

const DataTypeMap_t WEIGHT_TYPE_MAP
{
    {"FP16", ItvCv::DataType::FP16},
    {"FP32", ItvCv::DataType::FP32},
    {"INT8", ItvCv::DataType::INT8},
    {UNKNOWN_KEY, ItvCv::DataType::Unknown}
};
}

template<typename K, typename V>
std::map<V, K> ReverseMap(const std::map<K, V>& m)
{
    std::map<V, K> r;
    for (const auto& kv : m)
    {
        if(r.find(kv.second) != r.end())
        {
            continue;
        }
        r[kv.second] = kv.first;
    }
    return r;
}

template<ObjectType Type_t>
const auto& GetAvailablePoseMapByObjectType()
{
    throw std::runtime_error(fmt::format("Unsupported object type:{}", Type_t));
}

template<>
inline const auto& GetAvailablePoseMapByObjectType<ObjectType::itvcvObjectHuman>()
{
    return TagToEnumMaps::HUMAN_POINT_TYPES_MAP;
}

template<typename EnumType_t>
EnumType_t ConvertStringToEnum(const std::string& value, const std::map<std::string, EnumType_t>& availableTypes)
{
    if (availableTypes.find(TagToEnumMaps::UNKNOWN_KEY) == availableTypes.end())
    {
        throw std::logic_error(fmt::format("Doesn't contain \"{}\" key", TagToEnumMaps::UNKNOWN_KEY));
    }
    const auto mapElement = availableTypes.find(value.c_str());
    if (mapElement == availableTypes.end())
    {
        return availableTypes.at(TagToEnumMaps::UNKNOWN_KEY);
    }
    return mapElement->second;
}

template<typename EnumType, typename inputType>
std::string ConvertEnumToString(inputType value, const std::map<EnumType, std::string>& reverseAvailableTypes)
{
    const auto enumType = static_cast<EnumType>(value);
    return reverseAvailableTypes.at(enumType);
}

#endif
