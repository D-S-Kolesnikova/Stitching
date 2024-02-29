#ifndef _NETINFORMATIONLIB_H_
#define _NETINFORMATIONLIB_H_

#include <ItvCvUtils/ItvCvDefs.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>
#include <ItvCvUtils/Frame.h>
#ifdef ITVCV_NETWORKINFORMATION_EXPORT
    #define ITVCV_NETINFO_API ITVCV_API_EXPORT
#else
    #define ITVCV_NETINFO_API ITVCV_API_IMPORT
#endif

#include <boost/variant.hpp>
#include <boost/optional/optional.hpp>

#include <memory>

namespace ItvCv
{

enum class DataType
{
    FP16,
    FP32,
    INT8,
    Unknown
};

enum class ModelRepresentation
{
    openvino,
    caffe,
    onnx,
    ascend,
    unknown
};

enum class SubTypes
{
    None,
    BodySegments
};

enum class ArchitectureType
{
    Unknown = 0,

    // classification nets end with 1
    GoogleNet = 01,
    EfficientNet = 11,

    // detection nets end with 2
    SSD_MobileNetv2 = 02,
    Yolo = 12,
    SSD_ResNet34 = 22,
    SSD_PyTorch = 32,

    // pose estimation nets end with 3
    Openpose18_MobileNet = 03,
    HigherHRNet_AE = 13,

    // segmentation nets end with 4
    CustomSegmentation9_v1 = 04,
    DDRNet = 14,

    //reid nets end with 5
    Osnetfpn = 05,

    //point detection nets
    SuperPoint = 06
};

enum class ResizePolicy
{
    Unspecified,
    AsIs,
    ProportionalWithRepeatedBorders,
    ProportionalWithPaddedBorders
};

struct Version
{
    int major;
    int minor;
    int patch;
};

struct PoseLinker
{
    struct SPafElement
    {
        int idChannel;
        int idPointFrom;
        int idPointTo;
    };

    boost::optional<std::vector<SPafElement>> paf;
    //heatmap: position i - channel position in network, value - position in Pose struct. Depend on ObjectType
    std::vector<int> heatmap;
};

struct OpenPoseParams
{
    int boxSize = 0;
    float stride = 0.f;
    float minPeaksDistance = 0.f;
    float midPointsScoreThreshold = 0.f;
    float midPointsRatioThreshold = 0.f;
    float upSampleRatio = 0.f;
};

//AE - Associative Embedding
//https://arxiv.org/abs/1611.05424
struct AEPoseParams
{
    float delta = 0.f;
    float peakThreshold = 0.f;
    float tagsThreshold = 0.f;
    float confidenceThreshold = 0.f;
    int maxObjects = 0;
};

class ComplexLabelType
{
private:
    using BodySegmentType_t = std::vector<Segments::BodySegment>;

    template<SubTypes EType_t, typename T = void>
    struct SelecterSubType { using Type_t = boost::blank; };

    //compiler https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282
    //ru https://github.com/cpp-ru/ideas/issues/322
    template<typename T>
    struct SelecterSubType<SubTypes::BodySegments, T> { using Type_t = BodySegmentType_t; };

public:
    using SubTypes_t = boost::variant<boost::blank, BodySegmentType_t>;

    ComplexLabelType()
    : objectType(ObjectType::itvcvObjectUndefined)
    , subType(boost::blank())
    {
    };

    explicit ComplexLabelType(ObjectType objectType, SubTypes_t subTypes = boost::blank())
    : objectType(objectType)
    , subType(subTypes)
    {
    };

    template<SubTypes Type_t, typename T = typename SelecterSubType<Type_t>::Type_t>
    boost::optional<T> GetSubtype() const
    {
        if (typeid(T) == subType.type())
        {
            return boost::optional<T>(boost::get<T>(subType));
        }
        return {};
    };

    SubTypes WichSubtype() const
    {
        if(subType.type() == typeid(BodySegmentType_t))
        {
            return SubTypes::BodySegments;
        }
        return SubTypes::None;
    };

    ObjectType objectType;
    SubTypes_t subType;
};

struct ReidParams
{
    ComplexLabelType type;
    int vectorSize = 256;
    //unique name of data set group
    std::string datasetGroup = "reid";
    int version = 0;
};

struct Label
{
    std::string name;
    int position;
    ComplexLabelType type;
    ClassReportType reportType;
};

struct SemanticSegmentationParameters
{
    std::vector<Label> labels;
    //isSingleChannel == false, In result tensor channel is label(Label::position)
    //isSingleChannel == true, In result tensor pixel is label(Label::position)
    bool isSingleChannel;
};

struct PointDetectionParams
{
    int maxNumKeypoints;
    int numDescriptors;
};

struct PoseNetworkParams
{
    PoseLinker linker;
    boost::variant<boost::blank, OpenPoseParams, AEPoseParams> params;
    float minSubsetScore = 0.f;
    ComplexLabelType type;
};

// TODO: type depending on DataType
struct NormalizationValue
{
    float mean = 0.f;
    float scale = 1.f;
};

//General parameters for all nets
struct CommonParams
{
    ArchitectureType architecture = ArchitectureType::Unknown;
    itvcvAnalyzerType analyzerType = itvcvAnalyzerUnknown;
    ModelRepresentation modelRepresentation = ModelRepresentation::unknown;
    DataType weightType;
};

//Parameters for preprocessing and input images
struct InputParams
{
    int numChannels = 0;
    int inputWidth = 0;
    int inputHeight = 0;
    bool supportedDynamicBatch = false;
    ResizePolicy resizePolicy = ResizePolicy::Unspecified;
    PixelFormat pixelFormat = PixelFormat::Unspecified;
    std::vector<NormalizationValue> normalizationValues;
};

//Optional parameters: info about network
struct ModelDescription
{
    std::string author;
    std::string task;
    std::string info;
    Version metadataVersion;
};

struct NetworkInformation
{
    using Labels_t = std::vector<Label>;
    using NetParams_t =
        boost::variant<
            boost::blank,
            Labels_t,
            SemanticSegmentationParameters,
            PoseNetworkParams,
            ReidParams,
            PointDetectionParams>;

    std::string modelData = "";

    InputParams inputParams;
    CommonParams commonParams;
    NetParams_t networkParams;

    boost::optional<ModelDescription> description;
};

using PNetworkInformation = std::shared_ptr<NetworkInformation>;

ITVCV_NETINFO_API ItvCv::Version GetMetadataParserVersion();

ITVCV_NETINFO_API PNetworkInformation GetNetworkInformation(const char* pathToEncryptedModel);

ITVCV_NETINFO_API std::string ConsumeWeightsData(PNetworkInformation const& netInfo);

ITVCV_NETINFO_API std::string GenerateMetadata(const NetworkInformation& netInfo);

ITVCV_NETINFO_API bool DumpNetworkInformationToAnn(const std::string& weightsData, const NetworkInformation& netInfo, const std::string& pathOut, std::int64_t byteSize = 1024);
}
#endif
