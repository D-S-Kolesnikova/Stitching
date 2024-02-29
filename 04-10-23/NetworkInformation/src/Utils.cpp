#include "TagsConstant.h"
#include <NetworkInformation/Utils.h>

#include <boost/mpl/map.hpp>
#include <boost/mpl/pair.hpp>
#include <boost/mpl/at.hpp>

#include <fmt/format.h>

#include <set>
#include <cfloat>

namespace
{
namespace boostMpl = boost::mpl;
typedef boostMpl::map<
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerClassification)>, ItvCv::NetworkInformation::Labels_t>,
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerSSD)>, ItvCv::NetworkInformation::Labels_t>,
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerSiamese)>, ItvCv::ReidParams>,
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerMaskSegments)>, ItvCv::SemanticSegmentationParameters>,
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)>, ItvCv::PoseNetworkParams>,
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerPointMatcher)>, ItvCv::PointMatchingParams>,
    boostMpl::pair<boostMpl::int_<static_cast<int>(itvcvAnalyzerType::itvcvAnalyzerUnknown)>, boost::blank>
    > NetParametersMapTypes_t;

template<itvcvAnalyzerType AnalyzerType_t>
void ValidationNetParameters(const ItvCv::NetworkInformation& data)
{
    using Data_t = typename boost::mpl::at<NetParametersMapTypes_t, boostMpl::int_<static_cast<int>(AnalyzerType_t)>>::type;
    if (typeid(Data_t) != data.networkParams.type())
    {
        throw NetworkInfoUtils::ValidationException(
            fmt::format("Incorrect NetParametersType:{} for this AnalyzerType:{}", data.networkParams.which(), data.commonParams.analyzerType),
            NetworkInfoUtils::ValidationError::BadNetworkInformation);
    }
    const auto& netParamsData = boost::get<Data_t>(data.networkParams);
    NetworkInfoUtils::Validation(netParamsData);
}

template<>
void ValidationNetParameters<itvcvAnalyzerUnknown>(const ItvCv::NetworkInformation& data)
{
    throw NetworkInfoUtils::ValidationException(
        fmt::format("Unsupported AnalyzerType:{}", itvcvAnalyzerUnknown),
        NetworkInfoUtils::ValidationError::BadCommonParams);
}

template<ObjectType ObjectType_t>
void LinkerValidation(const ItvCv::PoseLinker& poseLinker, bool pafIsRequired)
{
    if (pafIsRequired)
    {
        if (!poseLinker.paf)
        {
            throw NetworkInfoUtils::ValidationException(
                fmt::format("Linker error. Must be contained paf linker"),
                NetworkInfoUtils::ValidationError::BadPoseNetworkParams);
        }
        if (poseLinker.paf->empty())
        {
            throw NetworkInfoUtils::ValidationException(
                fmt::format("Linker error. Got empty Paf"),
                NetworkInfoUtils::ValidationError::BadPoseNetworkParams);
        }
    }

    if (poseLinker.heatmap.empty())
    {
        throw NetworkInfoUtils::ValidationException(
            fmt::format("Linker error. Got empty heatmap vector"),
            NetworkInfoUtils::ValidationError::BadPoseNetworkParams);
    }

    const auto& mapTypes = ReverseMap(GetAvailablePoseMapByObjectType<ObjectType_t>());
    using EnumType_t = decltype(mapTypes.begin()->first);
    for (const auto pointIntType : poseLinker.heatmap)
    {
        const auto pointType = static_cast<EnumType_t>(pointIntType);
        const auto& mapIterator = mapTypes.find(pointType);
        if (mapIterator == mapTypes.end() || mapIterator->second == TagToEnumMaps::UNKNOWN_KEY)
        {
            throw NetworkInfoUtils::ValidationException(
                fmt::format("Linker error. Got unsupported pointType:{}", pointType),
                NetworkInfoUtils::ValidationError::BadPoseNetworkParams);
        }
    }
}
}

namespace NetworkInfoUtils
{
ITVCV_NETINFO_API void Validation(const ItvCv::NetworkInformation& data)
{
    Validation(data.inputParams);
    Validation(data.commonParams);

    switch (data.commonParams.analyzerType)
    {
    case itvcvAnalyzerClassification:
        ValidationNetParameters<itvcvAnalyzerClassification>(data);
        break;
    case itvcvAnalyzerSSD:
        ValidationNetParameters<itvcvAnalyzerSSD>(data);
        break;
    case itvcvAnalyzerSiamese:
        ValidationNetParameters<itvcvAnalyzerSiamese>(data);
        break;
    case itvcvAnalyzerMaskSegments:
        ValidationNetParameters<itvcvAnalyzerMaskSegments>(data);
        break;
    case itvcvAnalyzerHumanPoseEstimator:
        ValidationNetParameters<itvcvAnalyzerHumanPoseEstimator>(data);
        break;
    case itvcvAnalyzerPointMatcher:
        ValidationNetParameters<itvcvAnalyzerPointMatcher>(data);
        break;
    default:
        ValidationNetParameters<itvcvAnalyzerUnknown>(data);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::InputParams& data)
{
    if (data.inputHeight <= 0 || data.inputWidth <= 0)
    {
        throw ValidationException(
            fmt::format("Wrong input shape({}x{})", data.inputHeight, data.inputWidth),
            ValidationError::BadInputParams);
    }

    if (data.numChannels != static_cast<int>(data.normalizationValues.size()))
    {
        throw ValidationException(
            fmt::format("Count of normalization values does not match with count channels. {}!={}", data.normalizationValues.size(), data.numChannels),
            ValidationError::BadInputParams);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::CommonParams& data)
{
    if (data.analyzerType == itvcvAnalyzerUnknown)
    {
        throw ValidationException(
            fmt::format("Unsupported analyzer type:{}", data.analyzerType),
            ValidationError::BadInputParams);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::NetworkInformation::Labels_t& data)
{
    std::set<int> usedPosition;
    if (data.empty())
    {
        throw ValidationException(
            "Labels is empty",
            ValidationError::BadLabelsParams);
    }

    const auto maxPosition = static_cast<int>(data.size()) - 1;
    for (const auto& label : data)
    {
        if (usedPosition.find(label.position) != usedPosition.end())
        {
            throw ValidationException(
                fmt::format("Label(name:{}, objectType:{}) has duplicated position = {}", label.name, label.type.objectType, label.position),
                ValidationError::BadLabelsParams);
        }

        if (label.position > maxPosition)
        {
            throw ValidationException(
                fmt::format("Label(name:{}, objectType:{}) has position more than max possible position. {} > {}", label.name, label.type.objectType, label.position, maxPosition),
                ValidationError::BadLabelsParams);
        }
        usedPosition.emplace(label.position);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::ReidParams& data)
{
    if (data.datasetGroup.empty())
    {
        throw ValidationException(
            "Dataset group is empty",
            ValidationError::BadReidParams);
    }

    if (data.vectorSize <= 0)
    {
        throw ValidationException(
            fmt::format("Vector size must be more then 0. vectorSize={}", data.vectorSize),
            ValidationError::BadReidParams);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::SemanticSegmentationParameters& data)
{
    Validation(data.labels);
}

ITVCV_NETINFO_API void Validation(const ItvCv::PoseNetworkParams& data)
{
    const auto isOpenPoseParams = data.params.type() == typeid(ItvCv::OpenPoseParams);
    const auto isAEPoseParams = data.params.type() == typeid(ItvCv::AEPoseParams);

    if (data.type.objectType == itvcvObjectHuman)
    {
        LinkerValidation<itvcvObjectHuman>(data.linker, isOpenPoseParams);
    }
    else
    {
        throw  ValidationException(
            fmt::format("Got unsupported Pose objectType:{}", data.type.objectType),
            ValidationError::BadPoseNetworkParams);
    }

    if (isOpenPoseParams)
    {
        Validation(boost::get<ItvCv::OpenPoseParams>(data.params));
    }
    else if(isAEPoseParams)
    {
        Validation(boost::get<ItvCv::AEPoseParams>(data.params));
    }
    else
    {
        throw ValidationException(
            "ItvCv::PoseNetworkParams must be contain parameters",
            ValidationError::BadPoseNetworkParams);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::OpenPoseParams& data)
{
    if(data.boxSize <= 0)
    {
        throw ValidationException(
            fmt::format("Incorrect parameter BoxSize:{}, must be more then 0", data.boxSize),
            ValidationError::BadOpenPoseParams);
    }
    if (data.upSampleRatio <= FLT_EPSILON)
    {
        throw ValidationException(
            fmt::format("Incorrect upSampleRatio:{}, must be more then 0", data.boxSize),
            ValidationError::BadOpenPoseParams);
    }

    if (data.stride < FLT_EPSILON)
    {
        throw ValidationException(
            fmt::format("Incorrect stride:{}, must be more then 0", data.boxSize),
            ValidationError::BadOpenPoseParams);
    }

    if (data.midPointsRatioThreshold < FLT_EPSILON
        || data.midPointsRatioThreshold > (1 + FLT_EPSILON))
    {
        throw ValidationException(
            fmt::format("Incorrect midPointsRatioThreshold:{}, must be in range [0..1]", data.boxSize),
            ValidationError::BadOpenPoseParams);
    }

    if (data.midPointsScoreThreshold < FLT_EPSILON
        || data.midPointsScoreThreshold >(1 + FLT_EPSILON))
    {
        throw ValidationException(
            fmt::format("Incorrect midPointsScoreThreshold:{}, must be in range [0..1]", data.boxSize),
            ValidationError::BadOpenPoseParams);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::AEPoseParams& data)
{
    if (data.maxObjects <= 0)
    {
        throw ValidationException(
            fmt::format("Incorrect parameter maxObjects:{}, must be more then 0", data.maxObjects),
            ValidationError::BadAEPoseParams);
    }
    if (data.confidenceThreshold < FLT_EPSILON)
    {
        throw ValidationException(
            fmt::format("Incorrect confidenceThreshold:{}, must be more then 0", data.confidenceThreshold),
            ValidationError::BadAEPoseParams);
    }

    if (data.peakThreshold < FLT_EPSILON
        || data.peakThreshold >(1 + FLT_EPSILON))
    {
        throw ValidationException(
            fmt::format("Incorrect peakThreshold:{}, must be in range [0..1]", data.peakThreshold),
            ValidationError::BadAEPoseParams);
    }

    if (data.tagsThreshold < FLT_EPSILON
        || data.tagsThreshold >(1 + FLT_EPSILON))
    {
        throw ValidationException(
            fmt::format("Incorrect tagsThreshold:{}, must be in range [0..1]", data.tagsThreshold),
            ValidationError::BadAEPoseParams);
    }
}

ITVCV_NETINFO_API void Validation(const ItvCv::PointMatchingParams& data)
{
    if (data.maxNumKeypoints == 0)
    {
        throw ValidationException(
            "maxNumKeypoints is empty",
            ValidationError::BadPointMatchingParams);
    }
    if (data.numDescriptors == 0)
    {
        throw ValidationException(
            "numDescriptors is empty",
            ValidationError::BadPointMatchingParams);
    }
}

}