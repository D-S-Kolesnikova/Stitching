#ifndef NETWORKINFORMATION_UTILS_H
#define NETWORKINFORMATION_UTILS_H

#include <NetworkInformation/NetworkInformationLib.h>

namespace NetworkInfoUtils
{
enum class ValidationError
{
    NoError,
    Other,
    BadNetworkInformation,
    BadInputParams,
    BadCommonParams,
    BadLabelsParams,
    BadReidParams,
    BadSemanticSegmentationParameters,
    BadPointMatchingParams,
    BadPoseNetworkParams,
    BadOpenPoseParams,
    BadAEPoseParams
};

class ValidationException: public std::runtime_error
{
using Base = std::runtime_error;

public:
    explicit ValidationException(const std::string& errorMsg, ValidationError error = ValidationError::Other)
    : Base(errorMsg.c_str())
    , m_errorType(error)
    {
    }
    explicit ValidationException(const char* errorMsg, ValidationError error = ValidationError::Other)
        : Base(errorMsg)
        , m_errorType(error)
    {
    }
    ValidationError GetError() const
    {
        return m_errorType;
    }
private:
    ValidationError m_errorType;
};

//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::NetworkInformation& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::InputParams& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::CommonParams& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::NetworkInformation::Labels_t& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::ReidParams& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::SemanticSegmentationParameters& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::PoseNetworkParams& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::OpenPoseParams& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::AEPoseParams& data);
//Generate exception
ITVCV_NETINFO_API void Validation(const ItvCv::PointMatchingParams& data);

//noexcept
template<typename ArgType_t>
std::pair<ValidationError, std::string> ValidationParameters(const ArgType_t& data) noexcept
{
    try
    {
        NetworkInfoUtils::Validation(data);
    }
    catch(const ValidationException& error)
    {
        return std::make_pair(error.GetError(), error.what());
    }
    catch (...)
    {
        return std::make_pair(ValidationError::Other, "Got an unexpected error");
    }
    return std::make_pair(ValidationError::NoError, "No error");
}
}
#endif
