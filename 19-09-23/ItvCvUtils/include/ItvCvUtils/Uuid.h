#ifndef ITVCVUTILS_UUID_H
#define ITVCVUTILS_UUID_H

#include <ItvCvUtils/ItvCvUtils.h>
#include <boost/uuid/uuid.hpp>
#include <string>

namespace ItvCv
{
namespace Utils
{
ITVCV_UTILS_API boost::uuids::uuid GenerateUuid();
ITVCV_UTILS_API std::string ConvertUuidToString(boost::uuids::uuid const& u);

} // namespace Utils
} // namespace ItvCv

#endif
