#ifndef ITVCVUTILS_UNICODE_H
#define ITVCVUTILS_UNICODE_H

#include <string>

#include <ItvCvUtils/ItvCvUtils.h>

namespace ItvCv
{
namespace Utils
{
ITVCV_UTILS_API std::string ToUtf8(wchar_t const* wideStr);
ITVCV_UTILS_API std::wstring FromUtf8(char const* str);

} // namespace Utils
} // namespace ItvCv

#endif
