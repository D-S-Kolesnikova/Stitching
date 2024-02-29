#include <ItvCvUtils/Unicode.h>

#include <boost/locale/encoding_utf.hpp>
#include <string>

ITVCV_UTILS_API std::string ItvCv::Utils::ToUtf8(wchar_t const* wideStr)
{
    return boost::locale::conv::utf_to_utf<char>(wideStr);
}

ITVCV_UTILS_API std::wstring ItvCv::Utils::FromUtf8(char const* str)
{
    return boost::locale::conv::utf_to_utf<wchar_t>(str);
}
