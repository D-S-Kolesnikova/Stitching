#pragma once

#include <string>
#include "ItvCvUtils.h"

namespace ItvCvUtils
{
/*
 * Set current thread name, for debug only
 * Keep it short: on Linux it is restricted to 16 characters, including
 * terminating '\0'
 */
ITVCV_UTILS_API void SetThreadName(std::string const& name);
}
