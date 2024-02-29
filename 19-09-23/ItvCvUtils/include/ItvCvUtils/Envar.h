#pragma once

#include "ItvCvUtils.h"

#include <string>

namespace ItvCvUtils {

class ITVCV_UTILS_API CEnvar
{
public:
    static bool Set(const std::string &name, const std::string &value);
    static bool Lookup(const std::string &name, std::string &value);
    static std::string CvAscendScheduler();
    static std::string RuntimeDirectoryPrivate();
    static std::string RuntimeDirectoryShared();
    static std::string CvDumpInferenceInput();
    static std::string CvDumpInferenceOutput();
    static std::string CvDumpInferenceLayers();
    static std::string CvDumpDir();
    static std::string GpuCacheDir();
    static bool CvAscendResizeInputOnHW();
};

} // namespace ItvCvUtils
