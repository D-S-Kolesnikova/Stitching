#pragma once

#include "ItvCvUtils.h"
#include "Log.h"

namespace boost { namespace filesystem { class path; } }

namespace ItvCvUtils {

struct ProgramVersionInfo
{
    const char* VendorName = nullptr;
    const char* ProductName = nullptr;
    const char* VersionString = nullptr;
    const char* RevisionHash = nullptr;
    const char* BuildTime = nullptr;
};

ITVCV_UTILS_API
void SetupDumpHandler(ITV8::ILogger*, const boost::filesystem::path& dumpPath, const std::string& app, ProgramVersionInfo const& version, uint32_t dumpType);
ITVCV_UTILS_API
void SetupDumpHandler(ITV8::ILogger*, const std::string& dumpPathStr, const std::string& app, ProgramVersionInfo const& version, uint32_t dumpType);
ITVCV_UTILS_API bool WriteMinidump(ITV8::ILogger* = nullptr, std::string const& dumpPrefixAddon = std::string());
#ifdef _WIN32
ITVCV_UTILS_API bool WriteMinidumpForChild(ITV8::ILogger*, void* hChild, std::string const& dumpPrefix);
ITVCV_UTILS_API void HandleTerminate();
#else
ITVCV_UTILS_API bool WriteMinidumpForChild(ITV8::ILogger*, pid_t child, std::string const& dumpPrefix);
#endif

} // namespace ItvCvUtils

