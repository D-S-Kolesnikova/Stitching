#include <ItvCvUtils/Envar.h>

#include <boost/lexical_cast.hpp>

#if defined(_WIN32)
#include <windows.h>
#else
#include <stdlib.h>
#endif

#include <boost/filesystem/operations.hpp>

namespace
{
    using namespace ItvCvUtils;

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4505)
#endif
#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

    template<class TValue>
    static bool Set(const std::string &name, const TValue &value)
    {
        try
        {
            return CEnvar::Set(name, boost::lexical_cast<std::string>(value));
        }
        catch (const std::exception &) {}
        return false;
    }
#ifdef _WIN32
#pragma warning(pop)
#endif
#if __GNUC__
#pragma GCC diagnostic pop
#endif

    template<class TValue>
    static bool Lookup(const std::string &name, TValue &value)
    {
        try
        {
            std::string var;
            if (CEnvar::Lookup(name, var))
            {
                value = boost::lexical_cast<TValue>(var);
                return true;
            }
        }
        catch (const std::exception &) {}
        return false;
    }

    template<class TValue>
    static TValue Get(const std::string &name, const TValue &def)
    {
        TValue res = def;
        Lookup(name, res);
        return res;
    }
} // anonymous namespace

namespace ItvCvUtils
{
    bool CEnvar::Set(const std::string &name, const std::string &value)
    {
#if defined(_WIN32)
        return 0 != SetEnvironmentVariableA(name.c_str(), value.c_str());
#elif defined(__linux__)
        return 0 == setenv(name.c_str(), value.c_str(), 1);
#endif
    }

    bool CEnvar::Lookup(const std::string &name, std::string &value)
    {
#if defined(_WIN32)
        std::vector<char> buff(std::numeric_limits<short>::max());
        if (GetEnvironmentVariableA(name.c_str(), &buff[0], static_cast<DWORD>(buff.size())))
        {
            value.assign(&buff[0]);
            return true;
        }

#elif defined(__linux__)
        const char *var = getenv(name.c_str());
        if (0 != var)
        {
            value.assign(var);
            return true;
        }
#endif
        return false;
    }

    std::string CEnvar::CvAscendScheduler()
    {
        return Get<std::string>("HUAWEI_ASCEND_SCHEDULER", "");
    }

    std::string CEnvar::RuntimeDirectoryPrivate()
    {
        return Get<std::string>("RUNTIME_DIRECTORY_PRIVATE", boost::filesystem::temp_directory_path().string());
    }

    std::string CEnvar::RuntimeDirectoryShared()
    {
        return Get<std::string>("RUNTIME_DIRECTORY_SHARED", boost::filesystem::temp_directory_path().string());
    }

    std::string CEnvar::CvDumpInferenceInput()
    {
        return Get<std::string>("CV_DUMP_INFERENCE_INPUT", "0");
    }

    std::string CEnvar::CvDumpInferenceOutput()
    {
        return Get<std::string>("CV_DUMP_INFERENCE_OUTPUT", "0");
    }

    std::string CEnvar::CvDumpInferenceLayers()
    {
        return Get<std::string>("CV_DUMP_INFERENCE_LAYERS", "0");
    }

    std::string CEnvar::CvDumpDir()
    {
        return Get<std::string>("CV_DUMP_DIR", "");
    }

    std::string CEnvar::GpuCacheDir()
    {
        return Get<std::string>("GPU_CACHE_DIR", "");
    }

    bool CEnvar::CvAscendResizeInputOnHW()
    {
        return Get<bool>("CV_ASCEND_RESIZE_INPUT_ON_HW", "1");
    }

}


