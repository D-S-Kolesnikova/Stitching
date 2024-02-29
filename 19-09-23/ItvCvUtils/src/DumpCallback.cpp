#include <ItvCvUtils/DumpCallback.h>
#include <ItvCvUtils/Logf.h>

#include <mutex>
#include <boost/filesystem.hpp>
#include <boost/filesystem/detail/utf8_codecvt_facet.hpp>
#include <boost/format.hpp>
#include <boost/interprocess/detail/os_thread_functions.hpp>

#include <cstdlib>
#include <csignal>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <breakpad/client/windows/handler/exception_handler.h>
#include <breakpad/client/windows/common/ipc_protocol.h>

namespace {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif // _MSC_VER

    inline std::wstring MultibyteToWideString(const char* str)
    {
        mbstate_t st;
        memset(&st, 0, sizeof(st));

        size_t len = mbsrtowcs(NULL, &str, 0, &st);
        if (len == (size_t) -1)
            throw std::runtime_error("Illegal multibyte string sequence");

        if (len == 0)
            return std::wstring();

        std::vector<wchar_t> buffer(len + 1, 0);
        memset(&st, 0, sizeof(st));
        size_t written = mbsrtowcs(&buffer[0], &str, buffer.size(), &st);
        return std::wstring(&buffer[0], written);
    }

    inline std::wstring s2ws(const char* str)
    {
        try
        {
            return MultibyteToWideString(str);
        }
        catch(const std::exception&)
        {
            return std::wstring(L"???");
        }
    }

#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

    inline std::wstring s2ws(const std::string& str)
    {
        return s2ws(str.c_str());
    }

} // anonymous namespace

std::unique_ptr<google_breakpad::ExceptionHandler> g_breakpadExceptionHandler = nullptr;
std::wstring g_dumpDirPath(L".");

struct CallbackContext
{
    std::wstring prefix;
    ITV8::ILogger* logger;
};

namespace
{
    std::mutex g_mutex;
    CallbackContext g_ctx;
    std::atomic_flag g_terminate_handled = ATOMIC_FLAG_INIT;
}

void SignalHandler(int signal)
{
    std::signal(SIGABRT, &SignalHandler);
    if (signal == SIGABRT)
    {
        ItvCvUtils::HandleTerminate();
    }
}


bool CreateDumpCallback(const wchar_t* dump_path,
    const wchar_t* minidump_id,
    void* context,
    EXCEPTION_POINTERS* exinfo,
    MDRawAssertionInfo* assertion,
    bool succeeded)
{
    CallbackContext ctx;
    if (context == nullptr)
    {
        std::lock_guard<std::mutex> lock (g_mutex);
        ctx = g_ctx;    
    } 
    else ctx = *reinterpret_cast<CallbackContext*>(context); 

    static wchar_t old_name[MAX_PATH], new_name[MAX_PATH];
    _snwprintf(old_name, MAX_PATH, L"%s/%s.dmp", g_dumpDirPath.c_str(), minidump_id);
    _snwprintf(new_name, MAX_PATH, L"%s/%s.%s.dmp", g_dumpDirPath.c_str(), ctx.prefix.c_str(), minidump_id);
    _wrename(old_name, new_name);
    return true;
}

namespace ItvCvUtils
{
void HandleTerminate()
{
    if (g_terminate_handled.test_and_set() == true)
    {
        // Other thread already handling error.
        // Process is expected to die before the following sleep is done.
        std::this_thread::sleep_for(std::chrono::hours(1));
    }
    if (g_breakpadExceptionHandler)
    {
        g_breakpadExceptionHandler->WriteMinidump();
    }
    TerminateProcess(GetCurrentProcess(), -1);
}

void SetupDumpHandler(ITV8::ILogger* logger, const boost::filesystem::path& dumpPath, const std::string& app, ProgramVersionInfo const& version, uint32_t dumpType)
{
    static std::wstring g_prod(s2ws(version.ProductName));
    static std::wstring g_app(s2ws(app));
    static std::wstring g_ver(s2ws(version.VersionString));
    static std::wstring g_rev(s2ws(version.RevisionHash));

    const int customInfoCount = 4;
    static google_breakpad::CustomInfoEntry customInfoEntries[customInfoCount] = {
        google_breakpad::CustomInfoEntry(L"prod", g_prod.c_str()),
        google_breakpad::CustomInfoEntry(L"app", g_app.c_str()),
        google_breakpad::CustomInfoEntry(L"ver", g_ver.c_str()),
        google_breakpad::CustomInfoEntry(L"rev", g_rev.c_str()),
    };

    try
    {
        if (!boost::filesystem::exists(dumpPath))
        {
            boost::filesystem::create_directory(dumpPath);
        }
        g_dumpDirPath = dumpPath.wstring();
    }
    catch (std::exception& e)
    {
        throw std::runtime_error(std::string("SetupDumpHandler(): bad path: ") + e.what());
    }

    unsigned long pid = boost::interprocess::ipcdetail::get_current_process_id();

    {
        std::lock_guard<std::mutex> lock (g_mutex);        
        g_ctx = CallbackContext{ boost::filesystem::path(app + "." + std::to_string(pid)).wstring(), logger };
    }

    // Reset exception handler now to prevent wrong destuction/construction order.
    g_breakpadExceptionHandler.reset();

    google_breakpad::CustomClientInfo custom_info = { customInfoEntries, customInfoCount };
    g_breakpadExceptionHandler.reset(new google_breakpad::ExceptionHandler(g_dumpDirPath,
        NULL,
        CreateDumpCallback,
        nullptr,
        google_breakpad::ExceptionHandler::HANDLER_ALL,
        static_cast<MINIDUMP_TYPE>(dumpType), (wchar_t*)NULL, 
        &custom_info));
    auto previous_handler = std::signal(SIGABRT, &SignalHandler);
    if (previous_handler == SIG_ERR)
    {
        ITVCV_LOG(logger, ITV8::LOG_ERROR, "Can't setup abort handler!");
    }
    else
    {
        ITVCV_LOG(logger, ITV8::LOG_INFO, "Abort handler is set up!");
    }
    std::set_terminate(&HandleTerminate);
}
bool WriteMinidump(ITV8::ILogger* logger, std::string const& dumpPrefixAddon)
{
    if (g_breakpadExceptionHandler)
    {
        CallbackContext ctx;
        {
            std::lock_guard<std::mutex> lock (g_mutex);
            ctx = CallbackContext { g_ctx.prefix, logger };
        } 

        if (!dumpPrefixAddon.empty())
        {
            ctx.prefix += L'.';
            ctx.prefix += boost::filesystem::path(dumpPrefixAddon).wstring();
        }
        return google_breakpad::ExceptionHandler::WriteMinidump(g_dumpDirPath, CreateDumpCallback, &ctx);
    }
    return false;
}
bool WriteMinidumpForChild(ITV8::ILogger* logger, HANDLE child, std::string const& dumpPrefix)
{
    CallbackContext ctx { s2ws(dumpPrefix), logger };
    return google_breakpad::ExceptionHandler::WriteMinidumpForChild(
        child,
        0,
        g_dumpDirPath,
        CreateDumpCallback,
        &ctx,
        MiniDumpNormal
    );
}
}

#elif defined(__linux__)

#include <breakpad/client/linux/handler/exception_handler.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <memory>
#include <signal.h>

std::unique_ptr<google_breakpad::ExceptionHandler> g_breakpadExceptionHandler = nullptr;
std::string g_dumpDirPath(".");

struct CallbackContext
{
    pid_t pid;
    std::string prefix;
    ItvCvUtils::ProgramVersionInfo version;
    ITV8::ILogger * logger;
    bool passDumpToTheSystem;
};

namespace 
{
    std::mutex g_mutex;
    CallbackContext g_ctx;

    std::thread dump_generator_thread;
    std::once_flag thread_flag;
}

bool HandleDump(const char* path,
    CallbackContext const& ctx,
    bool /*succeeded*/)
{
    static char buf[PATH_MAX];
    auto safe_str = [](const char* str) { return str ? str : "N/A"; };
    snprintf(
        buf, sizeof(buf), "handle-crash-dump %d '%s' '%s' '%s-%s built %s'",
        ctx.pid,
        path,
        ctx.prefix.c_str(),
        safe_str(ctx.version.VersionString), safe_str(ctx.version.RevisionHash), safe_str(ctx.version.BuildTime)
    );
    auto rc = system(buf);
    (void) rc;
    return !ctx.passDumpToTheSystem;
}

bool CreateDumpCallback(const google_breakpad::MinidumpDescriptor& descriptor,
    void* context,
    bool succeeded)
{
    CallbackContext ctx;

    if (context == nullptr)
    {
        std::lock_guard<std::mutex> lock (g_mutex);
        ctx = g_ctx;    
    } 
    else  ctx = *(reinterpret_cast<CallbackContext*>(context));

    return HandleDump(descriptor.path(), ctx, succeeded);
}

bool WriteMinidumpAndHandleItOutsideOfCallbackToAvoidFileNotFoundError(ITV8::ILogger* logger, std::string const& dumpPrefixAddon)
{
    if (!g_breakpadExceptionHandler)
        return false;

    struct CallbackArgsStorage
    {
        std::string dumpFilePath;
        bool succeeded;

        CallbackArgsStorage()
        {
            const size_t estimatedPathSize = boost::filesystem::absolute(g_dumpDirPath).size() + sizeof("/5a583639-fb92-4aca-19d8b788-bb2d247c.dmp");
            dumpFilePath.reserve(estimatedPathSize);
        }

        static bool GrabCallbackArgs(
            const google_breakpad::MinidumpDescriptor& descriptor,
            void* context,
            bool succeeded)
        {
            auto ctx = reinterpret_cast<CallbackArgsStorage*>(context);
            ctx->dumpFilePath = descriptor.path();
            ctx->succeeded = succeeded;
            return true;
        } 
    };
    CallbackArgsStorage args;
    bool const written = google_breakpad::ExceptionHandler::WriteMinidump(
        g_dumpDirPath,
        CallbackArgsStorage::GrabCallbackArgs,
        &args
    );
    if (!written)
        return false;

    CallbackContext ctx;
    {
        std::lock_guard<std::mutex> lock (g_mutex);
        ctx = CallbackContext { getpid(), g_ctx.prefix, g_ctx.version, logger, false };
    }

    if (!dumpPrefixAddon.empty())
    {
        ctx.prefix += '.';
        ctx.prefix += dumpPrefixAddon;
    }
    HandleDump(args.dumpFilePath.c_str(), ctx, args.succeeded);
    return true;
}

namespace ItvCvUtils
{
      
void start_dump_generator_thread ()
{    
    auto sigset_ptr = std::make_unique<sigset_t>();
    sigemptyset(sigset_ptr.get());
    sigaddset(sigset_ptr.get(), SIGUSR1);
    pthread_sigmask(SIG_BLOCK, sigset_ptr.get(), NULL);
            
    dump_generator_thread = std::thread ([allow_SIGUSR1_mask = std::move(sigset_ptr)] ()
    {
        int sig;
        while (true)
        {
            sigwait(allow_SIGUSR1_mask.get(), &sig);
            
            if (sig == SIGUSR1)
            {          
                std::lock_guard<std::mutex> lock (g_mutex);
                auto logger = g_ctx.logger;
                if (WriteMinidump(logger, "generated_by_SIGUSR1_"))
                    ITVCV_LOG(logger, ITV8::LOG_INFO, "Requested by SIGUSR1 minidump has been created");
                else
                    ITVCV_LOGF(logger, ITV8::LOG_ERROR, "Failed to write requested by SIGUSR1 minidump for process: {}", getpid());
            }
        }
    });
    dump_generator_thread.detach();
}

void SetupDumpHandler(ITV8::ILogger* logger, const boost::filesystem::path& dumpPath, const std::string& app, ProgramVersionInfo const& version, uint32_t dumpType)
{
    try
    {
        if (!boost::filesystem::exists(dumpPath))
        {
            boost::filesystem::create_directory(dumpPath);
        }

        g_dumpDirPath = dumpPath.string();
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error(std::string("SetupDumpHandler(): bad path: ") + e.what());
    }

    {
        std::lock_guard<std::mutex> lock (g_mutex);
        g_ctx = CallbackContext { getpid(), app + "." + std::to_string(getpid()), version, logger, true };
    }

    // Reset exception handler now to prevent wrong destuction/construction order.
    g_breakpadExceptionHandler.reset();

    google_breakpad::MinidumpDescriptor desc{g_dumpDirPath};
    g_breakpadExceptionHandler.reset(new google_breakpad::ExceptionHandler(
        desc,
        nullptr,
        CreateDumpCallback,
        nullptr,
        true,
        -1));

    std::call_once(thread_flag, start_dump_generator_thread);
}

bool WriteMinidump(ITV8::ILogger* logger, std::string const& dumpPrefixAddon)
{
    return WriteMinidumpAndHandleItOutsideOfCallbackToAvoidFileNotFoundError(logger, dumpPrefixAddon);
}

bool WriteMinidumpForChild(ITV8::ILogger* logger, pid_t child, std::string const& dumpPrefix)
{
    CallbackContext ctx;
    {
        std::lock_guard<std::mutex> lock (g_mutex);
        ctx = CallbackContext { child, dumpPrefix, g_ctx.version, logger, false };
    } 
    return google_breakpad::ExceptionHandler::WriteMinidumpForChild(
        child,
        child,
        g_dumpDirPath,
        CreateDumpCallback,
        &ctx 
    );
}
}
#else
namespace ItvCvUtils
{
void SetupDumpHandler(ITV8::ILogger* logger, const boost::filesystem::path& dumpPath, const std::string& app, ProgramVersionInfo const& version, uint32_t dumpType)
{
}
bool WriteMinidump()
{
    return false;
}
}
#endif
namespace ItvCvUtils
{
    void SetupDumpHandler(ITV8::ILogger* logger, const std::string& dumpPathStr, const std::string& app, ProgramVersionInfo const& version, uint32_t dumpType)
    {
        SetupDumpHandler(logger, boost::filesystem::path(dumpPathStr), app, version, dumpType);
    }
}
