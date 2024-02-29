#include <ItvCvUtils/ThreadName.h>

#if defined _GNU_SOURCE
#include <pthread.h>

void SetThreadNameNative(std::string const& name)
{
    pthread_setname_np(pthread_self(), name.c_str());
}

#elif defined _WIN32
// See https ://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx

#include <windows.h>
// FIXME: there is a better way to set a thread name in new versions of windows
// Maybe it is worth to check in runtime if it is available and us it
// It allows to have the thread name in minidump files
// https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreaddescription
void SetThreadNameNative(DWORD dwThreadID, const char* threadName)
{
	const DWORD MS_VC_EXCEPTION = 0x406D1388;

#pragma pack(push,8)
	typedef struct tagTHREADNAME_INFO
	{
		DWORD dwType; // Must be 0x1000.
		LPCSTR szName; // Pointer to name (in user addr space).
		DWORD dwThreadID; // Thread ID (-1=caller thread).
		DWORD dwFlags; // Reserved for future use, must be zero.
	} THREADNAME_INFO;
#pragma pack(pop)

	THREADNAME_INFO info;
	info.dwType = 0x1000;
	info.szName = threadName;
	info.dwThreadID = dwThreadID;
	info.dwFlags = 0;
#pragma warning(push)
#pragma warning(disable: 6320 6322)
	__try
	{
		RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
	}
	__except (EXCEPTION_EXECUTE_HANDLER) { }
#pragma warning(pop)
}

void SetThreadNameNative(std::string const& name)
{
	SetThreadNameNative(static_cast<DWORD>(-1), name.c_str());
}

#else
#warning "SetThreadNameNative() is a no-op for this platform"
void SetThreadNameNative(std::string const& name)
{ }

#endif

namespace ItvCvUtils
{
void SetThreadName(std::string const& name)
{
    SetThreadNameNative(name);
}
}
