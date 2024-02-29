#pragma once

#if defined _WIN32 || defined __CYGWIN__
#    define ITVCV_API_IMPORT __declspec(dllimport)
#    define ITVCV_API_EXPORT __declspec(dllexport)
#elif __GNUC__ >= 4
#    define ITVCV_API_IMPORT __attribute__((visibility("default")))
#    define ITVCV_API_EXPORT __attribute__((visibility("default")))
#else
    #define ITVCV_API_IMPORT
    #define ITVCV_API_EXPORT
#endif

#if defined(__cplusplus) && __cplusplus > 201402L
#define ITVCV_DEPRECATED [[deprecated]]
#define ITVCV_DEPRECATED_MSG(msg) [[deprecated(msg)]]
#else
#define ITVCV_DEPRECATED
#define ITVCV_DEPRECATED_MSG(msg)
#endif

