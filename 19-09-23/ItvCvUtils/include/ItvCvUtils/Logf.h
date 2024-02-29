#pragma once

#include "Log.h"

#include <fmt/format.h>

/// Logs a message at specified level to logger. This macro checks for
/// validity of passed logger pointer and level.
#define ITVCV_LOGF(logger, level, formatString, ...)          \
    do                                                        \
    {                                                         \
        ITV8::ILogger* const mylogger = logger;               \
        if (!mylogger || mylogger->GetLogLevel() > level)     \
            break;                                            \
        mylogger->Log(level, ITV8_LINEINFO, fmt::format(FMT_STRING(formatString), __VA_ARGS__).c_str()); \
    } while (false)

/// Same as ITVCV_LOG with the difference that it prepends "this=<this>" to the log message
#define ITVCV_THIS_LOGF(logger, level, formatString, ...) \
    ITVCV_LOGF(logger, level, "this={} " formatString, static_cast<void*>(this), __VA_ARGS__)

