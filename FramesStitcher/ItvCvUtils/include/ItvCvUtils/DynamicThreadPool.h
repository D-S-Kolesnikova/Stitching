#pragma once

/// Copied and adapted from NGP

#include <ItvCvUtils/ItvCvUtils.h>

#include <ItvSdk/include/IErrorService.h>

#define BOOST_THREAD_PROVIDES_FUTURE

#include <boost/thread/future.hpp>

#include <cstddef>
#include <chrono>
#include <functional>
#include <thread>
#include <memory>

#if defined(__GNUC__) && (__GNUC__ >= 4)
#define CHECK_RESULT __attribute__ ((warn_unused_result))
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
#define CHECK_RESULT _Check_return_
#else
#define CHECK_RESULT
#endif


namespace ItvCvUtils
{
class XDynamicThreadPoolError: public std::runtime_error
{
public:
    typedef std::runtime_error Base;

    explicit XDynamicThreadPoolError(const std::string& message): Base(message.c_str()) {}

    explicit XDynamicThreadPoolError(const char* message): Base(message) {}
};

class IDynamicThreadPool
{
public:
    virtual ~IDynamicThreadPool() {}

    // Request a thread pool shutdown
    //
    // Result future becomes ready when all queued work is complete and all worker
    // threads have exited. Thread pool will NOT accept new tasks during and after
    // Shutdown().
    //
    // NOTE: wait()-ing on result future within a worker thread will result in a
    // dead lock, see IsWorkerThread().
    //
    // THREAD-SAFE
    virtual boost::shared_future<void> Shutdown() = 0;

    // Maxumum allowed number of threads
     //
     // DynamicThreadPool will attempt to spawn up to MaxThreads workers to
     // handle incoming tasks. In practice, the actual number of threads will be
     // limited by the OS configuration and resources.
     //
     // If newMaxThreads > currentMaxThreads excess threads will be released.
     // assert(maxThreads > 0 && maxThreads >= minIdleThreads)
    static constexpr auto UNLIMITED_THREADS = static_cast<size_t>(-1);
    virtual void SetMaxThreads(size_t maxThreads) = 0;
    virtual size_t GetMaxThreads() const = 0;

    // Current number of worker threads. For testing only
    virtual size_t GetThreadNum() const = 0;

    // Check if current thread is a worker thread
    virtual bool IsWorkerThread() const = 0;

    // Minimum number of idle threads
    // Thread pool will always keep minIdleThreads alive, even when it has nothing
    // to do.
    // Set to 0 for no idle threads
    // If minIdleThreads > maxThreads, maxThreads set to minIdleThreads
    virtual void SetMinIdleThreads(size_t minIdleThreads) = 0;
    virtual size_t GetMinIdleThreads() const = 0;

    // Maximum task queue length
    // Task queue is used to keep tasks before they are picked up by worker
    // threads.
    // assert (maxQueueLength > 0)
    //
    // Use UNLIMITED_QUEUE_LENGTH in *TESTS ONLY*. Production code should always
    // put a sane upper bound on the task queue. If you think that your case is
    // special, think again.
    static constexpr auto UNLIMITED_QUEUE_LENGTH = static_cast<size_t>(-1);
    virtual void SetMaxQueueLength(size_t maxQueueLength) = 0;
    virtual size_t GetMaxQueueLength() const = 0;

    // Maximum time a worker thread allowed to sleep waiting for work (default 15s).
    // If worker does not receive any tasks within maxIdleTime milliseconds it
    // is attempted to be released, subject to minIdleThreads constraint.
    virtual void SetMaxIdleThreadTime(std::chrono::milliseconds) = 0;

    template<class Rep, class Period>
    void SetMaxIdleThreadTime(std::chrono::duration<Rep, Period> duration)
    {
        SetMaxIdleThreadTime(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }

    // Add a task to the pool
    //
    // Post() adds the task to the work queue to be executed some time later
    // from an unspecified worker thread.
    //
    // Post() can reject the task for the following reasons:
    //
    //  * Shutdown() has been called, tasks are no longer accepted;
    //  * configured limit reached, see SetMaxQueueLength() and
    //      SetMaxMeanQueueingTime();
    //  * worker thread creation failed. That typically means that OS ran out of
    //    resources or reached a configured limit.
    //
    // Return
    //  - true when succeeded
    typedef std::function<void()> Task_t;
    CHECK_RESULT virtual bool Post(Task_t const& task) noexcept = 0;
    CHECK_RESULT virtual bool Post(Task_t&& task) noexcept = 0;
    // Add task to the pool
    // Return boost::future<...>
    // Throw std::runtime_error if Post(...) has failed
    template<class Fn>
    typename std::enable_if<
        !std::is_void<typename std::result_of<Fn()>::type>::value,
        boost::future<typename std::result_of<Fn()>::type>>::type
    PostTask(Fn&& fn);

    template<class Fn>
    typename std::enable_if<std::is_void<typename std::result_of<Fn()>::type>::value, boost::future<void>>::type PostTask(
        Fn&& fn);
};

using PDynamicThreadPool = std::shared_ptr<IDynamicThreadPool>;

//
// Create a dynamic thread pool
//
// Args:
//  logger
//  tag            - up 12 symbols including '\0'. Tag is used in logging and to
//                   prefix worker thread names that belong to this pool.
//
//  maxQueueLength - maximum number of tasks in the queue. *ALWAYS* put a sane upper bound
//                   on the queue length to avoid infinite queue under heavy
//                   load. If you think your case is special, think again.
//                   Expressing upper maxQueueLength via hardware_concurrency()
//                   is a good start, see examples in the code.
//
//  minIdleThreads - minimum number of threads that will be kept alive to
//                   minimise the reaction time. Use 0 if you don't expect a
//                   constant and steady influx of tasks.
//
//  maxThreads     - maximum number of threads the pool can spawn. To avoid
//                   performance degradation use values:
//
//                   <= hardware_concurrency()      - for CPU bound tasks
//                   other                          - for I/O bound or tasks with
//                                                    long sleep/wait
//
// On error throws std::exception
//
ITVCV_UTILS_API PDynamicThreadPool CreateDynamicThreadPool(
    ITV8::ILogger* logger,
    std::string const& tag,
    size_t maxQueueLength,
    size_t minIdleThreads = 0,
    size_t maxThreads = std::thread::hardware_concurrency());

} // namespace ItvCvUtils

#include "DynamicThreadPoolImpl.h"
