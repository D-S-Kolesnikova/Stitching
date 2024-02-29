#include <ItvCvUtils/DynamicThreadPool.h>
#include <ItvCvUtils/stopwatch.h>
#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/ThreadName.h>

#include <cassert>
#include <mutex>
#include <deque>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <iomanip>
#include <string>
#include <condition_variable>
#include <boost/config.hpp>

namespace
{

std::string maxThreadsToString(size_t maxThreads)
{
    if (maxThreads == ItvCvUtils::IDynamicThreadPool::UNLIMITED_THREADS)
        return "unlimited";

    return std::to_string(maxThreads);
}

std::string maxQueueLengthToString(size_t maxQueueLength)
{
    if (maxQueueLength == ItvCvUtils::IDynamicThreadPool::UNLIMITED_QUEUE_LENGTH)
        return "unlimited";

    return std::to_string(maxQueueLength);
}

class DynamicThreadPoolImpl:
    public ItvCvUtils::IDynamicThreadPool,
    public std::enable_shared_from_this<DynamicThreadPoolImpl>
{
    ITV8::ILogger *m_logger;

    mutable std::mutex m_mutex;
    std::condition_variable m_queueSignal;

    std::deque<Task_t> m_workQueue;

    bool m_shutdown;
    boost::promise<void> m_shutdownPromise;
    boost::shared_future<void> m_shutdownFuture;
    std::condition_variable m_workerFinishedSignal;

    struct Config
    {
        std::string tag;
        size_t maxThreads;
        size_t minIdleThreads;
        size_t maxQueueLength;
        std::chrono::milliseconds maxIdleTime;

        static Config Default()
        {
            Config c;
            c.maxThreads = std::thread::hardware_concurrency();
            c.minIdleThreads = 1;
            c.maxQueueLength = 1;
            c.maxIdleTime = std::chrono::milliseconds(15000);
            return c;
        }

    } m_config;

    // running statistics
    size_t m_currentThreads;
    size_t m_maxObservedPoolSize;
    size_t m_idleThreads;
    size_t m_maxObservedQueueLength;

    using WorkerId_t = size_t;
    std::atomic<WorkerId_t> m_nextWorkerId;

    std::map<size_t /* worker_id */, std::thread> m_workers;
    std::vector<std::thread> m_finishedWorkers;

public:
    DynamicThreadPoolImpl(ITV8::ILogger *logger, std::string const& tag, size_t maxQueueLength, size_t minIdleThreads, size_t maxThreads)
        : m_logger(logger)
        , m_shutdown(false)
        , m_shutdownFuture(m_shutdownPromise.get_future())
        , m_currentThreads(0)
        , m_maxObservedPoolSize(0)
        , m_idleThreads(0)
        , m_maxObservedQueueLength(0)
        , m_nextWorkerId(0)
    {
        if (!(maxThreads > 0) || minIdleThreads > maxThreads || !(maxQueueLength > 0))
        {
            ITVCV_LOG(
                m_logger,
                ITV8::LOG_ERROR,
                "invalid initial configuration: " << maxThreads << " minIdleThreads:"
                << minIdleThreads << " maxQueueLength:" << maxQueueLength);
            throw std::invalid_argument("invalid initial configuration");
        }

        m_config = Config::Default();
        m_config.tag = tag;
        m_config.maxQueueLength = maxQueueLength;
        m_config.maxThreads = maxThreads;
        m_config.minIdleThreads = minIdleThreads;

        LogConfig(m_config);
    }

    ~DynamicThreadPoolImpl()
    {
        if (isWorkerThreadLocked())
        {
            // special case, destructor is called from the last worker thread,
            // after it has already moved itself to m_finishedWorkers
            for (auto& w: m_finishedWorkers)
            {
                if (w.get_id() == std::this_thread::get_id())
                   w.detach();
                else if (w.joinable())
                    w.join();
            }
        }
        else
        {
            // Normal shutdown from a non-worker thread
            Shutdown();
        }

        ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "bye-bye");
    }

    /*
     * Configuration
     */
    void SetMaxThreads(size_t maxThreads) override
    {
        if (maxThreads == 0)
            throw std::invalid_argument("maxThreads should be > 0");

        std::unique_lock<std::mutex> lock(m_mutex);
        m_config.maxThreads = maxThreads;
        m_config.minIdleThreads = std::min(m_config.maxThreads, m_config.minIdleThreads);

        LogConfig(m_config);
        m_queueSignal.notify_all();
    }

    size_t GetMaxThreads() const override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_config.maxThreads;
    }

    size_t GetThreadNum() const override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_currentThreads;
    }

    void SetMinIdleThreads(size_t minIdleThreads) override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_config.minIdleThreads = minIdleThreads;

        m_config.maxThreads = std::max(m_config.maxThreads, m_config.minIdleThreads);

        LogConfig(m_config);
        m_queueSignal.notify_all();
    }

    size_t GetMinIdleThreads() const override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_config.minIdleThreads;
    }

    void SetMaxQueueLength(size_t maxQueueLength) override
    {
        if (maxQueueLength == 0)
            throw std::invalid_argument("maxQueueLength should be > 0");

        std::unique_lock<std::mutex> lock(m_mutex);
        m_config.maxQueueLength = maxQueueLength;
        LogConfig(m_config);
    }

    size_t GetMaxQueueLength() const override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_config.maxQueueLength;
    }

    void SetMaxIdleThreadTime(std::chrono::milliseconds duration) override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_config.maxIdleTime = duration;
        LogConfig(m_config);
        m_queueSignal.notify_all();
    }

    virtual bool IsWorkerThread() const override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return isWorkerThreadLocked();
    }

    virtual boost::shared_future<void> Shutdown() override
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_shutdown)
            return m_shutdownFuture;

        m_shutdown = true;
        ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "shutting down");

        if (isWorkerThreadLocked())
        {
            ITVCV_LOG(
                m_logger,
                ITV8::LOG_DEBUG,
                "shutdown is called from within a worker thread, spawning a dedicated shutdown thread");

            auto pself = shared_from_this();
            std::thread t([this, pself]() {
                std::unique_lock<std::mutex> lock(m_mutex);
                pself->doShutdownLocked(lock);
            });

            t.detach();
        }
        else
        {
            // called from a non-worker thread, do all the work here
            doShutdownLocked(lock);
        }

        return m_shutdownFuture;
    }

    void doShutdownLocked(std::unique_lock<std::mutex>& lock)
    {
        m_queueSignal.notify_all();

        while (m_workers.size() > 0)
            m_workerFinishedSignal.wait(lock);

        for (auto& w: m_finishedWorkers)
        {
            if (w.joinable()) w.join();
        }

        ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "shutdown complete");
        m_shutdownPromise.set_value();
    }

    bool Post(Task_t const& task) noexcept override
    {
        return Post(Task_t(task));
    }

    bool Post(Task_t&& task) noexcept override
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_shutdown)
        {
            ITVCV_LOG(m_logger, ITV8::LOG_WARNING, "Post() called during or after Shutdown()");
            return false;
        }

        auto queueLength = m_workQueue.size();
        if (queueLength >= m_config.maxQueueLength)
        {
            ITVCV_LOG(m_logger, ITV8::LOG_WARNING, "work queue is full, incoming task discarded");
            return false;
        }

        m_workQueue.emplace_back(std::move(task));
        m_maxObservedQueueLength = std::max(queueLength + 1, m_maxObservedQueueLength);

        if (m_idleThreads == 0 && m_currentThreads < m_config.maxThreads)
        {
            if (AddThreadLocked())
            {
                // don't need to notify because new thread will pick up new task
                return true;
            }
            else if (m_currentThreads == 0)
            {
                // could not start a new worker thread and there are no active workers
                // to handle the task later

                constexpr const char* logstring =
                    "incoming task discarded: unable to create a worker thread"
#if defined(__linux__)
                    ". Check the values of `sysctl kernel.pid_max` and `ulimit --process-count`"
#endif
                    ;
                ITVCV_LOG(m_logger, ITV8::LOG_WARNING, logstring);

                m_workQueue.pop_back();
                return false;
            }

            // there are other active threads that can handle the task later
        }

        if (m_workQueue.size() > 0)
        {
            m_queueSignal.notify_one();
        }

        return true;
    }

private:
    bool isWorkerThreadLocked() const
    {
        auto this_id = std::this_thread::get_id();

        for (auto const& kv: m_workers)
        {
            if (kv.second.get_id() == this_id)
                return true;
        }

        for (auto const& w: m_finishedWorkers)
        {
            if (w.get_id() == this_id)
                return true;
        }

        return false;
    }

    bool AddThreadLocked() noexcept
    {
        try
        {
            Config configCopy = m_config;
            auto workerId = m_nextWorkerId++;

            auto pself = shared_from_this();
            std::thread t([this, pself, workerId, configCopy]() {
                // save a copy of current config on stack, it is regularly updated
                // by WorkerFunc

                Config c = configCopy;
                ItvCvUtils::SetThreadName(c.tag + std::to_string(workerId));

                ITVCV_LOG(m_logger, ITV8::LOG_INFO, "worker started");

                {
                    ItvCvUtils::StopWatch stopWatch;

                    auto stat = WorkerFunc(m_logger, workerId, c);

                    using namespace std::chrono;
                    double load = 0.;
                    auto elapsed_us = stopWatch.Elapsed<std::chrono::microseconds>();
                    if (elapsed_us.count() > 0)
                        load = 100. * duration_cast<microseconds>(stat.busyTime).count() / elapsed_us.count();

                    ITVCV_LOG(
                        m_logger,
                        ITV8::LOG_DEBUG,
                        "stats: "
                          << "tasks executed: " << stat.tasksExecuted
                          << std::setprecision(3)
                          << ", busy: " << duration_cast<milliseconds>(stat.busyTime).count() / 1000. << "s"
                          << ", utilisation: " << load << "%");
                    ;
                }

                std::unique_lock<std::mutex> lock(m_mutex);

                ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "worker finished");
                LogStatsLocked();

                auto it = m_workers.find(workerId);
                if (it == m_workers.end())
                    std::abort();

                m_finishedWorkers.emplace_back(std::move(it->second));
                m_workers.erase(it);

                m_workerFinishedSignal.notify_all();
            });

            ++m_currentThreads;
            m_maxObservedPoolSize = std::max(m_maxObservedPoolSize, m_currentThreads);

            m_workers.emplace(workerId, std::move(t));

            ITVCV_LOG(m_logger, ITV8::LOG_DEBUG, "added worker thread");
            LogStatsLocked();

            return true;
        }
        catch(std::exception const& e)
        {
            ITVCV_LOG(m_logger, ITV8::LOG_ERROR, "unable to create worker thread: " << e.what());
        }

        return false;
    }

    struct IdleThreadToken
    {
        size_t& idleThreadsCounter;
        inline explicit IdleThreadToken(size_t& _idleThreadsCounter)
            : idleThreadsCounter(_idleThreadsCounter)
        {
            ++idleThreadsCounter;
        }

        inline ~IdleThreadToken()
        {
            --idleThreadsCounter;
        }
    };

    struct WorkerStat
    {
        size_t tasksExecuted;
        std::chrono::microseconds busyTime;
    };

    WorkerStat WorkerFunc(ITV8::ILogger* logger, WorkerId_t workerId, Config& config) BOOST_NOEXCEPT
    {
        const std::string logPrefix("/" + std::to_string(workerId) + " ");
        WorkerStat stat{0, std::chrono::microseconds(0)};
        auto lastTaskFinishedAt = std::chrono::steady_clock::now();

        for (;;)
        {
            std::unique_lock<std::mutex> lock(m_mutex);

            // Update our saved copy of the configuration. See WorkerFunc() caller
            config = m_config;

            if (m_currentThreads > m_config.maxThreads)
            {
                ITVCV_LOG(logger, ITV8::LOG_DEBUG, logPrefix << "stopping extra worker");
                --m_currentThreads;
                return stat;
            }

            if (!m_shutdown && m_workQueue.empty())
            {
                IdleThreadToken idleThread(m_idleThreads);

                auto deadline = lastTaskFinishedAt + m_config.maxIdleTime;
                auto rc = m_queueSignal.wait_until(lock, deadline);
                if (rc == std::cv_status::timeout && m_workQueue.empty())
                {
                    if (m_idleThreads > m_config.minIdleThreads)
                    {
                        ITVCV_LOG(logger, ITV8::LOG_DEBUG, logPrefix << "stopping idle worker");
                        --m_currentThreads;
                        return stat;
                    }
                    else {
                        // Task didn't arrive in time but we need to keep this
                        // worker alive. Set lastTaskFinishedAt to now() to wake
                        // up in the next idle test cycle.
                        lastTaskFinishedAt = std::chrono::steady_clock::now();
                    }
                }
            }

            if (m_shutdown && m_workQueue.empty())
            {
                ITVCV_LOG(logger, ITV8::LOG_DEBUG, logPrefix << "stopping worker due to pool shutdown");
                --m_currentThreads;
                return stat;
            }

            if (m_workQueue.empty())
                continue;

            decltype(m_finishedWorkers) finishedWorkers;
            m_finishedWorkers.swap(finishedWorkers);

            auto task = std::move(m_workQueue.front());
            m_workQueue.pop_front();

            //
            // Unlocked from here
            //
            lock.unlock();

            ++stat.tasksExecuted;
            ItvCvUtils::StopWatch stopWatch;
            try
            {
                task();
            }
            catch(std::exception const& e)
            {
                ITVCV_LOG(logger, ITV8::LOG_ERROR, logPrefix << "FATAL: WorkerFunc: task threw an unexpected exception: " << e.what());
                std::terminate();
            }
            catch(...)
            {
                ITVCV_LOG(logger, ITV8::LOG_ERROR, logPrefix << "FATAL: WorkerFunc: task threw an unexpected unknown exception");
                std::terminate();
            }

            for (auto& w: finishedWorkers)
            {
                if (w.joinable()) w.join();
            }

            stat.busyTime += stopWatch.Elapsed<std::chrono::microseconds>();
        }
    }

    void LogStatsLocked() BOOST_NOEXCEPT
    {
        ITVCV_LOG(
            m_logger,
            ITV8::LOG_DEBUG,
            "stats: "
            << "queue: " << m_workQueue.size() << "/" << maxQueueLengthToString(m_config.maxQueueLength)
                << " peak=" << m_maxObservedQueueLength
            << ", threads: " << m_currentThreads << "/" << maxThreadsToString(m_config.maxThreads)
                << " peak=" << m_maxObservedPoolSize
            << ", idle threads: " << m_idleThreads << "/min=" << m_config.minIdleThreads);
        ;
    }

    void LogConfig(Config const& c) BOOST_NOEXCEPT
    {
        ITVCV_LOG(
            m_logger,
            ITV8::LOG_DEBUG,
            "config: "
            << " max threads: " << maxThreadsToString(c.maxThreads)
            << ", min idle threads: " << c.minIdleThreads
            << ", max idle time (ms): " << c.maxIdleTime.count()
            << ", max queue length: " << maxQueueLengthToString(c.maxQueueLength));
        ;
    }
};

} // anonymous namespace

namespace ItvCvUtils
{
ITVCV_UTILS_API PDynamicThreadPool CreateDynamicThreadPool(
    ITV8::ILogger* logger,
    std::string const& tag,
    size_t maxQueueLength,
    size_t minIdleThreads,
    size_t maxThreads)
{
    return std::make_shared<DynamicThreadPoolImpl>(logger, tag, maxQueueLength, minIdleThreads, maxThreads);
}

} // namespace ItvCvUtils
