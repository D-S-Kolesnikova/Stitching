#include <ItvCvUtils/DynamicThreadPool.h>
#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/stopwatch.h>

#include <boost/test/unit_test.hpp>

#include <thread>
#include <future>
#include <array>
#include <iostream>

namespace
{

struct FixtureWithLog
{
    FixtureWithLog()
        : m_logger(ItvCv::Utils::CreateStreamLogger(std::cout, ITV8::LOG_DEBUG))
    { }

    inline ITV8::ILogger* GetLogger() const { return m_logger.get(); }
private:
    std::shared_ptr<ITV8::ILogger> m_logger;
};

} // anonymous namespace

BOOST_FIXTURE_TEST_SUITE(DynamicThreadPool, FixtureWithLog)

BOOST_AUTO_TEST_CASE(Simple)
{
    const size_t MAX_THREADS = 16;
    const size_t N_TASKS = 4 * MAX_THREADS;
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Simple", N_TASKS, MAX_THREADS / 2, MAX_THREADS);

    struct Task
    {
        std::promise<int> promise;
        std::future<int> future;
        int expect;

        Task()
            : promise()
            , future(promise.get_future())
        { }
    };

    // prepare tasks
    std::array<Task, N_TASKS> tasks;
    for (size_t i = 0; i < tasks.size(); ++i)
    {
        tasks[i].expect = i;
        auto ok = pool->Post([&tasks, i]() {
            std::this_thread::sleep_for(
                    std::chrono::milliseconds(100));
            tasks[i].promise.set_value(i);
        });

        BOOST_REQUIRE(ok);
    }

    // initiate pool shutdown, blocks until all worker threads exited
    pool->Shutdown();

    // all tasks should be complete by now
    for (auto &t: tasks)
    {
        BOOST_CHECK(t.future.get() == t.expect);
    }

    // can't add more tasks to the pool after shutdown
    BOOST_CHECK(pool->Post([](){}) == false);
    BOOST_CHECK(pool->GetThreadNum() == 0);
}

void waitForThreadCount(std::chrono::milliseconds tick, ItvCvUtils::PDynamicThreadPool pool,
    std::size_t expectedThreadCount, int maxAttemptCount = 100)
{
    for (int i = 0; i < maxAttemptCount; ++i)
    {
        std::this_thread::sleep_for(tick);
        if (pool->GetThreadNum() == expectedThreadCount)
            break;
    }
}

BOOST_AUTO_TEST_CASE(Idle)
{
    const size_t MAX_THREADS = 16;
    const size_t MAX_QUEUE = MAX_THREADS;
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Idle", MAX_QUEUE, 1, MAX_THREADS);
    const std::chrono::milliseconds IDLE_TIME(500);
    pool->SetMaxIdleThreadTime(IDLE_TIME);

    std::mutex m;
    std::condition_variable cv;
    size_t n = pool->GetMaxThreads();

    for (size_t i = 0; i < pool->GetMaxThreads(); ++i)
    {
        BOOST_REQUIRE(
            pool->Post([&](){
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::unique_lock<std::mutex> lock(m);
                if (--n == 0)
                    cv.notify_one();
            }) == true
        );
    }

    {
        ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "(test) waiting for workers to finish...");
        std::unique_lock<std::mutex> lock(m);
        while (n != 0)
            cv.wait(lock);
    }

    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "(test) waiting for thread pool to stop unneeded threads...");
    waitForThreadCount(IDLE_TIME, pool, pool->GetMinIdleThreads());
    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "(test) wait done");

    BOOST_CHECK_EQUAL(pool->GetThreadNum(), pool->GetMinIdleThreads());

    pool->Shutdown();
    BOOST_CHECK_EQUAL(pool->GetThreadNum() , 0u);
}

BOOST_AUTO_TEST_CASE(Shrink)
{
    size_t MAX_THREADS = 4;
    size_t N_TASKS = MAX_THREADS * 2;

    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Shrink", N_TASKS, 1, 4);
    const std::chrono::milliseconds IDLE_TIME(500);
    pool->SetMaxIdleThreadTime(IDLE_TIME);

    std::mutex m;
    std::condition_variable cv;
    size_t n = N_TASKS;

    for (size_t i = 0; i < N_TASKS; ++i)
    {
        BOOST_REQUIRE(
            pool->Post([&](){
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::unique_lock<std::mutex> lock(m);
                if (--n == 0)
                    cv.notify_one();
            }) == true
        );
    }

    // Now we have MaxThreads() executing and N_TASKS/2 in the queue

    std::unique_lock<std::mutex> lock(m);
    while (n >= N_TASKS - 2)
        cv.wait(lock);

    BOOST_CHECK_THROW(pool->SetMaxThreads(0), std::invalid_argument);
    //
    // Shrink the pool
    pool->SetMaxThreads(2);

    while (n > 0)
        cv.wait(lock);

    std::this_thread::sleep_for(2 * IDLE_TIME);
    BOOST_REQUIRE(pool->GetThreadNum() <= pool->GetMaxThreads());

    pool->SetMinIdleThreads(0);
    waitForThreadCount(IDLE_TIME, pool, 0u);
    BOOST_REQUIRE(pool->GetThreadNum() == 0);
    pool->Shutdown();
}

BOOST_AUTO_TEST_CASE(ZeroIdleThreads)
{
    using namespace std::chrono_literals;
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "ZeroIdleThr", 1, 0, 4);
    pool->SetMaxIdleThreadTime(100ms);

    BOOST_CHECK(pool->GetThreadNum() == 0);

    BOOST_CHECK(pool->Post([this] { ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "Hello from worker!"); }));

    std::this_thread::sleep_for(500ms);
    BOOST_CHECK(pool->GetThreadNum() == 0);
}

BOOST_AUTO_TEST_CASE(UnlimitedMaxThreads)
{
    using namespace std::chrono_literals;
    const size_t N_TASKS = 128;
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "UnlimMaxThr", N_TASKS, 0, ItvCvUtils::IDynamicThreadPool::UNLIMITED_THREADS);
    pool->SetMaxIdleThreadTime(100ms);

    BOOST_CHECK(pool->GetThreadNum() == 0);

    size_t taskCount = N_TASKS;

    std::mutex mutex;
    std::condition_variable condvar;

    for (size_t i = 0; i < N_TASKS; ++i)
        BOOST_CHECK(pool->Post([&, this] {
            ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG,  "Hello from worker!");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::unique_lock<std::mutex> lock(mutex);
            if (--taskCount == 0)
                condvar.notify_one();
        }));


    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "Waiting for tasks to complete");
    std::unique_lock<std::mutex> lock(mutex);
    condvar.wait(lock, [&]() { return taskCount == 0; });

    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "Waiting for idle threads to settle");
    ItvCvUtils::StopWatch sw;

    auto currentThreads = pool->GetThreadNum();
    while (currentThreads > 0)
    {
        std::this_thread::sleep_for(500ms);

        auto n = pool->GetThreadNum();
        if (n >= currentThreads)
        {
            ITVCV_LOG(GetLogger(), ITV8::LOG_ERROR, "thread count did not decrease in time");
            BOOST_REQUIRE_MESSAGE(n < currentThreads, "thread count did not decrease in time");
        }
        currentThreads = n;
    }

    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "idle threads took " << sw.ElapsedMs().count() << "ms to settle");

    BOOST_CHECK(pool->GetThreadNum() == 0);
}

BOOST_AUTO_TEST_CASE(Config)
{
    BOOST_CHECK_THROW(
        ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Invalid", 0, 0, 0),
        std::invalid_argument
    );

    BOOST_CHECK_THROW(
        ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Invalid", 1, 0, 0),
        std::invalid_argument
    );

    BOOST_CHECK_THROW(
        ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Invalid", 1, 2, 1),
        std::invalid_argument
    );

    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Config", 1, 1, 2);
    BOOST_CHECK_THROW(pool->SetMaxThreads(0), std::invalid_argument);
    BOOST_CHECK_THROW(pool->SetMaxQueueLength(0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(PostTask)
{
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "PostTask", 1);

    //auto f1 = pool->PostTask([] { return 1; });
    //BOOST_CHECK(f1.get() == 1);

    //auto f2 = pool->PostTask([&]() -> void {});
    //BOOST_CHECK_NO_THROW(f2.get());

    auto f3 = pool->PostTask([]() { throw std::runtime_error("Kaboom!"); });
    BOOST_CHECK_THROW(f3.get(), std::runtime_error);

    //pool->Shutdown();
    //BOOST_CHECK_THROW(pool->PostTask([](){}), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(ShutdownFromWorker)
{
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Shutdown", 1, 4, 4);
    auto future = pool->PostTask([this, &pool]{
            BOOST_CHECK(pool->IsWorkerThread());
            pool->Shutdown();
    });

    BOOST_CHECK(!pool->IsWorkerThread());
    BOOST_REQUIRE(future.wait_for(boost::chrono::seconds(1)) == boost::future_status::ready);
    pool->Shutdown().get();
}

BOOST_AUTO_TEST_CASE(DestroyFromWorker)
{
    constexpr auto THREAD_IDLE_TIME_MS = 100;
    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "Shutdown", 1, 0, 1);
    pool->SetMaxIdleThreadTime(std::chrono::milliseconds(THREAD_IDLE_TIME_MS));
    auto future = pool->PostTask([this, pool]{
            ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "Hi");
    });
    pool.reset();

    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "waiting for the last worker to stop");
    std::this_thread::sleep_for(std::chrono::milliseconds(THREAD_IDLE_TIME_MS) * 7);
    BOOST_REQUIRE(future.wait_for(boost::chrono::milliseconds(THREAD_IDLE_TIME_MS)) == boost::future_status::ready);
}

BOOST_AUTO_TEST_CASE(FailCreateThread, *boost::unit_test::disabled())
{
    // This test is flaky by nature thus it is disabled by default.
    // Creation or stopping of a random thread in any unrelated application in the OS can make this test fail.
    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "creating threads til failure...");

    std::mutex mutex;
    std::condition_variable condvar;
    bool done = false;

    std::atomic_bool called_flag{ false };

    std::vector<std::thread> threads;
    threads.reserve(10000);

    try
    {
        for (;;)
        {
            threads.emplace_back(
                std::thread([&]()
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        condvar.wait(lock, [&]() { return done; });
                    }
            ));
        }
    }
    catch (std::exception const& e)
    {
        ITVCV_LOG(GetLogger(), ITV8::LOG_WARNING, "thread creation failed: " << e.what());
    }

    ITVCV_LOG(GetLogger(), ITV8::LOG_DEBUG, "created " << threads.size() << " threads");

    auto pool = ItvCvUtils::CreateDynamicThreadPool(GetLogger(), "ThreadFail", 1, 0, 1);

    // pool should not be able to create a worker thread, so Post() will fail.
    // Task will never be executed
    BOOST_TEST(pool->Post([&]() { called_flag = true; }) == false);

    {
        std::unique_lock<std::mutex> lock(mutex);
        done = true;
        condvar.notify_all();
    }

    for (auto& t : threads)
        t.join();

    // pool will be able to create a worker thread after all the dummy threads have been released
    // and the task will be executed
    std::atomic_bool called_flag2{ false };
    BOOST_TEST(pool->Post([&]() { called_flag2 = true; }));

    pool->Shutdown().get();
    BOOST_TEST(called_flag == false);
    BOOST_TEST(called_flag2 == true);
}

BOOST_AUTO_TEST_SUITE_END()