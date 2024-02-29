#pragma once

namespace ItvCvUtils
{

template <class Fn>
typename std::enable_if<
            !std::is_void<typename std::result_of<Fn()>::type>::value,
            boost::future<typename std::result_of<Fn()>::type>
>::type
IDynamicThreadPool::PostTask(Fn&& fn)
{
    using Ret = typename std::result_of<Fn()>::type;
    auto promise = std::make_shared<boost::promise<Ret>>();
    auto future = promise->get_future();

    auto lambda = [promise, fn = std::move(fn)]() mutable {
        try
        {
            promise->set_value(fn());
        }
        catch(...)
        {
            promise->set_exception(boost::current_exception());
        }
    };

    if (!this->Post(lambda))
        throw XDynamicThreadPoolError("unable to post a task into the pool");

    return future;
}

template <class Fn>
typename std::enable_if<
            std::is_void<typename std::result_of<Fn()>::type>::value,
            boost::future<void>
>::type
IDynamicThreadPool::PostTask(Fn&& fn)
{
    using Ret = typename std::result_of<Fn()>::type;
    auto promise = std::make_shared<boost::promise<Ret>>();
    auto future = promise->get_future();

    auto lambda = [promise, fn = std::move(fn)]() {
        try
        {
            fn();
            promise->set_value();
        }
        catch(...)
        {
            promise->set_exception(boost::current_exception());
        }
    };

    if (!this->Post(lambda))
        throw XDynamicThreadPoolError("unable to post a task into the pool");

    return future;
}

} // namespace ItvCvUtils
