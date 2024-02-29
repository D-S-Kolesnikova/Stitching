#pragma once

#include <boost/asio.hpp>
#define BOOST_COROUTINES_NO_DEPRECATION_WARNING
#include <boost/asio/spawn.hpp>
#include <boost/scope_exit.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/system/system_error.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem/detail/utf8_codecvt_facet.hpp>

#include <fmt/format.h>

#include <string>
#include <mutex>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <functional>

namespace ItvCvUtils {
namespace SimpleRPC {

namespace error {
enum rpc_errors {
    marshalling_failed = 1,
    unsupported_call_id,
    server_busy,
    server_down,
    internal_error,
    user_defined_error
};

class Category : public boost::system::error_category
{
public:
    const char *name() const noexcept { return "ItvCvUtils::SimpleRPC error"; }
    std::string message(int ev) const
    {
        switch (static_cast<error::rpc_errors>(ev))
        {
        case marshalling_failed:
            return "Failed to parse RPC data";
        case unsupported_call_id:
            return "Unsupported RPC id";
        case internal_error:
            return "Internal server error";
        case server_busy:
            return "Server is busy, re-try later";
        case server_down:
            return "Server is down, re-try later";
        case user_defined_error:
            return "User defined error";
        }
        return "Unexpected error";
    }
};
inline boost::system::error_category const& GetSimpleRpcErrorCategory()
{
    static Category c;
    return c;
}
} // namespace error
} // namespace SimpleRPC
} // namespace ItvCvUtils

namespace boost {
namespace system {
template <> struct is_error_code_enum<ItvCvUtils::SimpleRPC::error::rpc_errors> {
    static const bool value = true;
};
} // namespace system
} // namespace boost

namespace ItvCvUtils {
namespace SimpleRPC {
namespace error {
inline boost::system::error_code make_error_code(rpc_errors e)
{
    return boost::system::error_code(static_cast<int>(e), GetSimpleRpcErrorCategory());
}
} // namespace error

struct Response
{
    int ec = 0;
    std::string msg;

    Response() = default;
    Response(boost::system::error_code const& e)
        : ec(e.value())
        , msg(fmt::format(FMT_STRING("{}[{}]: {}"), e.category().name(), e.value(), e.message()))
    {}
    Response(boost::system::system_error const& e)
        : ec(e.code().value())
        , msg(e.what())
    {}
};

template <std::uint32_t call_id, typename Ret, typename Arg> struct RpcDesc
{
    static constexpr std::uint32_t id = call_id;
    using response_type = Ret;
    using request_type = Arg;
};

struct IRpcResponseHandler
{
    virtual ~IRpcResponseHandler() {}
    virtual void process(Response const& response) = 0;
};

namespace detail {
struct Header
{
    std::uint32_t id;
    std::uint32_t size;
    std::uintptr_t tag;
};
struct VoidTag
{
    friend std::ostream& operator << (std::ostream& s, VoidTag const&) { return s; }
    friend std::istream& operator >> (std::istream& s, VoidTag&) { return s; }

    template <typename Archive>
    void serialize(Archive& , unsigned) {}
};

template <typename BaseStream>
class BasicUserDataStream : public BaseStream
{
public:
    template <typename ...DeviceArgs>
    BasicUserDataStream(DeviceArgs&&... args)
        : BaseStream(std::forward<DeviceArgs>(args)...)
    {
        std::locale global;
        this->imbue(std::locale(global, new boost::filesystem::detail::utf8_codecvt_facet));
    }
};
template <typename SourceOrSink>
using UserDataStream = BasicUserDataStream< boost::iostreams::stream< SourceOrSink > >;

template <typename RpcServer, std::uint32_t call_id>
void invoke_rpc_handler(RpcServer& server, void (RpcServer::*handler)(typename RpcServer::rpc_context_type const&), RpcDesc<call_id, void, void>, std::istream&, std::ostream&, typename RpcServer::rpc_context_type const& context )
{
    (server.*handler)(context);
}
template <typename RpcServer, std::uint32_t call_id, typename Ret>
void invoke_rpc_handler(RpcServer& server, Ret (RpcServer::*handler)(typename RpcServer::rpc_context_type const&), RpcDesc<call_id, Ret, void>, std::istream&, std::ostream& os, typename RpcServer::rpc_context_type const& context )
{
    boost::archive::binary_oarchive ar(os);
    ar << (server.*handler)(context);
}
template <typename RpcServer, std::uint32_t call_id, typename ArgHandler, typename ArgRpc, typename std::enable_if_t<std::is_convertible<ArgRpc&&, ArgHandler>::value, int> = 0>
void invoke_rpc_handler(RpcServer& server, void (RpcServer::*handler)(ArgHandler, typename RpcServer::rpc_context_type const&), RpcDesc<call_id, void, ArgRpc>, std::istream& is, std::ostream&, typename RpcServer::rpc_context_type const& context )
{
    ArgRpc request{};
    boost::archive::binary_iarchive iar(is);
    iar >> request;
    (server.*handler)(std::move(request), context);
}
template <typename RpcServer, std::uint32_t call_id, typename Ret, typename ArgHandler, typename ArgRpc, typename std::enable_if_t<std::is_convertible<ArgRpc&&, ArgHandler>::value, int> = 0>
void invoke_rpc_handler(RpcServer& server, Ret (RpcServer::*handler)(ArgHandler, typename RpcServer::rpc_context_type const&), RpcDesc<call_id, Ret, ArgRpc>, std::istream& is, std::ostream& os, typename RpcServer::rpc_context_type const& context)
{
    ArgRpc request{};
    boost::archive::binary_iarchive iar(is);
    iar >> request;
    boost::archive::binary_oarchive oar(os);
    oar << (server.*handler)(std::move(request), context);
}

template<typename, typename T>
struct has_set_exception {
    static_assert(
        std::integral_constant<T, false>::value,
        "Second template parameter needs to be of function type.");
};

// specialization that does the checking

template<typename C, typename Ret, typename... Args>
struct has_set_exception<C, Ret(Args...)> {
private:
    template<typename T>
    static constexpr auto check(T*)
    -> typename
        std::is_same<
            decltype( std::declval<T>().set_exception( std::declval<Args>()... ) ),
            Ret    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        >::type;  // attempt to call it and see if the return type is correct

    template<typename>
    static constexpr std::false_type check(...);

public:
    typedef decltype(check<C>(0)) type;
    static constexpr bool value = type::value;
};

template <typename Ret, template <typename T> class Promise = std::promise>
class PromiseHandler : public IRpcResponseHandler
{
    Promise<Ret> m_promise;

    template <typename U = Ret, typename std::enable_if_t<std::is_same<void, U>::value, int> = 0>
    void extract_value(std::string const& )
    {
        m_promise.set_value();
    }

    template <typename U = Ret, typename std::enable_if_t<!std::is_same<void, U>::value, int> = 0>
    void extract_value(std::string const& msg)
    {
        UserDataStream< boost::iostreams::array_source > is(msg.data(), msg.size());
        Ret val;
        boost::archive::binary_iarchive ar(is);
        ar >> val;
        m_promise.set_value( std::move(val) );
    }

    template <typename P = Promise<Ret>>
    void set_exception(std::string const& msg, std::true_type /*accepts_boost_exception_ptr*/)
    {
        m_promise.set_exception(boost::copy_exception(std::runtime_error(msg)));
    }

    template <typename P = Promise<Ret>>
    void set_exception(std::string const& msg, std::false_type /*accepts_boost_exception_ptr*/)
    {
        m_promise.set_exception(std::make_exception_ptr(std::runtime_error(msg)));
    }

public:
    PromiseHandler() = default;

    auto get_future() -> decltype(m_promise.get_future()) { return m_promise.get_future(); }
    void process(Response const& response) override
    {
        if (response.ec)
        {
            set_exception(response.msg, typename has_set_exception<Promise<Ret>, void(boost::exception_ptr)>::type());
            return;
        }
        extract_value(response.msg);
    }
};

template <typename Ret>
class CallbackHandler : public PromiseHandler<Ret, std::promise>
{
    std::function<void (std::future<Ret>&&)> m_then;

public:
    CallbackHandler(std::function<void(std::future<Ret>&& ready)> then) : m_then(std::move(then)) {}

    void process(Response const& response) override
    {
        auto future = this->get_future();
        PromiseHandler<Ret, std::promise>::process(response);
        m_then(std::move(future));
    }
};
} // namespace detail

#define RPC_DESCRIBE(call_id, response_type, call_name, request_type) \
    using rpc_##call_id##_type = ::ItvCvUtils::SimpleRPC::RpcDesc<call_id, response_type, request_type>; \
    template <typename RpcServer> \
    void process_rpc(RpcServer& server, rpc_##call_id##_type rpc, std::istream& is, std::ostream& os, typename RpcServer::rpc_context_type const& context ) \
    { \
        ItvCvUtils::SimpleRPC::detail::invoke_rpc_handler(server, &RpcServer::process_##call_name, rpc, is, os, context); \
    } \
    constexpr rpc_##call_id##_type rpc_##call_name

#define RPC_DISPATCH_TABLE() \
    ItvCvUtils::SimpleRPC::Response dispatch(std::uint32_t call_id, std::istream& is, rpc_context_type const& context) \
    { \
        switch ( call_id ) \
        {

#define RPC_DISPATCH_TABLE_ENTRY(call_name) \
        case rpc_##call_name.id: \
        { \
            ItvCvUtils::SimpleRPC::Response response; \
            { \
            ItvCvUtils::SimpleRPC::detail::UserDataStream<boost::iostreams::back_insert_device<std::string>> os(response.msg); \
            process_rpc(*this, rpc_##call_name, is, os, context); \
            } \
            return response; \
        } \
        break

#define RPC_DISPATCH_TABLE_END() \
        } \
        return ItvCvUtils::SimpleRPC::Response { make_error_code(ItvCvUtils::SimpleRPC::error::unsupported_call_id) }; \
    }


template <typename Derived>
class Server
{
protected:
    Server() = default;

public:
    struct RpcContext
    {
        boost::asio::yield_context yield;
    };
    using rpc_context_type = RpcContext;

    template <typename SessionOfDerived>
    rpc_context_type make_rpc_context(std::shared_ptr<SessionOfDerived> const& , boost::asio::yield_context const& yield)
    {
        return RpcContext{ yield };
    }

    template <typename SessionOfDerived>
    void handle_session(std::shared_ptr<SessionOfDerived> session, boost::asio::yield_context const& yield)
    {
        auto& socket = session->socket();
        while (true)
        {
            detail::Header rpc;
            async_read( socket, boost::asio::buffer(&rpc, sizeof(rpc)), yield );
            boost::asio::streambuf buf;
            if (rpc.size)
            {
                auto n = async_read( socket, buf.prepare(rpc.size), yield );
                buf.commit(n);
            }
            Response response;
            if (buf.size() != rpc.size)
            {
                response = Response{ make_error_code(error::marshalling_failed) };
            }
            else
            {
                try
                {
                    detail::BasicUserDataStream<std::istream> ss{&buf};
                    auto d = static_cast<Derived*>(this);
                    response = d->dispatch(rpc.id, ss, d->make_rpc_context(session, yield));
                }
                catch (boost::system::system_error const& e)
                {
                    response = Response{ e };
                }
                catch (std::exception const& e)
                {
                    response.ec = error::internal_error;
                    response.msg = e.what();
                }
            }
            rpc.id = response.ec;
            rpc.size = response.msg.size();
            std::initializer_list<boost::asio::const_buffer> bufs{ {&rpc, sizeof(rpc)}, {response.msg.data(), rpc.size} };
            async_write( socket, bufs, yield );
        }
    }
};

template <typename Derived>
class Client
{
protected:
    Client()
    {
        using dispatch_result_type = decltype(static_cast<Derived*>(this)->dispatch(static_cast<void*>(nullptr), std::declval<Response>()));
        static_assert(std::is_same<void, dispatch_result_type>::value, "Method is required: void Derived::dispatch(void*, Response const&)");
        static_assert(std::is_reference<decltype(static_cast<Derived*>(this)->socket())>::value, "Method is required: Socket& Derived::socket()");
    }

    template <typename Param>
    void async_call_raw(std::uint32_t call_id, void* tag, Param&& param)
    {
        auto buf = std::make_shared<boost::asio::streambuf>();
        auto data = buf->prepare(1024);
        buf->commit(sizeof(detail::Header));
        {
            detail::BasicUserDataStream<std::ostream> os(buf.get());
            boost::archive::binary_oarchive ar(os);
            ar << param;
        }
        auto& hdr = *reinterpret_cast<detail::Header*>(data.data());
        hdr.id = call_id;
        hdr.size = buf->size() - sizeof(hdr);
        hdr.tag = reinterpret_cast<std::uintptr_t>(tag);
        register_tag(tag);
        auto handler = [self = static_cast<Derived*>(this)->shared_from_this(), tag, buf](boost::system::error_code ec, size_t bytes_transferred)
        {
            if (!ec && buf->size() != bytes_transferred)
            {
                ec = make_error_code(error::marshalling_failed);
            }
            if (ec)
            {
                Response response { ec };
                self->socket().get_io_service().post([self, tag, ec]() {
                    if (self->unregister_tag(tag))
                        self->dispatch(tag, Response(ec));
                });
            }
        };
        async_write(static_cast<Derived*>(this)->socket(), buf->data(), handler);
    }

public:
    template <template <typename T> class Promise, std::uint32_t call_id, typename Ret>
    decltype(std::declval<Promise<Ret>>().get_future()) async_call(RpcDesc<call_id, Ret, void>)
    {
        auto promise = std::make_unique<detail::PromiseHandler<Ret, Promise>>();
        auto future = promise->get_future();
        async_call_raw(call_id, promise.release(), detail::VoidTag{});
        return future;
    }

    template <std::uint32_t call_id, typename Ret, typename Callback>
    void async_call(RpcDesc<call_id, Ret, void>, Callback&& cb)
    {
        auto promise = std::make_unique<detail::CallbackHandler<Ret>>(std::forward<Callback>(cb));
        async_call_raw(call_id, promise.release(), detail::VoidTag{});
    }

    template <template <typename T> class Promise, std::uint32_t call_id, typename Ret, typename ArgRpc, typename ArgPassed, typename std::enable_if_t<std::is_convertible<ArgPassed, ArgRpc>::value, int> = 0>
    decltype(std::declval<Promise<Ret>>().get_future()) async_call(RpcDesc<call_id, Ret, ArgRpc>, ArgPassed&& request)
    {
        auto promise = std::make_unique<detail::PromiseHandler<Ret, Promise>>();
        auto future = promise->get_future();
        async_call_raw(call_id, promise.release(), std::forward<ArgPassed>(request));
        return future;
    }

    template <std::uint32_t call_id, typename Ret, typename Callback, typename ArgRpc, typename ArgPassed, typename std::enable_if_t<std::is_convertible<ArgPassed, ArgRpc>::value, int> = 0>
    void async_call(RpcDesc<call_id, Ret, ArgRpc>, ArgPassed&& request, Callback&& cb)
    {
        auto promise = std::make_unique<detail::CallbackHandler<Ret>>(std::forward<Callback>(cb));
        async_call_raw(call_id, promise.release(), std::forward<ArgPassed>(request));
    }

public:
    void handle_session(boost::asio::yield_context const& yield)
    {
        auto& socket = static_cast<Derived*>(this)->socket();
        BOOST_SCOPE_EXIT_TPL(this_) {
            auto unresponded_tags = this_->flush_tags();
            if (!unresponded_tags.empty())
            {
                Response response { make_error_code(error::server_down) };
                for (auto tag : unresponded_tags)
                    static_cast<Derived*>(this_)->dispatch(tag, response);
            }
        } BOOST_SCOPE_EXIT_END
        while (true)
        {
            detail::Header rpc{};
            async_read( socket, boost::asio::buffer(&rpc, sizeof(rpc)), yield );
            Response response;
            response.ec = rpc.id;
            if (rpc.size)
            {
                response.msg.resize(rpc.size);
                auto n = async_read( socket, boost::asio::mutable_buffer(&response.msg[0], rpc.size), yield );
                if (n != rpc.size)
                {
                    response = Response{ make_error_code(error::marshalling_failed) };
                }
            }
            auto tag = reinterpret_cast<void*>(rpc.tag);
            if (unregister_tag(tag))
                static_cast<Derived*>(this)->dispatch(tag, response);
        }
    }

    void dispatch(void* tag, Response const& response)
    {
        std::unique_ptr<IRpcResponseHandler> handler { static_cast<IRpcResponseHandler*>(tag) };
        handler->process(response);
    }

private:
    void register_tag(void* tag)
    {
        std::lock_guard<std::mutex> lock(m_tagsMutex);
        m_tags.push_back(tag);
    }

    bool unregister_tag(void* tag)
    {
        std::lock_guard<std::mutex> lock(m_tagsMutex);
        auto it = std::remove(m_tags.begin(), m_tags.end(), tag);
        if (m_tags.end() != it)
        {
            m_tags.erase(it, m_tags.end());
            return true;
        }
        return false;
    }

    std::vector<void*> flush_tags()
    {
        std::lock_guard<std::mutex> lock(m_tagsMutex);
        return std::move(m_tags);
    }

private:
    std::mutex m_tagsMutex;
    std::vector<void*> m_tags;
};

} // namespace SimpleRPC
} // namespace ItvCvUtils

