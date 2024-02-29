#pragma once

#include "asio_unix_fd_passing.hpp"
#include "Envar.h"

#include <boost/filesystem.hpp>
#include <boost/scope_exit.hpp>
#include <boost/optional.hpp>

#include <boost/asio.hpp>
#define BOOST_COROUTINES_NO_DEPRECATION_WARNING
#include <boost/asio/spawn.hpp>
#include <boost/bind.hpp>

#include <fmt/format.h>

#include <array>
#include <vector>
#include <mutex>
#include <cstdint>
#include <memory>
#include <iosfwd>
#include <chrono>
#include <regex>

namespace ItvCvUtils
{

struct basic_fd_passing_server_data_header
{
    size_t streambuf_size;
    size_t num_fd;
};

template <typename fd_type, typename data_header_type = basic_fd_passing_server_data_header>
struct server_data_snapshot
{
    data_header_type header;
    boost::asio::streambuf streambuf;
    std::vector<fd_type> fd;

    server_data_snapshot(data_header_type const& gen)
        : header ( gen )
    {
        prepare_buffer_for_fd_passing(header.num_fd);
    }

    server_data_snapshot()
        : header ()
    {}

    void prepare_buffer_for_fd_passing(size_t num_fd)
    {
        fd.reserve(num_fd);
    }

    void complete()
    {
        header.streambuf_size = streambuf.size();
        if (0 == header.streambuf_size)
        {
            std::ostream os(&streambuf);
            os << '\0';
            header.streambuf_size = 1;
        }
        header.num_fd = fd.size();
    }

    std::streambuf* stream_buffer()
    {
        return &streambuf;
    }

    template <typename FD>
    void pass_fd(FD&& arg)
    {
        fd.emplace_back( std::forward<FD>(arg) );
    }
};

namespace detail
{

const mode_t DEFAULT_IPC_DIR_PERMISSIONS = (S_IRUSR|S_IWUSR|S_IXUSR|S_IRGRP|S_IWGRP|S_IXGRP|S_IROTH|S_IWOTH|S_IXOTH|S_ISVTX);

inline void SetPermissions(const boost::filesystem::path& p, mode_t ipcFilesPermissions)
{
    if ( -1 == ::chmod( p.c_str(), ipcFilesPermissions ) )
        throw boost::system::system_error(errno, boost::system::generic_category(), "chmod on IPC object has failed");
}

inline void ThrowIpcDirDoesNotExist(const char ipcTypeStr[])
{
    throw std::runtime_error(std::string("IPC directory for file descriptors passing server does not exist: ") + ipcTypeStr);
}

template <const std::uint32_t ipcTypeID, typename FOnPathItemDoesNotExist>
inline boost::filesystem::path GetIpcDirectory(FOnPathItemDoesNotExist onPathItemDoesNotExist)
{
    const char ipcTypeStr[] = { char(ipcTypeID&0xFF), char((ipcTypeID>>8)&0xFF), char((ipcTypeID>>16)&0xFF), char((ipcTypeID>>24)&0xFF), '\0' };
    boost::filesystem::path ipcDir{ CEnvar::RuntimeDirectoryShared() };
    bool handle_non_existent = !exists(ipcDir);
    if (handle_non_existent && !onPathItemDoesNotExist(ipcDir))
        ThrowIpcDirDoesNotExist(ipcTypeStr);
    ipcDir /= ipcTypeStr;
    if ((handle_non_existent || !exists(ipcDir)) && !onPathItemDoesNotExist(ipcDir))
        ThrowIpcDirDoesNotExist(ipcTypeStr);
    return ipcDir;
}

template <typename UnixDomainSocket> void AssignNonInheritableNativeSocketImpl(UnixDomainSocket& sock)
{
    boost::asio::local::stream_protocol proto;
    sock.assign(proto, ::socket(proto.family(), proto.type() | SOCK_CLOEXEC, proto.protocol()) );
}

template <typename T> void unused_variable(T const&) {}

class IpcChannelFSOSentry : boost::noncopyable
{
    boost::filesystem::path m_fso;

public:
    IpcChannelFSOSentry() = default;
    explicit IpcChannelFSOSentry(boost::filesystem::path ipcChannel) : m_fso( std::move(ipcChannel) ) {}
    IpcChannelFSOSentry(IpcChannelFSOSentry&& other) : m_fso( std::move(other.m_fso) ) {}
    IpcChannelFSOSentry& operator=(IpcChannelFSOSentry&& other)
    {
        m_fso.swap( other.m_fso );
        return *this;
    }

    ~IpcChannelFSOSentry()
    {
        if (!m_fso.empty())
        {
            boost::system::error_code ec;
            remove( m_fso, ec );
        }
    }

    const boost::filesystem::path& path() const { return m_fso; }
    void release() { m_fso.clear(); }
};

#ifdef __linux__
inline
boost::optional<boost::filesystem::path> GetFileNameByHandle(boost::asio::local::stream_protocol::acceptor::native_handle_type fd)
{
    struct stat stat;
    if (-1 != fstat(fd, &stat) && 0 < stat.st_nlink)
    {
        std::ifstream socks("/proc/net/unix");
        std::regex re(fmt::format(FMT_STRING("\\s+{}\\s+(.*)$"), stat.st_ino));
        std::string line;
        while (std::getline(socks, line))
        {
            std::smatch m;
            if (std::regex_search(line, m, re))
                return boost::filesystem::path(m[1].str());
        }
    }
    return boost::none;
}
#else
# error "GetFileNameByHandle has not been implemented for your platform"
#endif

} // namespace detail

template <class Derived, typename data_header_type = basic_fd_passing_server_data_header>
class FDPassingServer
{
    template <typename FOnUnusedIpcChannel>
    static void cleanupIpcDir(boost::asio::io_service& io_service, boost::filesystem::path const& ipcDir, boost::filesystem::path const& toBeCheckedSynchronously, FOnUnusedIpcChannel onUnusedIpcChannelFound)
    {
        const bool checkForSync = exists( toBeCheckedSynchronously );
        for ( boost::filesystem::directory_iterator ipcDirIt( ipcDir ), itEnd; ipcDirIt != itEnd; ++ipcDirIt )
        {
            auto const& ipcChannel = ipcDirIt->path();
            auto testClient = std::make_shared<boost::asio::local::stream_protocol::socket>( io_service );
            detail::AssignNonInheritableNativeSocketImpl( *testClient );
            auto handleConnect = [testClient, ipcChannel, onUnusedIpcChannelFound](boost::system::error_code const& ec)
            {
                const bool ipcChannelIsDead = !!ec;
                if (ipcChannelIsDead)
                {
                    detail::IpcChannelFSOSentry toRemove( ipcChannel );
                    onUnusedIpcChannelFound( ipcChannel.filename().string() );
                }
            };
            if (checkForSync && ipcChannel == toBeCheckedSynchronously)
            {
                boost::system::error_code ec;
                testClient->connect( ipcChannel.string(), ec );
                handleConnect( ec );
            }
            else
                testClient->async_connect( ipcChannel.string(), std::move(handleConnect) );
        }
    }
    
protected:
    template <typename FOnUnusedIpcChannel>
    FDPassingServer(boost::asio::io_service& io_service, std::string const& ipcChannelName, FOnUnusedIpcChannel onUnusedIpcChannelFound, mode_t ipcFilesPermissions = detail::DEFAULT_IPC_DIR_PERMISSIONS)
        : m_acceptor( io_service )
        , m_strand( io_service )
        , m_acceptorRestartTimer( io_service )
    {
        bool didIpcDirExist = true;
        auto onIpcDirDoesNotExist = [&didIpcDirExist, ipcFilesPermissions](const boost::filesystem::path& p) -> bool
        {
            didIpcDirExist = false;
            if ( !create_directory(p) )
                return false;
            detail::SetPermissions( p, ipcFilesPermissions );
            return true;
        };
        auto ipcDir = detail::GetIpcDirectory<Derived::handshake_phrase>( onIpcDirDoesNotExist );
        auto ipcChannel = ipcDir;
        ipcChannel /= ipcChannelName;
        if (didIpcDirExist)
        {
            cleanupIpcDir(io_service, ipcDir, ipcChannel, std::move(onUnusedIpcChannelFound));
        }

        detail::AssignNonInheritableNativeSocketImpl(m_acceptor);
        m_acceptor.bind( ipcChannel.string() );

        m_fso = detail::IpcChannelFSOSentry( std::move( ipcChannel ) );
        detail::SetPermissions( m_fso.path(), ipcFilesPermissions & ~S_ISVTX );

        m_acceptor.listen();
    }

    FDPassingServer(boost::asio::io_service& io_service, boost::asio::local::stream_protocol::acceptor::native_handle_type native_acceptor)
        : m_acceptor( io_service, boost::asio::local::stream_protocol(), native_acceptor )
        , m_strand( io_service )
        , m_acceptorRestartTimer( io_service )
    {
        if ( auto maybePath = detail::GetFileNameByHandle(native_acceptor) )
            m_fso = detail::IpcChannelFSOSentry( std::move( *maybePath ) );
    }

    ~FDPassingServer()
    {
    }

    boost::asio::local::stream_protocol::acceptor::native_handle_type release()
    {
        m_fso.release();
        return m_acceptor.release();
    }

    boost::filesystem::path ipcChannel() const { return m_fso.path(); }

    typedef boost::asio::fd_cref fd_type;
    typedef server_data_snapshot<fd_type, data_header_type> server_data_snapshot_type;
    typedef std::shared_ptr<server_data_snapshot_type> server_data_snapshot_ptr_type;

protected:
    class Session : public std::enable_shared_from_this<Session>
    {
        boost::asio::local::stream_protocol::socket m_socket;
        boost::asio::io_service::strand m_strand;
        std::shared_ptr< server_data_snapshot_type > m_data;
        std::shared_ptr< Derived > m_owner;
        union {
            unsigned char m_idle_buf[4];
            std::uint32_t m_handshake_buf;
        };

    public:
        Session(boost::asio::io_service& io_service, std::shared_ptr< server_data_snapshot_type > data, std::shared_ptr< Derived > const& owner)
            : m_socket(io_service)
            , m_strand( io_service )
            , m_data( std::move(data) )
            , m_owner( owner )
            , m_idle_buf()
        {}

        enum State { started_successfully, closed };

        boost::asio::local::stream_protocol::socket& socket() { return m_socket; }
        std::shared_ptr< server_data_snapshot_type > data() const { return m_data; }

        template <typename FOnSessionStateChanged>
        static void start(std::shared_ptr<Session> const& self, FOnSessionStateChanged onSessionStateChanged)
        {
            boost::asio::spawn( self->m_strand, boost::bind(&Session::run<FOnSessionStateChanged>, self, _1, std::move(onSessionStateChanged)) );
        }

        bool shutdown()
        {
            boost::system::error_code ec;
            m_socket.shutdown( boost::asio::local::stream_protocol::socket::shutdown_both, ec );
            if (!ec)
            {
                m_socket.close(ec);
                return true;
            }
            return false;
        }

    private:
        bool handshake(boost::asio::yield_context yield)
        {
            async_read( m_socket, boost::asio::buffer(&m_handshake_buf, sizeof(m_handshake_buf)), yield );
            return Derived::handshake_phrase == m_handshake_buf;
        }

        void pass_data_and_wait_for_disconnect(boost::asio::yield_context yield)
        {
            async_write( m_socket, boost::asio::buffer(&m_data->header, sizeof(m_data->header)), yield );
            async_write( m_socket, boost::asio::attach_const_file_descriptors( m_data->streambuf.data(), m_data->fd ), yield );
            m_owner->handle_session( std::static_pointer_cast<typename Derived::session_type>(this->shared_from_this()), yield );
        }

        template <typename FOnSessionStateChanged>
        static void run(std::shared_ptr<Session> const& self, boost::asio::yield_context yield, FOnSessionStateChanged const& onSessionStateChanged)
        try
        {
            BOOST_SCOPE_EXIT_TPL(&self, &onSessionStateChanged) {
                if (self->shutdown())
                    onSessionStateChanged(self, closed);
            } BOOST_SCOPE_EXIT_END
            if (!self->handshake(yield))
                return;
            onSessionStateChanged(self, started_successfully);
            self->pass_data_and_wait_for_disconnect(yield);
        }
        catch(std::exception const&){}
    };
    typedef std::shared_ptr<Session> session_ptr;

public:
    void publishServerData(std::shared_ptr< server_data_snapshot_type > data)
    {
        data->complete();
        spawn_new_session( std::move(data) );
    }

    void shutdownPublisher()
    {
        m_strand.dispatch( [self = static_cast<Derived*>(this)->shared_from_this()]() {
            self->shutdown();
        } );
    }

    template <typename SessionOfDerived>
    void handle_session(std::shared_ptr<SessionOfDerived> session, boost::asio::yield_context const& yeild)
    {
        std::uint8_t idle_buf;
        async_read( session->socket(), boost::asio::buffer(&idle_buf, sizeof(idle_buf)), yeild );
    }

protected:
    struct RestartAcceptorIn
    {
        std::chrono::milliseconds timeout;
    };

    boost::optional<RestartAcceptorIn> onAcceptorError(boost::system::error_code const& ec)
    {
        using namespace std::chrono_literals;
        if (boost::asio::error::no_buffer_space == ec
            || boost::asio::error::no_descriptors == ec
            || boost::asio::error::no_memory == ec )
            return RestartAcceptorIn { 10s };
        return boost::none;
    }

private:
    std::shared_ptr<Session> make_session(boost::asio::io_service& io, std::shared_ptr< server_data_snapshot_type >&& data)
    {
        return std::make_shared<Session>(io, std::move(data), this->shared_from_this());
    }

    void shutdown()
    {
        m_fso = detail::IpcChannelFSOSentry();
        m_acceptorRestartTimer.cancel();
        boost::system::error_code ec;
        m_acceptor.close(ec);

        for ( auto& session : m_sessions )
            session->shutdown();
        m_sessions.clear();
    }

    void spawn_new_session(std::shared_ptr< server_data_snapshot_type > data)
    {
        auto self = static_cast<Derived*>(this)->shared_from_this();
        m_strand.dispatch(
            [self, new_session = self->make_session(m_acceptor.get_io_service(), std::move(data))]()
            {
                if (self->m_acceptor.is_open())
                    self->m_acceptor.async_accept(
                        new_session->socket(),
                        boost::bind(&FDPassingServer::handle_accept, self,
                            new_session, boost::asio::placeholders::error)
                    );
            }
        );
    }

    void register_session(const session_ptr& session)
    {
        m_strand.dispatch( [self = static_cast<Derived*>(this)->shared_from_this(), session]() {
            self->m_sessions.push_back( session );
        } );
    }

    void unregister_session(const session_ptr& session)
    {
        m_strand.dispatch( [self = static_cast<Derived*>(this)->shared_from_this(), session]() {
            self->m_sessions.erase( std::remove(self->m_sessions.begin(), self->m_sessions.end(), session), self->m_sessions.end() );
        } );
    }

    void handle_accept(session_ptr new_session, const boost::system::error_code& error)
    {
        if (!error)
        {
            auto self = static_cast<Derived*>(this)->shared_from_this();
            auto handle_session_state_transition = [self](session_ptr const& session, typename Session::State state)
            {
                if ( Session::started_successfully == state )
                    self->register_session(session);
                else
                    self->unregister_session(session);
            };

            Session::start( new_session, handle_session_state_transition );
            spawn_new_session( new_session->data() );
        }
        else if (boost::asio::error::operation_aborted != error)
        {
            if ( boost::optional<RestartAcceptorIn> restartIn = static_cast<Derived*>(this)->onAcceptorError(error) )
            {
                if (std::chrono::milliseconds::zero() == restartIn->timeout)
                {
                    spawn_new_session( new_session->data() );
                }
                else
                {
                    m_acceptorRestartTimer.expires_from_now( boost::posix_time::milliseconds(restartIn->timeout.count()) );
                    m_acceptorRestartTimer.async_wait(
                        [self = static_cast<Derived*>(this)->shared_from_this(), data = std::move(new_session->data())] (boost::system::error_code const& ec) {
                            if (!ec)
                                self->spawn_new_session( std::move(data) );
                        }
                    );

                }
            }
        }
    }

private:
    boost::asio::local::stream_protocol::acceptor m_acceptor;
    boost::asio::io_service::strand m_strand;
    boost::asio::deadline_timer m_acceptorRestartTimer;
    detail::IpcChannelFSOSentry m_fso;

    std::vector< session_ptr > m_sessions;
};

template <class Derived, typename data_header_type = basic_fd_passing_server_data_header>
class FDReceivingClient
{
protected:
    FDReceivingClient(boost::asio::io_service& io_service, std::string const& ipcChannelName)
        : m_socket( io_service )
        , m_strand( io_service )
        , m_sessionClosed( ATOMIC_FLAG_INIT )
    {
        detail::AssignNonInheritableNativeSocketImpl(m_socket);
        const std::uint32_t handshake_phrase = Derived::handshake_phrase;
        auto failIfIpcDirDoesNotExist = [](const boost::filesystem::path&) { return false; };
        auto ipcChannel = detail::GetIpcDirectory<handshake_phrase>( failIfIpcDirDoesNotExist ) /= ipcChannelName;
        m_socket.connect(ipcChannel.string());
        m_socket.send( boost::asio::buffer(&handshake_phrase, sizeof(handshake_phrase)) );
    }

    ~FDReceivingClient()
    {
        closeConnection();
    }

    typedef boost::asio::fd_ref fd_type;
    typedef server_data_snapshot<fd_type, data_header_type> server_data_snapshot_type;
    typedef std::shared_ptr<server_data_snapshot_type> server_data_snapshot_ptr_type;

public:
    server_data_snapshot_ptr_type initializeServerDataBuffer()
    {
        data_header_type header;
        boost::asio::read( m_socket, boost::asio::buffer(&header, sizeof(header)) );
        return std::make_shared<server_data_snapshot_type>(header);
    }

    void readServerData(server_data_snapshot_type& data)
    {
        if (data.fd.size() != data.header.num_fd)
            throw std::logic_error( fmt::format( FMT_STRING("server passed {} file descriptors but reader expects {}"), data.header.num_fd, data.fd.size() ) );
        boost::asio::read( m_socket, boost::asio::attach_mutable_file_descriptors( data.streambuf.prepare(data.header.streambuf_size), data.fd ) );
        data.streambuf.commit( data.header.streambuf_size );
    }

    bool closeConnection()
    {
        if (m_sessionClosed.test_and_set())
            return false;
        boost::system::error_code shutdown_error;
        m_socket.shutdown( boost::asio::local::stream_protocol::socket::shutdown_both, shutdown_error );
        if (!shutdown_error)
            m_socket.close(shutdown_error);
        return true;
    }

    enum class EDisconnectInitiatedBy { Server, Client };
    void asyncWaitForServerDisconnect()
    {
        boost::asio::spawn( m_strand, [self = static_cast<Derived*>(this)->shared_from_this()](boost::asio::yield_context yield)
        {
            try
            {
                self->handle_session(yield);
            }
            catch(std::exception const&){}
            bool justClosed = self->closeConnection();
            self->onServerDisconnected(justClosed ? EDisconnectInitiatedBy::Server : EDisconnectInitiatedBy::Client);
        } );
    }

    /// @note to be overriden by Derived
    void handle_session( boost::asio::yield_context const& yeild )
    {
        std::uint8_t idle_buf;
        async_read( m_socket, boost::asio::buffer(&idle_buf, sizeof(idle_buf)), yeild );
    }

    boost::asio::local::stream_protocol::socket& socket() { return m_socket; }

private:
    boost::asio::local::stream_protocol::socket m_socket;
    boost::asio::io_service::strand m_strand;
    std::atomic_flag m_sessionClosed;
};

} // namespace ItvCvUtils
