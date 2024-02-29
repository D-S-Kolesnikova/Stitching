#include <ItvCvUtils/simple_rpc.hpp>

#define BOOST_THREAD_PROVIDES_FUTURE
#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/thread/future.hpp>
#include <boost/optional.hpp>
#include <mutex>
#include <condition_variable>

#include <boost/test/unit_test.hpp>

namespace {

RPC_DESCRIBE(0u, void, sleep_ms, unsigned);
RPC_DESCRIBE(1u, std::string, echo, std::string);
RPC_DESCRIBE(2u, void, error, std::string);

class Server
    : public ItvCvUtils::SimpleRPC::Server<Server>
    , public std::enable_shared_from_this<Server>
{
    class Session : public std::enable_shared_from_this<Session>
    {
        boost::asio::ip::tcp::socket m_socket;
        boost::asio::io_service::strand m_strand;
        std::shared_ptr< Server > m_server;

    public:
        Session(boost::asio::io_service& io_service, std::shared_ptr< Server > owner)
            : m_socket(io_service)
            , m_strand( io_service )
            , m_server( std::move(owner) )
        {}

        enum State { started_successfully, closed };

        boost::asio::ip::tcp::socket& socket() { return m_socket; }

        template <typename FOnSessionStateChanged>
        static void start(std::shared_ptr<Session> const& self, FOnSessionStateChanged onSessionStateChanged)
        {
            boost::asio::spawn( self->m_strand, boost::bind(&Session::run<FOnSessionStateChanged>, self, _1, std::move(onSessionStateChanged)) );
        }

        bool shutdown()
        {
            boost::system::error_code ec;
            m_socket.shutdown( boost::asio::ip::tcp::socket::shutdown_both, ec );
            if (!ec)
            {
                m_socket.close(ec);
                return true;
            }
            return false;
        }

    private:
        template <typename FOnSessionStateChanged>
        static void run(std::shared_ptr<Session> const& self, boost::asio::yield_context yield, FOnSessionStateChanged const& onSessionStateChanged)
        try
        {
            BOOST_SCOPE_EXIT_TPL(&self, &onSessionStateChanged) {
                self->shutdown();
                onSessionStateChanged(self, closed);
            } BOOST_SCOPE_EXIT_END
            onSessionStateChanged(self, started_successfully);
            self->m_server->handle_session(self, yield);
        }
        catch(std::exception const&){}
    };
    typedef std::shared_ptr<Session> session_ptr;

public:
    Server(boost::asio::io_service& io, std::function<void (unsigned nrSessions)> onSessionsChanges)
        : m_acceptor(io)
        , m_strand(io)
        , m_onSessionsChanges(std::move(onSessionsChanges))
    {
        boost::asio::ip::tcp::endpoint ep(boost::asio::ip::tcp::v4(), 0);
        m_acceptor.open(ep.protocol());
        m_acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
        m_acceptor.bind(ep);
	m_acceptor.listen();
    }

    unsigned short port() const
    {
        return m_acceptor.local_endpoint().port();
    }

    void start()
    {
        spawn_new_session();
    }

    void stop()
    {
        m_strand.dispatch( [self = this->shared_from_this()]() {
            self->shutdown();
        } );
    }

public:
    using ItvCvUtils::SimpleRPC::Server<Server>::rpc_context_type;
    using session_type = Session;

    RPC_DISPATCH_TABLE()
        RPC_DISPATCH_TABLE_ENTRY(sleep_ms);
        RPC_DISPATCH_TABLE_ENTRY(echo);
        RPC_DISPATCH_TABLE_ENTRY(error);
    RPC_DISPATCH_TABLE_END()

    void process_sleep_ms(unsigned timeout_ms, rpc_context_type const& context)
    {
        boost::asio::steady_timer timer(m_acceptor.get_io_service());
        timer.expires_after( std::chrono::milliseconds(timeout_ms) );
        timer.async_wait(context.yield);
    }

    std::string process_echo(std::string&& msg, rpc_context_type const&)
    {
        return std::move(msg);
    }

    void process_error(std::string&& msg, rpc_context_type const&)
    {
        throw boost::system::system_error(make_error_code(ItvCvUtils::SimpleRPC::error::user_defined_error), std::move(msg));
    }

private:
    void shutdown()
    {
        boost::system::error_code ec;
        m_acceptor.close(ec);

        for ( auto& session : m_sessions )
            session->shutdown();
    }

    void spawn_new_session()
    {
        auto self = this->shared_from_this();
        m_strand.dispatch(
            [self, new_session = std::make_shared<Session>(m_acceptor.get_io_service(), self)]()
            {
                if (self->m_acceptor.is_open())
                    self->m_acceptor.async_accept(
                        new_session->socket(),
                        boost::bind(&Server::handle_accept, self,
                            new_session, boost::asio::placeholders::error)
                    );
            }
        );
    }

    void register_session(const session_ptr& session)
    {
        m_strand.dispatch( [self = this->shared_from_this(), session]() {
            self->m_sessions.push_back( session );
            self->m_onSessionsChanges( self->m_sessions.size() );
        } );
    }

    void unregister_session(const session_ptr& session)
    {
        m_strand.dispatch( [self = this->shared_from_this(), session]() {
            self->m_sessions.erase( std::remove(self->m_sessions.begin(), self->m_sessions.end(), session), self->m_sessions.end() );
            self->m_onSessionsChanges( self->m_sessions.size() );
        } );
    }

    void handle_accept(session_ptr new_session, const boost::system::error_code& error)
    {
        if (!error)
        {
            auto self = this->shared_from_this();
            auto handle_session_state_transition = [self](session_ptr const& session, typename Session::State state)
            {
                if ( Session::started_successfully == state )
                    self->register_session(session);
                else
                    self->unregister_session(session);
            };

            Session::start( new_session, handle_session_state_transition );
            spawn_new_session();
        }
    }

private:
    boost::asio::ip::tcp::acceptor m_acceptor;
    boost::asio::io_service::strand m_strand;

    std::vector< session_ptr > m_sessions;
    std::function<void (unsigned nrSessions)> m_onSessionsChanges;
};

class Client
    : public ItvCvUtils::SimpleRPC::Client<Client>
    , public std::enable_shared_from_this<Client>
{
public:
    enum class SessionClosedBy { Server, Client };
    Client(boost::asio::io_service& io, unsigned short port, std::function<void (SessionClosedBy)> onSessionClosed)
        : m_socket(io)
        , m_onSessionClosed(std::move(onSessionClosed))
        , m_sessionClosed{ ATOMIC_FLAG_INIT }
    {
        m_socket.connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), port));
    }

    void start()
    {
        boost::asio::spawn(m_socket.get_io_service(), std::bind(&Client::run, shared_from_this(), std::placeholders::_1));
    }

    void stop()
    {
        closeConnection();
    }

    boost::asio::ip::tcp::socket& socket() { return m_socket; }

private:
    void run(boost::asio::yield_context const& yield)
    {
        try
        {
            handle_session(yield);
        }
        catch (std::exception const&){}
        m_onSessionClosed(closeConnection() ? SessionClosedBy::Server : SessionClosedBy::Client);
    }

    bool closeConnection()
    {
        if (m_sessionClosed.test_and_set())
            return false;
        boost::system::error_code shutdown_error;
        m_socket.shutdown( boost::asio::ip::tcp::socket::shutdown_both, shutdown_error );
        if (!shutdown_error)
            m_socket.close(shutdown_error);
        return true;
    }

private:
    boost::asio::ip::tcp::socket m_socket;
    std::function<void (SessionClosedBy)> m_onSessionClosed;
    std::atomic_flag m_sessionClosed;
};

class Executor
{
    boost::asio::io_service m_io;
    boost::optional<boost::asio::io_service::work> m_work;
    boost::thread_group m_workers;
public:
    Executor() = default;
    ~Executor() { Stop(); }
    
    void Start(size_t nrThreads)
    {
        if (m_work)
            throw std::logic_error("Executor double start");
        m_work.emplace(m_io);
        if (0 == nrThreads)
            nrThreads = boost::thread::hardware_concurrency();
        for (size_t i = 0; i < nrThreads; ++i)
            m_workers.create_thread([this](){ m_io.run(); });
    }

    void Stop()
    {
        if (!m_work)
            return;
        m_work.reset();
        m_workers.join_all();
    }

    boost::asio::io_service& GetIO() { return m_io; }
};

class Fixture
{
    Executor m_serverExec;
    std::shared_ptr<Server> m_server;
    Executor m_clientsExec;

    unsigned m_nrClients;
    std::mutex m_nrClientsMutex;
    std::condition_variable m_nrClientsChanged;

private:
    void SetNumberOfClients(unsigned nrClients)
    {
        std::lock_guard<std::mutex> lock(m_nrClientsMutex);
        m_nrClients = nrClients;
        m_nrClientsChanged.notify_all();
    }

public:
    enum class ThreadindPolicy { Single, Multi };

    Fixture(ThreadindPolicy server, ThreadindPolicy clients)
        : m_server(std::make_shared<Server>(m_serverExec.GetIO(), std::bind(&Fixture::SetNumberOfClients, this, std::placeholders::_1) ))
    {
        m_serverExec.Start(ThreadindPolicy::Single == server ? 1u : 0u);
        m_clientsExec.Start(ThreadindPolicy::Single == clients ? 1u : 0u);
        m_server->start();
    }

    void StopServer()
    {
        m_server->stop();
    }

    std::shared_ptr<Client> StartClient(std::function<void (Client::SessionClosedBy)> onSessionClosed = [](auto){})
    {
        auto client = std::make_shared<Client>(m_clientsExec.GetIO(), m_server->port(), std::move(onSessionClosed));
        client->start();
        return client;
    }

    bool WaitTillNumberOfConnectedClientsEquals(unsigned N, std::chrono::milliseconds timeout = {})
    {
        std::unique_lock<std::mutex> lock(m_nrClientsMutex);
        if (0u == timeout.count())
        {
            m_nrClientsChanged.wait(lock, [this, N]() { return m_nrClients == N; });
            return true;
        }
        return m_nrClientsChanged.wait_for(lock, timeout, [this, N]() { return m_nrClients == N; });
    }

    ~Fixture()
    {
        m_server->stop();
        m_clientsExec.Stop();
        m_serverExec.Stop();
    }
};

template <Fixture::ThreadindPolicy server, Fixture::ThreadindPolicy clients>
struct TFixture : public Fixture
{
    TFixture() : Fixture(server, clients) {}
};
using Fixture_ServerST_ClientST = TFixture<Fixture::ThreadindPolicy::Single, Fixture::ThreadindPolicy::Single>;
using Fixture_ServerMT_ClientMT = TFixture<Fixture::ThreadindPolicy::Multi, Fixture::ThreadindPolicy::Multi>;

} // anonymous namespace

BOOST_FIXTURE_TEST_SUITE(SimpleRPC_SingleThreaded, Fixture_ServerST_ClientST)

BOOST_AUTO_TEST_CASE(Echo)
{
    auto client = this->StartClient();
    const std::string sECHO("ECHO");
    auto future = client->async_call<std::promise>(rpc_echo, sECHO);
    BOOST_REQUIRE(future.wait_for(std::chrono::seconds(60)) == std::future_status::ready);
    BOOST_REQUIRE_EQUAL(sECHO, future.get());
}

BOOST_AUTO_TEST_CASE(Error)
{
    auto client = this->StartClient();
    const std::string sMSG("ERRMSG");
    auto future = client->async_call<std::promise>(rpc_error, sMSG);
    BOOST_REQUIRE(future.wait_for(std::chrono::seconds(60)) == std::future_status::ready);
    try
    {
        future.get();
        BOOST_FAIL("future.get() was supposed to re-throw the exception passed from server");
    }
    catch (std::exception const& e)
    {
        BOOST_REQUIRE(nullptr != strstr(e.what(), sMSG.c_str()));
    }
}

BOOST_AUTO_TEST_CASE(Sleep)
{
    auto client = this->StartClient();
    unsigned const TIMEOUT_MS = 200u;
    auto future = client->async_call<std::promise>(rpc_sleep_ms, TIMEOUT_MS);
    BOOST_REQUIRE(future.wait_for(std::chrono::milliseconds(static_cast<unsigned>(TIMEOUT_MS * 0.5))) == std::future_status::timeout);
    BOOST_REQUIRE(future.wait_for(std::chrono::milliseconds(static_cast<unsigned>(TIMEOUT_MS * 0.7))) == std::future_status::ready);
}

BOOST_AUTO_TEST_CASE(SequentialProcessing)
{
    auto client = this->StartClient();
    unsigned const TIMEOUT_FAST_MS = 200u;
    unsigned const TIMEOUT_SLOW_MS = 3 * TIMEOUT_FAST_MS;
    std::vector<boost::future<void>> futures;
    futures.emplace_back( client->async_call<boost::promise>(rpc_sleep_ms, TIMEOUT_SLOW_MS) );
    futures.emplace_back( client->async_call<boost::promise>(rpc_sleep_ms, TIMEOUT_FAST_MS) );
    auto faster = boost::wait_for_any(futures.begin(), futures.end());
    BOOST_REQUIRE_MESSAGE(futures.begin() == faster, "Requests from single client are expected to be handled sequentially");
}

BOOST_AUTO_TEST_CASE(Sleep2Clients)
{
    auto clientFast = this->StartClient();
    auto clientSlow = this->StartClient();
    unsigned const TIMEOUT_FAST_MS = 200u;
    unsigned const TIMEOUT_SLOW_MS = 3 * TIMEOUT_FAST_MS;
    std::vector<boost::future<void>> futures;
    futures.emplace_back( clientSlow->async_call<boost::promise>(rpc_sleep_ms, TIMEOUT_SLOW_MS) );
    futures.emplace_back( clientFast->async_call<boost::promise>(rpc_sleep_ms, TIMEOUT_FAST_MS) );
    auto faster = boost::wait_for_any(futures.begin(), futures.end());
    BOOST_REQUIRE(&*futures.rbegin() == &*faster);
    BOOST_REQUIRE(futures.begin()->wait_for(boost::chrono::milliseconds(static_cast<unsigned>((TIMEOUT_SLOW_MS - TIMEOUT_FAST_MS) * 1.2))) == boost::future_status::ready);
}

BOOST_AUTO_TEST_CASE(ServerDisconnected)
{
    constexpr int N = 10;
    auto counter = std::make_shared<int>(N);
    auto promise = std::make_shared<std::promise<void>>();
    auto all_clients_closed = promise->get_future();
    auto onDisconnect = [counter, promise](Client::SessionClosedBy by)
    {
        if (Client::SessionClosedBy::Server == by)
            if (0 == --*counter)
                promise->set_value();
    };
    for (int i = 0; i < N; ++i)
        this->StartClient(onDisconnect);
    BOOST_REQUIRE( WaitTillNumberOfConnectedClientsEquals(N, std::chrono::seconds(1)) );
    this->StopServer();
    BOOST_REQUIRE( WaitTillNumberOfConnectedClientsEquals(0, std::chrono::seconds(1)) );
    BOOST_REQUIRE( all_clients_closed.wait_for(std::chrono::seconds(60)) == std::future_status::ready );
}

BOOST_AUTO_TEST_CASE(ClientDisconnected)
{
    constexpr int N = 10;
    auto counter = std::make_shared<int>(N);
    auto promise = std::make_shared<std::promise<void>>();
    auto all_clients_closed = promise->get_future();
    auto onDisconnect = [counter, promise](Client::SessionClosedBy by)
    {
        if (Client::SessionClosedBy::Client == by)
            if (0 == --*counter)
                promise->set_value();
    };
    std::vector<std::shared_ptr<Client>> clients;
    for (int i = 0; i < N; ++i)
        clients.emplace_back( this->StartClient(onDisconnect) );
    BOOST_REQUIRE( WaitTillNumberOfConnectedClientsEquals(N, std::chrono::seconds(1)) );
    for (auto const& client : clients)
        client->stop();
    BOOST_REQUIRE( WaitTillNumberOfConnectedClientsEquals(0, std::chrono::seconds(1)) );
    BOOST_REQUIRE( all_clients_closed.wait_for(std::chrono::seconds(60)) == std::future_status::ready );
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(SimpleRPC_MultiThreaded, Fixture_ServerMT_ClientMT)

BOOST_AUTO_TEST_CASE(Echo)
{
    auto const N = std::thread::hardware_concurrency();
    boost::barrier start_signal(N + 1);
    boost::thread_group threads;
    std::atomic<unsigned> nr_fails{ 0 };
    for (auto t = N; t > 0; --t)
        threads.create_thread([&start_signal, this, &nr_fails, t](){
            start_signal.wait();
            auto client = this->StartClient();
            for (int iTry = 0; iTry < 100; ++iTry)
            {
                auto sECHO = fmt::format(FMT_STRING("T{}: ECHO#{}"), t, iTry);
                auto future = client->async_call<std::promise>(rpc_echo, sECHO);
                if (future.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
                    ++nr_fails;
                else if (sECHO != future.get())
                    ++nr_fails;
            }
            start_signal.wait();
            client->stop();
        });
    start_signal.wait();
    BOOST_REQUIRE( WaitTillNumberOfConnectedClientsEquals(N, std::chrono::seconds(1)) );
    start_signal.wait();
    BOOST_REQUIRE( WaitTillNumberOfConnectedClientsEquals(0, std::chrono::seconds(1)) );
    threads.join_all();
    BOOST_REQUIRE_EQUAL(0u, nr_fails.load());
}

BOOST_AUTO_TEST_SUITE_END()

