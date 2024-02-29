#ifndef _ASIO_UNIX_FD_PASSING_HPP
#define _ASIO_UNIX_FD_PASSING_HPP

#ifdef BOOST_ASIO_DECL
#error File must be included before boost::asio headers
#endif

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_LOCAL_SOCKETS)

#include <boost/asio/detail/impl/socket_ops.ipp>
#include <boost/asio/detail/consuming_buffers.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>

#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>

#include <stdexcept>

namespace boost {
namespace asio {

class fd_ref
{
    int* fd_;

public:
    fd_ref() : fd_(nullptr) {}
    explicit fd_ref(int &fd) : fd_(&fd) {}
    operator int () const { return *fd_; }
    int* operator & () { return fd_; }
    int const* operator & () const { return fd_; }
};

class fd_cref
{
    int fd_;

public:
    fd_cref() : fd_(-1) {}
    explicit fd_cref(int const &fd) : fd_(fd) {}
    operator int () const { return fd_; }
    int const* operator & () const { return &fd_; }
};

template <typename MutableBufferSequence, typename MutableFDSequence>
struct mutable_buffers_with_attached_file_descriptors
{
    MutableBufferSequence buffers_;
    MutableFDSequence file_descriptors_;
};

template <typename MutableBufferSequence> inline
mutable_buffers_with_attached_file_descriptors< MutableBufferSequence, std::array<fd_ref, 1> >
attach_mutable_file_descriptors( MutableBufferSequence const& buffers, fd_ref fd )
{
    return { buffers, { fd } };
}

template <typename MutableBufferSequence, size_t numFD> inline
mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, std::array<fd_ref, numFD> >
attach_mutable_file_descriptors( MutableBufferSequence const& buffers, std::array<fd_ref, numFD> const& fds )
{
    return { buffers, fds };
}

template <typename MutableBufferSequence, typename MutableFDSequence> inline
mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence>
attach_mutable_file_descriptors( MutableBufferSequence const& buffers, MutableFDSequence const& fds )
{
    return { buffers, fds };
}

template <typename MutableBufferSequence, typename MutableFDSequence>
struct is_mutable_buffer_sequence< mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence> >
    : is_mutable_buffer_sequence< MutableBufferSequence >
{};

template <typename MutableBufferSequence, typename MutableFDSequence>
struct is_const_buffer_sequence< mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence> >
    : is_const_buffer_sequence< MutableBufferSequence >
{};

template <typename MutableBufferSequence, typename MutableFDSequence> inline
auto buffer_sequence_begin( mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence> & c ) -> decltype(buffer_sequence_begin(c.buffers_) )
{
    return buffer_sequence_begin(c.buffers_);
}

template <typename MutableBufferSequence, typename MutableFDSequence> inline
auto buffer_sequence_begin( mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence> const & c ) -> decltype(buffer_sequence_begin(c.buffers_) )
{
    return buffer_sequence_begin(c.buffers_);
}

template <typename MutableBufferSequence, typename MutableFDSequence> inline
auto buffer_sequence_end( mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence> & c ) -> decltype(buffer_sequence_end(c.buffers_) )
{
    return buffer_sequence_end(c.buffers_);
}

template <typename MutableBufferSequence, typename MutableFDSequence> inline
auto buffer_sequence_end( mutable_buffers_with_attached_file_descriptors<MutableBufferSequence, MutableFDSequence> const & c ) -> decltype(buffer_sequence_end(c.buffers_) )
{
    return buffer_sequence_end(c.buffers_);
}

template <typename ConstBufferSequence, typename ConstFDSequence>
struct const_buffers_with_attached_file_descriptors
{
    ConstBufferSequence buffers_;
    ConstFDSequence file_descriptors_;
};

template <typename ConstBufferSequence > inline
const_buffers_with_attached_file_descriptors<ConstBufferSequence, std::array<fd_cref, 1> >
attach_const_file_descriptors( ConstBufferSequence const& buffers, fd_cref fd )
{
    return { buffers, { fd } };
}

template <typename ConstBufferSequence, size_t numFD> inline
const_buffers_with_attached_file_descriptors<ConstBufferSequence, std::array<fd_cref, numFD> >
attach_const_file_descriptors( ConstBufferSequence const& buffers, std::array<fd_cref, numFD> const& fds )
{
    return { buffers, fds };
}

template <typename ConstBufferSequence, typename ConstFDSequence> inline
const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence>
attach_const_file_descriptors( ConstBufferSequence const& buffers, ConstFDSequence const& fds )
{
    return { buffers, fds };
}

template <typename ConstBufferSequence, typename ConstFDSequence>
struct is_mutable_buffer_sequence< const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence> >
    : is_mutable_buffer_sequence< ConstBufferSequence >
{};

template <typename ConstBufferSequence, typename ConstFDSequence>
struct is_const_buffer_sequence< const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence> >
    : is_const_buffer_sequence< ConstBufferSequence >
{};

template <typename ConstBufferSequence, typename ConstFDSequence> inline
auto buffer_sequence_begin( const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence> & c ) -> decltype(buffer_sequence_begin(c.buffers_) )
{
    return buffer_sequence_begin(c.buffers_);
}

template <typename ConstBufferSequence, typename ConstFDSequence> inline
auto buffer_sequence_begin( const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence> const & c ) -> decltype(buffer_sequence_begin(c.buffers_) )
{
    return buffer_sequence_begin(c.buffers_);
}

template <typename ConstBufferSequence, typename ConstFDSequence> inline
auto buffer_sequence_end( const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence> & c ) -> decltype(buffer_sequence_end(c.buffers_) )
{
    return buffer_sequence_end(c.buffers_);
}

template <typename ConstBufferSequence, typename ConstFDSequence> inline
auto buffer_sequence_end( const_buffers_with_attached_file_descriptors<ConstBufferSequence, ConstFDSequence> const & c ) -> decltype(buffer_sequence_end(c.buffers_) )
{
    return buffer_sequence_end(c.buffers_);
}

namespace detail {

static const size_t max_fd_num_per_msg = 64;
typedef socket_ops::buf native_buffer_type;

namespace socket_ops {

inline signed_size_type recvmsg( socket_type s,
    buf* bufs, size_t bufs_count,
    fd_ref fds[], size_t fds_count,
    int flags, int* out_flags,
    socket_addr_type* addr, std::size_t* addrlen,
    boost::system::error_code& ec )
{
    clear_last_error();

    msghdr msg = msghdr();
    msg.msg_iov = bufs;
    msg.msg_iovlen = static_cast<int>(bufs_count);
    if (nullptr != addr)
    {
        init_msghdr_msg_name(msg.msg_name, addr);
        msg.msg_namelen = static_cast<int>(*addrlen);
    }

    std::aligned_storage< CMSG_SPACE(sizeof(int) * max_fd_num_per_msg), alignof(struct cmsghdr)>::type cmsg_buf;
    memset(&cmsg_buf, 0, sizeof(cmsg_buf));
    msg.msg_control = &cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    flags |= MSG_CMSG_CLOEXEC;

    signed_size_type result = error_wrapper(::recvmsg(s, &msg, flags), ec);
    if (nullptr != addrlen)
    {
        *addrlen = msg.msg_namelen;
    }
    if (result < 0)
    {
        if (nullptr != out_flags) *out_flags = 0;
        return result;
    }
    ec = boost::system::error_code();
    if (nullptr != out_flags) *out_flags = msg.msg_flags;

    if((msg.msg_flags & MSG_CTRUNC) == MSG_CTRUNC)
    {
        /* we did not provide enough space for the ancillary element array */
        ec = boost::asio::error::message_size;
        return -1;
    }

    /* iterate ancillary elements */
    for(cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
       nullptr != cmsg;
       cmsg = CMSG_NXTHDR(&msg, cmsg))
    {
        if( (SOL_SOCKET == cmsg->cmsg_level) && (SCM_RIGHTS == cmsg->cmsg_type) )
        {
            size_t fds_count_actual = 0;
            for (uint8_t const *fd_storage = static_cast<uint8_t const*>(CMSG_DATA(cmsg)), *fd_storage_last = fd_storage + cmsg->cmsg_len - CMSG_LEN(sizeof(int));
                 fd_storage <= fd_storage_last;
                 fd_storage += sizeof(int), ++fds_count_actual)
            {
                if (fds_count > fds_count_actual)
                {
                    if ( int* fd = &fds[fds_count_actual] )
                    {
                        memcpy( fd, fd_storage, sizeof(int) );
                        continue;
                    }
                }
                int fd;
                memcpy(&fd, fd_storage, sizeof(int));
                ::close(fd);
            }
            if (fds_count < fds_count_actual)
            {
                ec = boost::asio::error::message_size;
                return -1;
            }
            break;
        }
    }
    return result;
}

inline signed_size_type recvmsg( socket_type s,
    std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int flags, int* out_flags,
    socket_addr_type* addr, std::size_t* addrlen,
    boost::system::error_code& ec )
{
    return recvmsg(s,
         bufs.first, count.first,
         bufs.second, count.second,
         flags, out_flags,
         addr, addrlen,
         ec
    );
}

inline signed_size_type sendmsg( socket_type s,
    const buf* bufs, size_t bufs_count,
    fd_cref fds[], size_t fds_count,
    int flags, 
    socket_addr_type const* addr, std::size_t addrlen,
    boost::system::error_code& ec )
{
    clear_last_error();

    if (fds_count > max_fd_num_per_msg)
    {
        ec = boost::asio::error::message_size;
        return -1;
    }

    msghdr msg = msghdr();
    msg.msg_iov = const_cast<buf*>(bufs);
    msg.msg_iovlen = static_cast<int>(bufs_count);
    if (nullptr != addr)
    {
        init_msghdr_msg_name(msg.msg_name, addr);
        msg.msg_namelen = static_cast<int>(addrlen);
    }

    std::aligned_storage< CMSG_SPACE(sizeof(int) * max_fd_num_per_msg), alignof(struct cmsghdr)>::type cmsg_buf;
    if (0 != fds_count)
    {
        memset(&cmsg_buf, 0, sizeof(cmsg_buf));
        msg.msg_control = &cmsg_buf;
        msg.msg_controllen = sizeof(cmsg_buf);
        struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int) * fds_count);
        /* Initialize the payload: */
        auto fd_storage = reinterpret_cast<int*>(CMSG_DATA(cmsg));
        for (size_t i = 0; i < fds_count; ++ i)
            memcpy(fd_storage + i , &fds[i], sizeof(int) );
        /* Sum of the length of all control messages in the buffer: */
        msg.msg_controllen = cmsg->cmsg_len;
    }
#if defined(__linux__)
    flags |= MSG_NOSIGNAL;
#endif // defined(__linux__)

    signed_size_type result = error_wrapper(::sendmsg(s, &msg, flags), ec);
    if (result >= 0)
    {
        ec = boost::system::error_code();
    }
    return result;
}

inline signed_size_type sendmsg( socket_type s,
    std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, 
    socket_addr_type const* addr, std::size_t addrlen,
    boost::system::error_code& ec )
{
    return sendmsg( s,
        bufs.first, count.first,
        bufs.second, count.second,
        flags,
        addr, addrlen,
        ec
    );
}
    
inline size_t sync_recvmsg(socket_type s, state_type state, std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int in_flags, int* out_flags, socket_addr_type* addr,
    std::size_t* addrlen, bool all_empty, boost::system::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = boost::asio::error::bad_descriptor;
    return 0;
  }

  // A request to read 0 bytes on a stream is a no-op.
  if (all_empty && (state & stream_oriented))
  {
    ec = boost::system::error_code();
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::recvmsg(s, bufs, count, in_flags, out_flags, addr, addrlen, ec);

    // Check if operation succeeded.
    if (bytes > 0)
      return bytes;

    // Check for EOF.
    if ((state & stream_oriented) && bytes == 0)
    {
      ec = boost::asio::error::eof;
      return 0;
    }

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != boost::asio::error::would_block
          && ec != boost::asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_read(s, 0, -1, ec) < 0)
      return 0;
  }
}

inline size_t sync_recv(socket_type s, state_type state, std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int flags, bool all_empty, boost::system::error_code& ec)
{
    return sync_recvmsg( s, state, bufs, count, flags, nullptr, nullptr, nullptr, all_empty, ec );
}

inline size_t sync_recvfrom(socket_type s, state_type state, std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int flags, socket_addr_type* addr, size_t* addrlen, boost::system::error_code& ec)
{
    return sync_recvmsg( s, state, bufs, count, flags, nullptr, addr, addrlen, false, ec );
}

inline size_t sync_recvmsg(socket_type s, state_type state, std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int in_flags, int& out_flags, boost::system::error_code& ec)
{
    return sync_recvmsg( s, state, bufs, count, in_flags, &out_flags, nullptr, nullptr, false, ec );
}

inline size_t sync_sendmsg(socket_type s, state_type state, std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, socket_addr_type const* addr, size_t addrlen, bool all_empty, boost::system::error_code& ec)
{
  if (s == invalid_socket)
  {
    ec = boost::asio::error::bad_descriptor;
    return 0;
  }

  // A request to write 0 bytes to a stream is a no-op.
  if (all_empty && (state & stream_oriented))
  {
    ec = boost::system::error_code();
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    signed_size_type bytes = socket_ops::sendmsg(s, bufs, count, flags, addr, addrlen, ec);

    // Check if operation succeeded.
    if (bytes >= 0)
      return bytes;

    // Operation failed.
    if ((state & user_set_non_blocking)
        || (ec != boost::asio::error::would_block
          && ec != boost::asio::error::try_again))
      return 0;

    // Wait for socket to become ready.
    if (socket_ops::poll_write(s, 0, -1, ec) < 0)
      return 0;
  }
}

inline size_t sync_send(socket_type s, state_type state, std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, bool all_empty, boost::system::error_code& ec)
{
    return sync_sendmsg(s, state, bufs, count, flags, nullptr, 0, all_empty, ec);
}

inline size_t sync_sendto(socket_type s, state_type state, std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, const socket_addr_type* addr, size_t addrlen, boost::system::error_code& ec)
{
    return sync_sendmsg(s, state, bufs, count, flags, addr, addrlen, false, ec);
}

#if !defined(BOOST_ASIO_HAS_IOCP)

inline bool non_blocking_recvmsg(socket_type s,
    std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int in_flags, int* out_flags, socket_addr_type* addr, size_t* addrlen, bool is_stream,
    boost::system::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Read some data.
    signed_size_type bytes = socket_ops::recvmsg(s, bufs, count, in_flags, out_flags, addr, addrlen, ec);

    // Check for end of stream.
    if (is_stream && bytes == 0)
    {
      ec = boost::asio::error::eof;
      return true;
    }

    // Retry operation if interrupted by signal.
    if (ec == boost::asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == boost::asio::error::would_block
        || ec == boost::asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = boost::system::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

inline bool non_blocking_recv(socket_type s,
    std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int flags, bool is_stream,
    boost::system::error_code& ec, size_t& bytes_transferred)
{
    return non_blocking_recvmsg(s, bufs, count, flags, nullptr, nullptr, nullptr, is_stream, ec, bytes_transferred);
}

inline bool non_blocking_recvfrom(socket_type s,
    std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int flags, socket_addr_type* addr, size_t* addrlen, 
    boost::system::error_code& ec, size_t& bytes_transferred)
{
    return non_blocking_recvmsg(s, bufs, count, flags, nullptr, addr, addrlen, false, ec, bytes_transferred);
}

inline bool non_blocking_recvmsg(socket_type s,
    std::pair<buf*, fd_ref*> bufs, std::pair<size_t, size_t> count,
    int in_flags, int& out_flags, 
    boost::system::error_code& ec, size_t& bytes_transferred)
{
    return non_blocking_recvmsg(s, bufs, count, in_flags, &out_flags, nullptr, nullptr, false, ec, bytes_transferred);
}

inline bool non_blocking_sendmsg(socket_type s,
    std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, socket_addr_type const* addr, size_t addrlen,
    boost::system::error_code& ec, size_t& bytes_transferred)
{
  for (;;)
  {
    // Write some data.
    signed_size_type bytes = socket_ops::sendmsg(s, bufs, count, flags, addr, addrlen, ec);

    // Retry operation if interrupted by signal.
    if (ec == boost::asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == boost::asio::error::would_block
        || ec == boost::asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = boost::system::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

inline bool non_blocking_send(socket_type s,
    std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, 
    boost::system::error_code& ec, size_t& bytes_transferred)
{
    return non_blocking_sendmsg(s, bufs, count, flags, nullptr, 0, ec, bytes_transferred);
}

inline bool non_blocking_sendto(socket_type s,
    std::pair<const buf*, fd_cref*> bufs, std::pair<size_t, size_t> count,
    int flags, socket_addr_type const* addr, size_t addrlen,
    boost::system::error_code& ec, size_t& bytes_transferred)
{
    return non_blocking_sendmsg(s, bufs, count, flags, addr, addrlen, ec, bytes_transferred);
}

#endif // !defined(BOOST_ASIO_HAS_IOCP)

} // namespace socket_ops

template <typename Buffers, typename MutableFDSequence>
struct prepared_buffers_max< mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence> >
{
  enum { value = prepared_buffers_max< Buffers>::value };
};

template <typename Buffers, typename ConstFDSequence>
struct prepared_buffers_max< const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence> >
{
  enum { value = prepared_buffers_max< Buffers>::value };
};

template <typename Buffer, typename Buffers, typename MutableFDSequence, typename Buffer_Iterator>
class consuming_buffers< Buffer, mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence>, Buffer_Iterator >
    : public consuming_buffers< Buffer, Buffers, Buffer_Iterator >
{
    using base_t = consuming_buffers< Buffer, Buffers, Buffer_Iterator >;

    boost::optional<MutableFDSequence> fd_;

public:
    consuming_buffers( mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence> const& buf_fd )
        : base_t( buf_fd.buffers_ )
        , fd_( buf_fd.file_descriptors_ )
    {}

    MutableFDSequence const& file_descriptors() const { return *fd_; }

    auto prepare(size_t max_size) -> decltype( attach_mutable_file_descriptors( base_t::prepare(max_size), file_descriptors() ) )
    {
        return attach_mutable_file_descriptors( base_t::prepare(max_size), file_descriptors() );
    }

    void consume(size_t size)
    {
        base_t::consume(size);
        fd_ = boost::in_place();
    }
};

template <typename Buffer, typename Buffers, typename ConstFDSequence, typename Buffer_Iterator>
class consuming_buffers< Buffer, const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence>, Buffer_Iterator >
    : public consuming_buffers< Buffer, Buffers, Buffer_Iterator >
{
    using base_t = consuming_buffers< Buffer, Buffers, Buffer_Iterator >;

    boost::optional<ConstFDSequence> fd_;

public:
    consuming_buffers( const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence> const& buf_fd )
        : consuming_buffers< Buffer, Buffers, Buffer_Iterator >( buf_fd.buffers_ )
        , fd_( buf_fd.file_descriptors_ )
    {}

    ConstFDSequence const& file_descriptors() const { return *fd_; }

    auto prepare(size_t max_size) -> decltype( attach_const_file_descriptors( base_t::prepare(max_size), file_descriptors() ) )
    {
        return attach_const_file_descriptors( base_t::prepare(max_size), file_descriptors() );
    }

    void consume(size_t size)
    {
        base_t::consume(size);
        fd_ = boost::in_place();
    }
};

template <typename Buffer, typename Buffers, typename FDBuffer>
class buffer_sequence_adapter_with_attached_fd_buffers
    : public buffer_sequence_adapter< Buffer, Buffers >
{
    FDBuffer fd_[max_fd_num_per_msg];
    size_t num_fd_;

protected:
    template <typename BufferSequence, typename FDSequence>
    buffer_sequence_adapter_with_attached_fd_buffers( BufferSequence const& buffers, FDSequence const& file_descriptors )
        : buffer_sequence_adapter< Buffer, Buffers >( buffers )
        , num_fd_()
    {
        if ( max_fd_num_per_msg < file_descriptors.size() )
            throw std::length_error("file descriptors sequence is too long to be sent and received in a single message");
        for ( auto& fd : file_descriptors )
            fd_[num_fd_++] = fd;
    }

public:
    std::pair<native_buffer_type*, FDBuffer*> buffers()
    {
        return std::make_pair( buffer_sequence_adapter< Buffer, Buffers >::buffers(), fd_ );
    }

    std::pair<size_t, size_t> count() const
    {
        return std::make_pair( buffer_sequence_adapter< Buffer, Buffers >::count(), num_fd_ );
    }
};

template <typename Buffer, typename Buffers, typename MutableFDSequence>
class buffer_sequence_adapter< Buffer, mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence> >
  : public buffer_sequence_adapter_with_attached_fd_buffers< Buffer, Buffers, fd_ref >
{
    using base_t = buffer_sequence_adapter_with_attached_fd_buffers< Buffer, Buffers, fd_ref >;
public:
    explicit buffer_sequence_adapter( mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence> const& buf_fd )
        : base_t( buf_fd.buffers_, buf_fd.file_descriptors_ )
    {}

    using base_t::all_empty;
    static bool all_empty(const mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence>& buffer_sequence)
    {
        return base_t::all_empty( buffer_sequence.buffers_ );
    }

    static void validate(const mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence>& buffer_sequence)
    {
        return base_t::validate( buffer_sequence.buffers_ );
    }

    static Buffer first(const mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence>& buffer_sequence)
    {
        return base_t::first( buffer_sequence.buffers_ );
    }
};

template <typename Buffer, typename Buffers, typename ConstFDSequence>
class buffer_sequence_adapter< Buffer, const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence> >
  : public buffer_sequence_adapter_with_attached_fd_buffers< Buffer, Buffers, fd_cref >
{
    using base_t = buffer_sequence_adapter_with_attached_fd_buffers< Buffer, Buffers, fd_cref >;
public:
    explicit buffer_sequence_adapter( const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence> const& buf_fd )
        : buffer_sequence_adapter_with_attached_fd_buffers< Buffer, Buffers, fd_cref >( buf_fd.buffers_, buf_fd.file_descriptors_ )
    {}

    using base_t::all_empty;
    static bool all_empty(const const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence>& buffer_sequence)
    {
        return base_t::all_empty( buffer_sequence.buffers_ );
    }

    static void validate(const const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence>& buffer_sequence)
    {
        return base_t::validate( buffer_sequence.buffers_ );
    }

    static Buffer first(const const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence>& buffer_sequence)
    {
        return base_t::first( buffer_sequence.buffers_ );
    }
};

template <typename Buffer, typename Buffers, typename MutableFDSequence, typename Buffer_Iterator>
class buffer_sequence_adapter< Buffer, consuming_buffers< Buffer, mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence>, Buffer_Iterator > >
  : public buffer_sequence_adapter_with_attached_fd_buffers< Buffer, consuming_buffers<Buffer, Buffers, Buffer_Iterator>, fd_ref >
{
public:
    explicit buffer_sequence_adapter( consuming_buffers< Buffer, mutable_buffers_with_attached_file_descriptors<Buffers, MutableFDSequence>, Buffer_Iterator > const& buf_fd )
        : buffer_sequence_adapter_with_attached_fd_buffers< Buffer, consuming_buffers<Buffer, Buffers, Buffer_Iterator>, fd_ref >( buf_fd, buf_fd.file_descriptors() )
    {}
};

template <typename Buffer, typename Buffers, typename ConstFDSequence, typename Buffer_Iterator>
class buffer_sequence_adapter< Buffer, consuming_buffers< Buffer, const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence>, Buffer_Iterator > >
  : public buffer_sequence_adapter_with_attached_fd_buffers< Buffer, consuming_buffers<Buffer, Buffers, Buffer_Iterator>, fd_cref >
{
public:
    explicit buffer_sequence_adapter( consuming_buffers< Buffer, const_buffers_with_attached_file_descriptors<Buffers, ConstFDSequence>, Buffer_Iterator > const& buf_fd )
        : buffer_sequence_adapter_with_attached_fd_buffers< Buffer, consuming_buffers<Buffer, Buffers, Buffer_Iterator>, fd_cref >( buf_fd, buf_fd.file_descriptors() )
    {}
};

} // namespace detail
} // namespace asio
} // namespace boost

#else
#   error Platform does not support UNIX domain sockets
#endif

#endif//_ASIO_UNIX_FD_PASSING_HPP

