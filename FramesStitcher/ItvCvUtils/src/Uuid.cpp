#include <ItvCvUtils/Uuid.h>
#include <boost/uuid/uuid_io.hpp>
#include <algorithm>
#include <sstream>

extern "C"
{
#if defined(_WIN32)
#include <Rpc.h>
#elif defined(__linux__)
#include <uuid/uuid.h>
#include <type_traits>
#else
#error "Cannot generate GUID: Unknown system"
#endif
}

namespace ItvCv {
namespace Utils {

    boost::uuids::uuid GenerateUuid()
    {
        boost::uuids::uuid uuid{}; // zero initialized

#if defined(_WIN32)

        // We don't care about UuidCreate success.
        // If there was critical error we will return nil uuid.
        // If there was warning like RPC_S_UUID_LOCAL_ONLY it's ok.
        ::UuidCreate((UUID*)uuid.data);

        // To convert Microsoft's GUID format to RFC 4122 UUID we shoud
        // change endianness of first 3 fields from native to big.
        std::reverse(&uuid.data[0], &uuid.data[4]);
        std::reverse(&uuid.data[4], &uuid.data[6]);
        std::reverse(&uuid.data[6], &uuid.data[8]);

#elif defined(__linux__)

        // ISO C++ forbids casting to an array type uuid_t {aka unsigned char [16]}
        using uuid_pointer_type =
            std::add_pointer<
                std::remove_extent<uuid_t>::type
            >::type;

        ::uuid_generate_random((uuid_pointer_type)uuid.data);

#endif

        return uuid;
    }

    std::string ConvertUuidToString(boost::uuids::uuid const& u)
    {
        std::stringstream ss;
        ss << u;
        return ss.str();
    }

}
}
