find_package(jsoncpp REQUIRED)
find_package(fmt REQUIRED)

itvcv_add_module(
    NetworkInformation
    PUBLIC_INCLUDE_DIRS include
    PRIVATE_INCLUDE_DIRS src
    SOURCES
        include/NetworkInformation/NetworkInformationLib.h
        include/NetworkInformation/Utils.h
        src/NetworkInformation.cpp
        src/NetworkInformationPool.cpp
        src/NetworkInformationPool.h
        src/MetadataJson.cpp
        src/MetadataJson.h
        src/TagsConstant.h
        src/Utils.cpp

    PUBLIC_DEPS
        PACKAGES
            ItvCvUtils
        MODULES
            ItvCvUtils
    PRIVATE_DEPS
        fmt::fmt
        cryptoWrapper
        Boost::boost
        Boost::filesystem
        Boost::date_time
        Boost::regex
        Boost::thread
        jsoncpp_lib
    DEVBIN_SOURCES
        src/main.cpp
    DEVBIN_MORE_PRIVATE_DEPS
        fmt::fmt
    PYBIND11_SOURCES
        pybind/pybind.cpp
)