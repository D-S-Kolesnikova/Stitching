find_package(ffmpeg)
find_package(fmt)
find_package(breakpad)

itvcv_add_module(
    ItvCvUtils
    PUBLIC_INCLUDE_DIRS include
    PRIVATE_INCLUDE_DIRS src
    SOURCES
        include/ItvCvUtils/ItvCvDefs.h
        include/ItvCvUtils/ItvCvUtils.h
        include/ItvCvUtils/VideoConverter.h
        include/ItvCvUtils/NeuroAnalyticsApi.h
        include/ItvCvUtils/Unicode.h
        include/ItvCvUtils/Uuid.h
        include/ItvCvUtils/Colorspaces.h
        include/ItvCvUtils/CalibrationRods.h
        include/ItvCvUtils/Math.h
        include/ItvCvUtils/GeneralStructuresC.h
        include/ItvCvUtils/Envar.h
        include/ItvCvUtils/Log.h
        include/ItvCvUtils/Logf.h
        include/ItvCvUtils/Cpuid.h
        src/CalibrationRods.cpp
        src/Math.cpp
        include/ItvCvUtils/Pose.h
        include/ItvCvUtils/PoseC.h
        include/ItvCvUtils/PointOperators.h
        include/ItvCvUtils/FpsCounter.h
        include/ItvCvUtils/DynamicThreadPool.h
        include/ItvCvUtils/DynamicThreadPoolImpl.h
        include/ItvCvUtils/asio_unix_fd_passing.hpp
        include/ItvCvUtils/fd_passing_server.hpp
        include/ItvCvUtils/simple_rpc.hpp
        include/ItvCvUtils/DumpCallback.h
        src/Cpuid.cpp
        src/NeuroAnalyticsApi.cpp
        src/VideoConverter.cpp
        src/Unicode.cpp
        src/Uuid.cpp
        src/Envar.cpp
        src/Log.cpp
        src/Pose.cpp
        src/PoseC.cpp
        src/FpsCounter.cpp
        src/DynamicThreadPool.cpp
        src/ThreadName.cpp
        include/ItvCvUtils/ThreadName.h
        src/DumpCallback.cpp
        src/DetectionPostProcessing.cpp
        include/ItvCvUtils/DetectionPostProcessing.h
        include/ItvCvUtils/DescriptorAccumulators.h
        src/DescriptorAccumulators.cpp
    PRIVATE_DEPS
        Boost::boost
        Boost::system
        Boost::filesystem
        Boost::serialization
        Boost::thread
        ${breakpad_LIBRARIES}
        ffmpeg::swscale
        ffmpeg::avutil
        fmt::fmt
    PYBIND11_SOURCES
        pybind/pybind.h
        pybind/pybind.cpp
)

if(WIN32)
    # disable SAFESEH because of ffmpeg
    if(MSVC_VERSION GREATER_EQUAL 1910 AND MSVC_VERSION LESS 1919)
        target_link_options(ItvCvUtils PRIVATE /SAFESEH:NO)
    endif()
    target_link_libraries(ItvCvUtils PRIVATE Rpcrt4.lib)
elseif(UNIX)
    target_link_libraries(ItvCvUtils PRIVATE uuid)
endif()

set_target_properties(ItvCvUtils PROPERTIES OUTPUT_NAME CvUtils)

itvcv_add_module(
    ItvCvUtilsTest
    TEST_OF ItvCvUtils
    PRIVATE_INCLUDE_DIRS
        tests
        src
    SOURCES
        tests/main.cpp
        tests/TestDynamicThreadPool.cpp
        tests/TestSimpleRPC.cpp
    PRIVATE_DEPS
        Boost::thread
        Boost::coroutine
        Boost::serialization
        Boost::filesystem
        fmt::fmt
)

if(WIN32)
    # disable SAFESEH because of boost::context
    # libboost_context-vc141-mt-x32-1_68.lib(jump_i386_ms_pe_masm.obj) : error LNK2026: module unsafe for SAFESEH image.
    if(MSVC_VERSION GREATER_EQUAL 1910 AND MSVC_VERSION LESS 1919)
        target_link_options(ItvCvUtilsTest PRIVATE /SAFESEH:NO)
    endif()
endif()


