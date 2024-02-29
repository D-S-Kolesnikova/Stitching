message(STATUS "Current CMake version: CMAKE_VERSION=${CMAKE_VERSION}")
cmake_minimum_required(VERSION 3.12)

enable_testing()

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Put all output into single directory to ease deployment.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/!bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/!bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/!lib")

# Reset the configurations list to our needs.
set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as no one was specified externally.")
    set(CMAKE_CONFIGURATION_TYPES "${DEFAULT_BUILD_TYPE}" CACHE STRING "" FORCE)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "RelWithDebInfo" "Debug" "Release") # possible values for cmake-gui
endif()

# Helpers to print properties and variables
# cmake_print_properties, cmake_print_variables
include(CMakePrintHelpers)

# where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/ is checked
# We have custom CMake modules in our build tree.
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/Build/cmake"
    "${CMAKE_SOURCE_DIR}/external/build-tools/cmake")
list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_SOURCE_DIR}/Build/cmake")

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/Build/cmake/toolchain.cmake")

include(buildtools.common.helpers)

include(ItvCvCompilerPreset)

project(ComputerVision)

include(ItvCvMacros)
include(ItvCvModules)
include(PrepareSDK)

buildtools_get_git_commit_hash("${CMAKE_SOURCE_DIR}" CV_SOURCES_COMMIT_HASH)

set(CV_BAMBOO_BUILD_KEY $ENV{bamboo_buildKey})
if(NOT CV_BAMBOO_BUILD_KEY)
    set(CV_BAMBOO_BUILD_KEY "local")
endif()

set(CV_BAMBOO_BUILD_NUMBER $ENV{bamboo_buildNumber})
if(NOT CV_BAMBOO_BUILD_NUMBER)
    set(CV_BAMBOO_BUILD_NUMBER 0)
    set(CV_BUILD_ID "${CV_BAMBOO_BUILD_KEY}")
else()
    set(CV_BUILD_ID "${CV_BAMBOO_BUILD_KEY}-${CV_BAMBOO_BUILD_NUMBER}")
endif()

if(NOT CV_TIMESTAMP)
    string(TIMESTAMP CV_TIMESTAMP "%Y%m%dT%H%M%S" UTC)
endif()

if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/Build/ItvCvVersion.h.in ${CMAKE_BINARY_DIR}/Build/ItvCvVersion.h @ONLY)
    configure_file(${CMAKE_SOURCE_DIR}/Build/Version.rc ${CMAKE_BINARY_DIR}/Build/Version.rc COPYONLY)
endif()

# link to pthreds on Linux
if(UNIX)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

# Add headers consisting of common AxxonSoft interfaces ('axxonsoft.sdk' repo):
set(submodule_interfaces_dir "${CMAKE_SOURCE_DIR}/external/axxonsoft.sdk")
file(GLOB submodule_Interfaces_content "${submodule_interfaces_dir}/*")
if(NOT submodule_Interfaces_content)
  message(FATAL_ERROR "You have to checkout Interfaces submodule first (${submodule_interfaces_dir})")
endif()
include_directories("${submodule_interfaces_dir}")

# Boost is required to build all packages.
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_ADDITIONAL_VERSIONS "${detector_boost_version}")
set(BOOST_ROOT "${SDK_HOME}/${detector_boost_install_dir}")

string(REPLACE "_" "." BOOST_VERSION ${detector_boost_version})
find_package(Boost ${BOOST_VERSION} REQUIRED
             COMPONENTS
                    system thread signals filesystem date_time
                    regex locale unit_test_framework program_options
                    coroutine iostreams
                    serialization)
mark_as_advanced(Boost_LIB_DIAGNOSTIC_DEFINITIONS)

if(ITVCV_TARGET_ARCH MATCHES "win64|x86_64")
  find_package(TBB REQUIRED CONFIG)
endif()

if(ITVCV_TARGET_ARCH MATCHES "win64|x86_64|arm64")
  set(CUDA_VERSION_STRING "11.6")
  set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_DIR}")
  set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
  set(CUDA_CADART_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.so")
  set(CUDA_BIN_PATH "${SDK_HOME}/${detector_cuda_install_dir}")
  find_package(CUDA REQUIRED CONFIG)
endif()

find_package(OpenCV REQUIRED)

option(MODULE_ALL "Build all modules.")
option(MODULE_ItvCvUtils "Common utilities" ON)

# used by developers to run and test the libraries
option(BUILD_EXECUTABLES "Build executables")
option(ENABLE_OPENCV_GUI "Enable OpenCV GUI for executables" OFF)
option(CV_DISABLE_CPP_OPTIMIZATION "Disable C++ compiler optimizations" OFF)

option(BUILD_PYBIND11 "Build PyBin11 bindings" OFF)
include(PreparePyBind11)

itvcv_add_module_subdir("ItvCvUtils")

# required by axxonnext-ui
itvcv_add_module_subdir("2dMapping/production/2dMapping")
itvcv_add_module_subdir("Fisheye/FisheyeCalibrate")
itvcv_add_module_subdir("ImmersionMode/ImmersionMode")
itvcv_add_module_subdir("PTZCalibrate/production/MultiCamPositionPredictor")
itvcv_add_module_subdir("PTZCalibrate/production/PTZCalibration2")
itvcv_add_module_subdir("RadarCalibration")

# required by axxonnext-detectorpack
itvcv_add_module_subdir("DegradationDetector")
itvcv_add_module_subdir("cryptoWrapper"          TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("HuaweiAscend"           TARGET_CONFIG_MATCHES x86_64 arm64)
itvcv_add_module_subdir("NetworkInformation"     TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("Benchmark"              TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("BarcodeReader"          TARGET_CONFIG_MATCHES win64 x86_64)
itvcv_add_module_subdir("HpePostProcessing"      TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("InferenceWrapper"       TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("NeuroFilter"            TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("NeuroWrapperForIIT"     TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("MaskPerformer"          TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("SmallFacesDetector"     TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("TNeuroTrackerLib"       TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("EmptyShelvesDetectorLib" TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("PoseAnalysis"           TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("PrivacyShield"          TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("PCLib")
itvcv_add_module_subdir("WaterLevel"             TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("QueueMaskBuilder")
itvcv_add_module_subdir("QueueCounter")
itvcv_add_module_subdir("IIT_DETECTOR_9.2")
itvcv_add_module_subdir("AbandonedObjects"       TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("SemifinishedDetection"  TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("SweetHeartingLib"       TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("StaticImageDetectorLib" TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("SiameseTesterBin"       TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("GlobalTrackerLib"       TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("FloatingObjects"        TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("PolyHomography"         TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)
itvcv_add_module_subdir("FramesStitcher"         TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)

# Not required axxonnext-detectorpack
itvcv_add_module_subdir("Utils/DataSetConverter" TARGET_CONFIG_MATCHES win64 x86_64 armhf arm64)

itvcv_final_tests_preparation(${CV_BUILD_BIN_DIRS})

include(ItvCvGenConfig)
include(ItvCvGenMetaFiles)

include(MakeUserSripts)

make_user_sripts()
