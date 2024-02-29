find_package(fmt REQUIRED)

set(USE_TENSORRT_BACKEND OFF)
set(USE_INFERENCE_ENGINE_BACKEND OFF)

if(OpenVINO_DIR)
  set(USE_INFERENCE_ENGINE_BACKEND ON)
endif()

if(TensorRT_DIR)
  set(USE_TENSORRT_BACKEND ON)
endif()

option(USE_ATLAS300_BACKEND "Use Huawei Atlas300 backend" ${MODULE_HuaweiAscend_SUPPORTED})
option(USE_INFERENCE_ENGINE_BACKEND "Use Intel's Inference Engine backend" ${USE_INFERENCE_ENGINE_BACKEND})
option(USE_TENSORRT_BACKEND "Use Nvidia's TensorRT backend" ${USE_TENSORRT_BACKEND})

set(InferenceWrapper_PRIVATE_LIB_DEPS
  fmt::fmt
  HpePostProcessing
  NetworkInformation
  cryptoWrapper
  opencv_imgproc
  opencv_dnn
  Boost::boost
  Boost::filesystem
  Boost::thread
  Boost::date_time
  Boost::regex
  stb
  )

if (NOT(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4))
    find_package(stb REQUIRED CONFIG)
    list(APPEND InferenceWrapper_PRIVATE_LIB_DEPS stb)
endif()

if(UNIX)
  list(APPEND InferenceWrapper_PRIVATE_LIB_DEPS dl)
endif()

set(InferenceWrapper_PUBLIC_LIB_DEPS
  ItvCvUtils
  NetworkInformation
  opencv_core)

set(InferenceWrapper_PUBLIC_PACKAGE_DEPS
  ItvCvUtils
  NetworkInformation
  OpenCV)

set(InferenceWrapper_SRCS
  src/AnalyzerFactory.h
  src/AnalyzerPool.cpp
  src/AnalyzerPool.h
  src/IAnalyzer.h
  src/InferenceHelperFunctions.cpp
  src/InferenceHelperFunctions.h
  src/InferenceWrapperLib.cpp
  src/Channel.cpp
  src/Channel.h
  src/GenericChannelState.cpp
  src/GenericChannelState.h
  include/InferenceWrapper/InferenceWrapperLibCommon.h
  include/InferenceWrapper/InferenceWrapperLib.h
  include/InferenceWrapper/InferenceEngine.h
  include/InferenceWrapper/InferenceChannelParams.h
  include/InferenceWrapper/InferenceEngineParams.h
  include/InferenceWrapper/AnalyzerTraits.h)

if(USE_INFERENCE_ENGINE_BACKEND)
  find_package(OpenVINO REQUIRED)
  find_package(TBB REQUIRED CONFIG)
  list(APPEND InferenceWrapper_PUBLIC_PACKAGE_DEPS OpenVINO)
  list(APPEND InferenceWrapper_PRIVATE_LIB_DEPS openvino::frontend::onnx openvino::runtime TBB::tbb)
  list(APPEND InferenceWrapper_SRCS
    src/AnalyzerIntelIE.cpp
    src/AnalyzerIntelIE.h)
endif()

if(USE_TENSORRT_BACKEND)
  find_package(CUDA REQUIRED CONFIG)
  find_package(cuDNN REQUIRED CONFIG)
  find_package(TensorRT REQUIRED)
  list(APPEND InferenceWrapper_PRIVATE_LIB_DEPS
    TensorRT::nvinfer
    TensorRT::nvparsers
    TensorRT::nvonnxparser
    TensorRT::nvinfer_plugin
    cuDNN::cudnn
    CUDA::runtime)
  if(ITVCV_TARGET_ARCH MATCHES "win64|x86_64")
    list(APPEND InferenceWrapper_PRIVATE_LIB_DEPS
      CUDA::nppc
      CUDA::nppidei
      opencv_cudaimgproc
      opencv_cudawarping)
  endif()
  list(APPEND InferenceWrapper_SRCS
    src/AnalyzerTensorRT.cpp
    src/AnalyzerTensorRT.h
    src/TensorRT/ReusableRequest.cpp
    src/TensorRT/ReusableRequest.h
    src/TensorRT/CacheHelpers.cpp
    src/TensorRT/CacheHelpers.h
    src/TensorRT/Int8Calibrator.cpp
    src/TensorRT/Int8Calibrator.h
    src/TensorRT/Logger.cpp
    src/TensorRT/Logger.h
    src/TensorRT/ErrorRecorder.cpp
    src/TensorRT/ErrorRecorder.h
    src/TensorRT/Buffers.cpp
    src/TensorRT/Buffers.h)
  list(APPEND InferenceWrapper_PUBLIC_PACKAGE_DEPS TensorRT CUDA)
endif()

if(USE_ATLAS300_BACKEND)
  list(APPEND InferenceWrapper_PRIVATE_LIB_DEPS HuaweiAscend)
  list(APPEND InferenceWrapper_PUBLIC_PACKAGE_DEPS HuaweiAscend)
  list(APPEND InferenceWrapper_SRCS
    src/AnalyzerAtlas300.cpp
    src/AnalyzerAtlas300.h)
endif()


itvcv_add_module(
  InferenceWrapper
  PUBLIC_INCLUDE_DIRS include
  PRIVATE_INCLUDE_DIRS src
  SOURCES
    ${InferenceWrapper_SRCS}
  PUBLIC_DEPS
    PACKAGES
      ${InferenceWrapper_PUBLIC_PACKAGE_DEPS}
    MODULES
      ${InferenceWrapper_PUBLIC_LIB_DEPS}
  PRIVATE_DEPS
    ${InferenceWrapper_PRIVATE_LIB_DEPS}
  DEVBIN_SOURCES
    src/main.cpp
  DEVBIN_MORE_PRIVATE_DEPS
    opencv_highgui
    Boost::system
    Boost::program_options
  PYBIND11_MORE_PRIVATE_DEPS
    opencv_imgproc
    opencv_imgcodecs
  PYBIND11_SOURCES
    pybind/pybind.cpp
)

foreach(BACKEND IN ITEMS USE_INFERENCE_ENGINE_BACKEND USE_TENSORRT_BACKEND USE_ATLAS300_BACKEND)
  if(${BACKEND})
    target_compile_definitions(InferenceWrapper PRIVATE "${BACKEND}=1")
  endif()
endforeach()

if(ENABLE_OPENCV_GUI AND BUILD_EXECUTABLES)
  target_compile_definitions(InferenceWrapperBin PRIVATE "ENABLE_OPENCV_GUI=1")
endif()

if(USE_TENSORRT_BACKEND AND (ITVCV_TARGET_ARCH MATCHES "win64|x86_64"))
  add_executable(NeuroAnalyticsGpuCacheGenerator src/NeuroAnalyticsGpuCacheGenerator.cpp)
  if(TBB_FOUND AND (ITVCV_TARGET_ARCH STREQUAL "x86_64"))
    target_link_libraries(NeuroAnalyticsGpuCacheGenerator PRIVATE TBB::tbb)
  endif()
  target_link_libraries(NeuroAnalyticsGpuCacheGenerator PRIVATE
    NetworkInformation
    InferenceWrapper
    ItvCvUtils
    fmt::fmt
    Boost::boost
    Boost::program_options
    Boost::filesystem
    Boost::system
    Boost::thread)
  install(TARGETS NeuroAnalyticsGpuCacheGenerator
    EXPORT NeuroAnalyticsGpuCacheGeneratorTargets
    RUNTIME DESTINATION InferenceWrapper/bin)
endif()
