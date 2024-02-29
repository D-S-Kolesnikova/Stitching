add_executable(FramesStitcherBin 
    StitchingEngine/StitchingEngine.cpp
    StitchingEngine/StitchingEngineImpl.cpp
    StitchingEngineTester/StitchingTester.cpp
    StitchingEngine/LightGlue.cpp
    StitchingEngine/SuperPoint.cpp)

target_include_directories(FramesStitcherBin PRIVATE
    StitchingEngine
    ${ONNXRUNTIME_INCLUDE_DIRS})

set(ONNXRUNTIME_INCLUDE_DIRS "E:/devel/helper/homography/projects/myProjectOpencv/Superpoint-LightGlue-Image-Stiching/Thirdparty/onnxruntime-win-x64-1.15.1/include")
set(ONNXRUNTIME_LIBRARIES "E:/devel/helper/homography/projects/myProjectOpencv/Superpoint-LightGlue-Image-Stiching/Thirdparty/onnxruntime-win-x64-1.15.1/lib")

target_link_libraries(FramesStitcherBin PRIVATE
    opencv_imgproc
    opencv_core
    opencv_highgui
    opencv_stitching
    opencv_features2d
    opencv_xfeatures2d
    opencv_imgcodecs
    opencv_calib3d
    ${ONNXRUNTIME_LIBRARIES})

install(TARGETS FramesStitcherBin
    EXPORT FramesStitcherTargets
    RUNTIME DESTINATION FramesStitcher/bin)
