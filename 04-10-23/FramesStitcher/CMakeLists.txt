add_executable(FramesStitcherBin 
    StitchingEngine/StitchingEngine.cpp
    StitchingEngine/StitchingEngineImpl.cpp
    StitchingEngineTester/StitchingTester.cpp
    StitchingEngine/SuperPoint.cpp)

target_include_directories(FramesStitcherBin PRIVATE
    StitchingEngine)

target_link_libraries(FramesStitcherBin PRIVATE
    opencv_imgproc
    opencv_core
    opencv_highgui
    opencv_stitching
    opencv_features2d
    opencv_xfeatures2d
    opencv_imgcodecs
    opencv_calib3d
    Boost::thread
    ItvCvUtils
    NetworkInformation
    InferenceWrapper)

install(TARGETS FramesStitcherBin
    EXPORT FramesStitcherTargets
    RUNTIME DESTINATION FramesStitcher/bin)
