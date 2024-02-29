#include "StitchingEngine.h"
#include "StitchingEngineImpl.h"
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::detail;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void* createStitching(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                                 const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, stError* outputError)
{
    return CreateStitchingEngine(nFrames, direction, widths, heights, strides, colorSchemes, frameDataBGR, outputIntinsicCameraMatrix, outputRotationMatrix, ratio, outputError);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void getStitchingPoints(void* stObject, const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                                   const int nPoints, const float* inputPoints, float* outputPoints, stError* outputError, const float ratio, const int nFrame)
{
    if (0 == stObject) return getPointsStatic(type, inputIntinsicCameraMatrix, inputRotationMatrix, nPoints, inputPoints, outputPoints, ratio, outputError);
    IStitchingWrapper* stitchingWrapper = static_cast<IStitchingWrapper*>(stObject);
    return stitchingWrapper->getPoints(type, inputIntinsicCameraMatrix, inputRotationMatrix, nPoints, inputPoints, outputPoints, outputError, nFrame);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void destroyStitching(void* stObject)
{
    IStitchingWrapper* stitchingWrapper = static_cast<IStitchingWrapper*>(stObject);
    delete stitchingWrapper;
}
