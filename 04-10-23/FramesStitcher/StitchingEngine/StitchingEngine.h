#ifndef _STITCHINGENGINE_DEF_
#define _STITCHINGENGINE_DEF_

#ifdef STITCH_EXPORTS
#define STITCH_API extern "C" __declspec(dllexport)
#else
#define STITCH_API extern "C" __declspec(dllimport)
#endif

#include <utility>
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <fstream>

////////////////////////////////////////////////////////////////////////
enum stError
{
    ESTNoError,
    ESTBadInputParams,
    ESTCalibrationError,
    ESTCamReconstructionError,
    ESTGetPointsError
};

////////////////////////////////////////////////////////////////////////
enum stWarpType
{
    stWarpPlane,
    stWarpCylindrical,
    stWarpSpherical,
    stWarpFisheye,
    stWarpStereographic,
    srWarpCompressedPlaneA2B1,
    srWarpCompressedPlaneA15B1,
    srWarpCompressedPlanePortraitA2B1,
    srWarpCompressedPlanePortraitA15B1,
    srWarpPaniniA2B1,
    srWarpPaniniA15B1,
    srWarpPaniniPortraitA2B1,
    srWarpPaniniPortraitA15B1,
    srWarpMercator,
    srWarpTransverseMercator
};

////////////////////////////////////////////////////////////////////////
// Panorama matrix construction function.
// The input is supplied with n-frames from cameras with known dimensions and stride
// arranged horizontally in the same order (either from left to right or from right to left))
// At the output we get a matrix of camera parameters and rotation, the size of 3x3
// nFrames - number of stitched frames
// direction - direction of stitching: 0 - horizontal, 1 - vertical
// widths, heights, strides - widths, heights and strides of transmitted frames, respectively
// colorSchemes - color scheme gray=8(8UC1),  RGB=16(8UC3)
// frameDataBGR - BGR frames
// outputIntinsicCameraMatrix - camera intrinsic matrix
// outputRotationMatrix - rotation matrix
// netPath - path to LightGlue net
STITCH_API void* createStitching(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                                 const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, const char* netPath, stError* outputError);


////////////////////////////////////////////////////////////////////////
// The function returns the conversion of the source points to panoramic ones from the previously created panorama engine
// stObject - previously created with createStitching panorama object
// stWarpType - warping type
// inputIntinsicCameraMatrix - camera intrinsic matrix
// inputRotationMatrix -  rotation matrix
// nPoints - the number of points from the original image
// inputPoints - points from the original image in the format x1,y1,x2,y2,..,xn,yn, 2*nPoints points
// outputPoints - corresponding points on the panorama
// nFrame - the number of the image for which the matrices should be taken, if -1, then the transmitted ones are used
// inputIntinsicCameraMatrix and inputRotationMatrix
// ratio - this parameter is needed to call a function with  stObject = 0.
//         if stObject non-zero, then this parameter is not used
STITCH_API void getStitchingPoints(void* stObject, const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                                   const int nPoints, const float* inputPoints, float* outputPoints, stError* outputError, const float ratio, const int nFrame = -1);


////////////////////////////////////////////////////////////////////////
// Removing the engine
STITCH_API void destroyStitching(void* stObject);

#endif //_STITCHINGENGINE_DEF_