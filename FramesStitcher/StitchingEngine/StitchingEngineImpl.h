#ifndef _STITCHINGENGINEIMPL_H_
#define _STITCHINGENGINEIMPL_H_
#include "StitchingEngine.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class IStitchingWrapper
{
public:
    virtual void getPoints(const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                           const int nPoints, const float* inputPoints, float* outputPoints, stError* outputError, const int nFrame = -1) = 0;
    virtual ~IStitchingWrapper() {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IStitchingWrapper* CreateStitchingEngine(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                                         const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, stError* outputError);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void getPointsStatic(const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                     const int nPoints, const float* inputPoints, float* outputPoints, const float wrapperRatio, stError* outputError);

#endif
