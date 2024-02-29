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
    ESTNoError,                 // ������ ���
    ESTBadInputParams,          // ������ �� ������� ����������
    ESTCalibrationError,        // ������ � �������� ����������
    ESTCamReconstructionError,  // ������ � �������� ������������� ���������� ���������� ������
    ESTGetPointsError           // ������ ��� ������ �������������� �����
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
// ������� ���������� ������� ��������.
// �� ���� �������� n-������ � ����� � ���������� ��������� � stride-��
// ���������� �� ����������� � ����� ������� (���� ����� �������, ���� ������ ������)
// �� ������ �������� ������� ���������� ������ � ��������, ������ ������ 3�3
// nFrames - ����� ��������� ������
// direction - ����������� ������: 0 - ��������������, 1 - ������������
// widths, heights, strides - ������, ������ � ������� ������������ ������, �������������
// colorSchemes - �������� �����, ������ �������������� �����=8 (8UC1) � 16 ������ RGB=16(8UC3)
// frameDataBGR - ���� �����, ����������� � BGR �������
// outputIntinsicCameraMatrix - �������������� ������� ���������� ���������� ������
// outputRotationMatrix - �������������� ������� ���������
STITCH_API void* createStitching(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                                 const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, stError* outputError);


////////////////////////////////////////////////////////////////////////
// ������� ���������� �������������� �������� ����� � ���������� � ����� ���������� ������ ��������
// stObject - ����� ��������� � ������� createStitching ������ ��������
// stWarpType - ��� warping-a
// inputIntinsicCameraMatrix - ������� ���������� ���������� ������
// inputRotationMatrix - ������� �������� 
// nPoints - ����� ����� � �������� �����������
// inputPoints - ����� � ��������� ����������� � ������� x1,y1,x2,y2,..,xn,yn, ����� 2*nPoints �����
// outputPoints - ��������������� ����� �� ��������
// nFrame - ����� ��������, ��� ������� ������� ����� �������, ���� ��� ������� ����� -1, �� ������������ ���������� 
// inputIntinsicCameraMatrix � inputRotationMatrix
// ratio - ���� �������� ����� ��� ������ ������� � stObject = 0. 
//         ���� stObject ���������, �� ���� �������� �� ������������
STITCH_API void getStitchingPoints(void* stObject, const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                                   const int nPoints, const float* inputPoints, float* outputPoints, stError* outputError, const float ratio, const int nFrame = -1);


////////////////////////////////////////////////////////////////////////
// �������� ������
STITCH_API void destroyStitching(void* stObject);

#endif //_STITCHINGENGINE_DEF_