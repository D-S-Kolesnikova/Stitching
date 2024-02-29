#ifndef _INFERENCEWRAPPERLIB_H_
#define _INFERENCEWRAPPERLIB_H_

#include "ItvCvUtils/NeuroAnalyticsApi.h"
#include "ItvCvUtils/ItvCvDefs.h"

#include "InferenceWrapperLibCommon.h"

#include <ItvSdk/include/IErrorService.h>

#ifdef ITVCV_INFERENCEWRAPPER_EXPORT
#define IW_API_C extern "C" ITVCV_API_EXPORT
#else
#define IW_API_C extern "C" ITVCV_API_IMPORT
#endif

// ##########################################
// >>>>>>>>>>>>>>>>>> CREATION AND DELETION
// ##########################################
/**
 * @brief ������� ������ ���������
 *
 * @param error ��� ������ ���������� �������
 * @param logger ��������� ������
 * @param netType ��� ����
 * @param mode ������ ��������� ������: ����, GPU, CPU, Movidius
 * @param colorScheme �������� �����, ������ �������������� �����=8 (8UC1) � RGB=16(8UC3)
 * @param nClasses ���������� �������, ����� ������� �������������� �������������
 * @param gpuDeviceNumToUse ����� ������������� GPU
 * @param modelFilePath ������ ���� � deploy ����� ����
 * @param weightsFilePath ������ ���� � caffemodel ����� ����
 * @param pluginDir ���������� �������
 * @param heightForMaskEngine
 * @param widthForMaskEngine
 * @return void*
 */
IW_API_C void* iwCreate(
    itvcvError* error,
    ITV8::ILogger* logger,
    itvcvAnalyzerType netType,
    itvcvModeType mode,
    const int colorScheme,
    const int nClasses,
    const int gpuDeviceNumToUse,
    const char* modelFilePath,
    const char* weightsFilePath,
    const char* pluginDir = "",
    bool int8 = false);

IW_API_C void iwCreateAsync(
    void* userData,
    itvcvCreationCallback_t creationCallback,
    itvcvError* error,
    ITV8::ILogger* logger,
    itvcvAnalyzerType netType,
    itvcvModeType mode,
    const int colorScheme,
    const int nClasses,
    const int gpuDeviceNumToUse,
    const char* modelFilePath,
    const char* weightsFilePath,
    const char* pluginDir = "",
    bool int8 = false);

// ������� ������� ��������� ������
IW_API_C void iwDestroy(void* iwObject);

// ##########################################
// >>>>>>>>>>>>>>>>>> CLASSIFICATION
// ##########################################
// ������� �������� � ����� ����������� ���������� ������ ������� ProcessFrame, ���������� ������ ���, ����� ���� ������������� ����������� �������
// iwObject - ������, �������� ���������� ����
// error - � �� ����������� ������
// nFrames - ���������� ������������ ������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
// results - ���� �������� ����������
IW_API_C void iwProcessFrames(
    void* iwObject,
    itvcvError* error,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB,
    float** const results);

// ������� �������� � ����� ����������� ���������� ������ ������� ProcessFrame, ���������� ������ ���, ����� ���� ������������� ����������� �������
// iwObject - ������, �������� ���������� ����
// error - � �� ����������� ������
// userData - ������ �� ������������. ��� �������� �������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
// resultCallBack - callback, ��������� ����� ��������� ����������
IW_API_C void iwAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB);

// ##########################################
// >>>>>>>>>>>>>>>>>> SSD
// ##########################################
// ������� �������� � ����� ����������� ���������� ������ ������� ProcessFrame, ���������� ������ ���, ����� ���� ������������� ����������� �������
// iwObject - ������, �������� ���������� ����
// error - � �� ����������� ������
// uuid - ���������� ����
// nFrames - ���������� ������������ ������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
// nResults - �����������  ����������� � ���� (���� ����� �������� �� 1-��� ����� � ����)
// results - ���� �������� ����������
//
// ����� ������� ������� ������
// ������ ��� results=0 ��� ��������� ���������� �����������
// ������ ��� ����� ������� ������ ����������� ����������� � nResults � ���������� �������� ������ � �������� ��������� � results
IW_API_C void iwssdProcessFrames(
    void* iwObject,
    itvcvError* error,
    const char* uuid,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB,
    int* nResults,
    float*** const results);

// ������� �������� � ����� ����������� ���������� ������ ������� ProcessSubFrame, ���������� ������ ���, ����� ���� ������������� ����������� �������
// iwObject - ������, �������� ���������� ����
// error - � �� ����������� ������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
// windowW, windowH - ������, ������ ����
// windowStepW, windowStepH - ��� ���� �� ������, ������
// nResults - ����������  ����������� � ���� (���� ����� �������� �� 1-��� ����� � ����)
// results - ���� �������� ����������
//
// ����� �������� ������� ������
// ������ ��� results=0 ��� ��������� ���������� �����������
// ������ ��� ����� ������� ������ ���������� ����������� � nResults � ���������� �������� ������ � �������� ��������� � results
IW_API_C void iwssdProcessSubFrame(
    void* iwObject,
    itvcvError* error,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB,
    const int windowW,
    const int windowH,
    const int windowStepW,
    const int windowStepH,
    int* nResults,
    float*** const results);

// ������� �������� � ����� ����������� ���������� ������ ������� ProcessFrame, ���������� ������ ���, ����� ���� ������������� ����������� �������
// iwObject - ������, �������� ���������� ����
// error - � �� ����������� ������
// userData - ������ �� ������������. ��� �������� �������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
// resultCallBack - callback, ��������� ����� ��������� ����������
IW_API_C void iwssdAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwSSDResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB);

// ##########################################
// >>>>>>>>>>>>>>>>>> SIEMESE
// ##########################################
IW_API_C void iwsiameseProcessFrames(
    void* iwObject,
    itvcvError* error,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataBGR,
    float** const results);

IW_API_C void iwsiameseProcessSameFrame(
    void* iwObject,
    itvcvError* error,
    const int nRectangles,
    const int *xs,
    const int *ys,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataBGR,
    float** const results);

IW_API_C float iwsiameseCompareVectors(
    float* const vector1,
    float* const vector2,
    const int vectorSize);

// ##########################################
// >>>>>>>>>>>>>>>>>> HPE
// ##########################################
// ������� �������� � ����� ����������� ���������� ������ ��������� ������, ����� �������� ������ ����� ����������  ������� ihpeProcessFrame
// ������ ��� ��� ��������� ������ � OutHitmap=NULL �  outPafs=NULL
// ������ ��� ��� ��������� �����������
// pafsDims  hitmapDims- ���������� ���������(dimensions) ��� hitmap � PAFS � ����� (��� ���� ���������� ����� �������� ������� ������ ������� ������(nFrame) ���� �������)
// iwObject -������, �������� ���������� ����
//  OutHitmap,  outPafs - �������������� ��������� ������ ���� ����� �������� ������ � ������ ���������� �������� ����� �����  ������������ pafsDims ��� OutHitmap  � ������������ hitmapDims ��� outPafs
IW_API_C void ihpeGetResult(
    void* iwObject,
    itvcvError* error,
    const char* uuid,
    int* countsPersonPose,
    ItvCvUtils::PoseC** result);

// ������� �������� � ����� ����������� ���������� ������ ������� ProcessFrame, ���������� ������ ���, ����� ���� �������� Hitmap � PAFS
// iwObject - ������, �������� ���������� ����
// nFrames - ���������� ������������ ������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
IW_API_C void ihpeProcessFrame(
    void* iwObject,
    itvcvError* error,
    const char* uuid,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB);

// ������� �������� � ����� ����������� ���������� ������ ������� ProcessFrame, ���������� ������ ���, ����� ���� �������� Hitmap � PAFS
// iwObject - ������, �������� ���������� ����
// nFrames - ���������� ������������ ������
// width, height, stride - ������, ������ � ������ ������������ ������
// frameDataRGB - �����, � ������ ������ ����������� ���� � ������������ �������� ������ - ���������� RBG (rbgrbgrbg....)
IW_API_C void ihpeAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwHPEResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB);

IW_API_C void iwmaskAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwMASKResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB);

// ##########################################
// >>>>>>>>>>>>>>>>>> STATISTICS
// ##########################################

IW_API_C itvcvError iwGetInputGeometry(void* iwObject, ITV8::Size*);
IW_API_C void iwTakeStats(void* iwObject, ITV8::uint32_t period_ms);

#endif
