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
 * @brief создает движок синхронно
 *
 * @param error код ошибки выполнения функции
 * @param logger глобалный логгер
 * @param netType тип сети
 * @param mode способ обработки кадров: АВТО, GPU, CPU, Movidius
 * @param colorScheme цветовая схема, сейчас поддерживается серая=8 (8UC1) и RGB=16(8UC3)
 * @param nClasses количество классов, среди которых осуществляется классификация
 * @param gpuDeviceNumToUse номер используемого GPU
 * @param modelFilePath полный путь к deploy файлу сети
 * @param weightsFilePath полный путь к caffemodel файлу сети
 * @param pluginDir директория плагина
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

// Функция удаляет созданный движок
IW_API_C void iwDestroy(void* iwObject);

// ##########################################
// >>>>>>>>>>>>>>>>>> CLASSIFICATION
// ##########################################
// Функция вызывает у ранее полученного экземпляра движка функцию ProcessFrame, вызывается каждый раз, когда надо анализировать оставленный предмет
// iwObject - движок, которому передается кадр
// error - в неё записвается ошибка
// nFrames - количество передаваемых кадров
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
// results - куда записать результаты
IW_API_C void iwProcessFrames(
    void* iwObject,
    itvcvError* error,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB,
    float** const results);

// Функция вызывает у ранее полученного экземпляра движка функцию ProcessFrame, вызывается каждый раз, когда надо анализировать оставленный предмет
// iwObject - движок, которому передается кадр
// error - в неё записвается ошибка
// userData - данные от пользователя. для передаче обратно
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
// resultCallBack - callback, вызвается после получения результата
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
// Функция вызывает у ранее полученного экземпляра движка функцию ProcessFrame, вызывается каждый раз, когда надо анализировать оставленный предмет
// iwObject - движок, которому передается кадр
// error - в неё записвается ошибка
// uuid - определает кадр
// nFrames - количество передаваемых кадров
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
// nResults - кольичества  результатов в баче (пока нужно передать по 1-ому кадру в баче)
// results - куда записать результаты
//
// нужно вызвать функцию дважде
// первый раз results=0 для получения количества результатов
// второй раз можно указать нужные кольичество результатов в nResults и необходимо выделять память и передать указатель в results
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

// Функция вызывает у ранее полученного экземпляра движка функцию ProcessSubFrame, вызывается каждый раз, когда надо анализировать оставленный предмет
// iwObject - движок, которому передается кадр
// error - в неё записвается ошибка
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
// windowW, windowH - ширина, высота окна
// windowStepW, windowStepH - шаг окна по ширине, высоте
// nResults - количество  результатов в баче (пока нужно передать по 1-ому кадру в баче)
// results - куда записать результаты
//
// нужно вызывать функцию дважды
// первый раз results=0 для получения количества результатов
// второй раз можно указать нужное количество результатов в nResults и необходимо выделять память и передать указатель в results
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

// Функция вызывает у ранее полученного экземпляра движка функцию ProcessFrame, вызывается каждый раз, когда надо анализировать оставленный предмет
// iwObject - движок, которому передается кадр
// error - в неё записвается ошибка
// userData - данные от пользователя. для передаче обратно
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
// resultCallBack - callback, вызвается после получения результата
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
// Функция получает у ранее полученного экземпляра движка результат работы, нужно вызывать дважды после выполнение  функции ihpeProcessFrame
// первый раз для выделения памяти с OutHitmap=NULL и  outPafs=NULL
// второй раз для получения результатов
// pafsDims  hitmapDims- количество измерений(dimensions) для hitmap и PAFS в кадре (для этой переменной нужно выделить столько памяти сколько кадров(nFrame) было поданно)
// iwObject -движок, которому передается кадр
//  OutHitmap,  outPafs - необработанный результат работы сети нужно выделить памяти в начале количество поданных даров потом  произведениу pafsDims для OutHitmap  и произведениу hitmapDims для outPafs
IW_API_C void ihpeGetResult(
    void* iwObject,
    itvcvError* error,
    const char* uuid,
    int* countsPersonPose,
    ItvCvUtils::PoseC** result);

// Функция вызывает у ранее полученного экземпляра движка функцию ProcessFrame, вызывается каждый раз, когда надо получить Hitmap и PAFS
// iwObject - движок, которому передается кадр
// nFrames - количество передаваемых кадров
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
IW_API_C void ihpeProcessFrame(
    void* iwObject,
    itvcvError* error,
    const char* uuid,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB);

// Функция вызывает у ранее полученного экземпляра движка функцию ProcessFrame, вызывается каждый раз, когда надо получить Hitmap и PAFS
// iwObject - движок, которому передается кадр
// nFrames - количество передаваемых кадров
// width, height, stride - ширина, высота и страйд передаваемых кадров
// frameDataRGB - кадры, в данный момент принимается кадр с единственной цветовой схемой - упакованый RBG (rbgrbgrbg....)
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
