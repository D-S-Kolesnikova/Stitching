#ifndef _COMPUTERVISIONCOMMON_H_
#define _COMPUTERVISIONCOMMON_H_

#ifdef _MSC_VER
    #ifdef ITVCV_EXPORTS
    #define ITVCV_API extern "C" __declspec(dllexport)
    #else
    #define ITVCV_API extern "C" __declspec(dllimport)
    #endif
#else
    #ifdef ITVCV_EXPORTS
    #ifdef __GNUC__
    #define ITVCV_API extern "C" __attribute__((visibility("default")))
    #else // __GNUC__
    #define ITVCV_API extern "C"
    #endif // __GNUC__
    #else // CV_EXPORTS
    #define ITVCV_API extern "C"
    #endif // CV_EXPORTS
#endif // _MCS_VER


#include <ItvSdk/include/IErrorService.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum itvcvAnalyzerType
{
    itvcvAnalyzerClassification,
    itvcvAnalyzerSSD,
    itvcvAnalyzerHumanPoseEstimator,
    itvcvAnalyzerMaskSegments
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum itvcvModeType
{
    itvcvModeGPU,
    itvcvModeCPU,
    itvcvModeFPGAMovidius,
    itvcvModeIntelGPU,
    itvcvModeHetero,
    itvcvModeHDDL,
    itvcvModeMULTI,
    itvcvModeBalanced
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum itvcvError
{
    itvcvErrorSuccess, // нет ошибок
    itvcvErrorFrameSize, // неверно переданы размеры кадра
    itvcvErrorSensitivity, // неверно передана чувствительность
    itvcvErrorColorScheme, // неверно передана цветовая схема
    itvcvErrorNMeasure, // неверно передана количество "замеров" для анализа огня
    itvcvErrorFrame, // передан "плохой" кадр
    itvcvErrorFramesNumber, // ошибка в переданном числе кадров, которые обрабатываются единовременно
    itvcvErrorLoadANN, // ошибка при загрузке сети #
    itvcvErrorCaffeInitialization, // caffe не удалось инициализироваться
    itvcvErrorNoCapableGPUDevice, // не найдено ни одно подходящего GPU ресурса
    itvcvErrorInvalidGPUNumber, // указан неверно номер используемого GPU
    itvcvErrorInconsistentGPUNumber, // указан номер GPU, не соответсвующего заявленным требованиям
    itvcvErrorCUDADeviceCount, // ошибка при попытке получить информацию об устройстве GPU
    itvcvErrorCUDASetDevice, // ошибка при установке опеделенного устройства GPU
    itvcvErrorNoCapableMovidiusDevice, // не найдено ни одно подходящего Movidius ресурса
    itvcvErrorInvalidMovidiusNumber, // указан неверно номер используемого Movidius устройства
    itvcvErrorMovidiusGetDeviceName, // ошибка при получении имени устройства Movidius
    itvcvErrorMovidiusOpenDevice, // ошибка при открытии устройства Movidius
    itvcvErrorMovidiusCloseDevice, // ошибка при закрытии устройства Movidius
    itvcvErrorMovidiusCreateGraph, // ошибка при создании графа для Movidius
    itvcvErrorMovidiusDeallocateGraph, // ошибка при удалении графа для Movidius
    itvcvErrorMovidiusLoadTensor, // ошибка при загрузке тензора для Movidius
    itvcvErrorMovidiusGetResult, // ошибка при получении результата от Movidius
    itvcvErrorOther, // другие возникшие ошибки
    itvcvErrorChoosingGPU,
    itvcvErrorLabelsValues,
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! уровень логирования
enum itvcvLogSeverity
{
    ITV8::LOG_ERROR,
    ITV8::LOG_WARNING,
    itvcvLogInfo,
    itvcvLogDebug
};

//! Описание типа callBack-a для логгирования
typedef void(*itvcvLogCallback_t) (void* userData, const char* str, itvcvLogSeverity severity);

#endif
