#ifndef _NEUROANALYTICSAPI_H_
#define _NEUROANALYTICSAPI_H_

#include <ItvCvUtils/ItvCvUtils.h>

#include <ItvSdk/include/IErrorService.h>
#include <memory>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ClassReportType - report type for network
//Report - for ssd object reporting to server/ for classification is negative class
//Skip - class must be skipped and must be in calculations(is nose class)
//NotReport - for ssd object not reporting to server/ for classification is positive class
enum ClassReportType
{
    itvcvNeedReport,
    itvcvNeedSkip,
    itvcvNeedNotReport
};
namespace Pose
{
enum HumanPosePointType
{
    humanPointUnknown = -1,
    humanPointNose = 0,
    humanPointNeck = 1,
    humanPointRightShoulder = 2,
    humanPointRightElbow = 3,
    humanPointRightWrist = 4,
    humanPointLeftShoulder = 5,
    humanPointLeftElbow = 6,
    humanPointLeftWrist = 7,
    humanPointRightHip = 8,
    humanPointRightKnee = 9,
    humanPointRightAnkle = 10,
    humanPointLeftHip = 11,
    humanPointLeftKnee = 12,
    humanPointLeftAnkle = 13,
    humanPointRightEye = 14,
    humanPointLeftEye = 15,
    humanPointRightEar = 16,
    humanPointLeftEar = 17
};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace Segments
{
enum BodySegment
{
    Unknown = 0,
    Body = 1,
    Head = 2,
    Hips = 3,
    Shin = 4,
    Foot = 5,
    Hand = 6,
    Shoulder = 7,
    Forearm = 8,
    All = 9,
    Face = 10
};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum ObjectType
{
    itvcvObjectNoise = 0,
    itvcvObjectHuman = 1,
    itvcvObjectGroupOfHumans = 2,
    itvcvObjectVehicle = 3,
    itvcvObjectReserved = 4, // Legacy. It was used by GUI as unattended
    itvcvObjectFace = 5,
    itvcvObjectAnimal = 6,
    itvcvObjectRobotDog = 7,
    itvcvObjectSmoke = 8,
    itvcvObjectFire = 9,
    itvcvObjectCar = 10,
    itvcvObjectMotorcycle = 11,
    itvcvObjectBus = 12,
    itvcvObjectBicyclist = 13,
    itvcvObjectSack = 14,
    itvcvObjectBox = 15,
    itvcvObjectGasBottle = 16,
    itvcvObjectChild = 17,
    itvcvObjectCat = 18,
    itvcvObjectOther = 19,
    itvcvObjectUndefined = 9999,
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum itvcvAnalyzerType
{
    itvcvAnalyzerClassification,
    itvcvAnalyzerSSD,
    itvcvAnalyzerSiamese,
    itvcvAnalyzerHumanPoseEstimator,
    itvcvAnalyzerMaskSegments,
    itvcvAnalyzerPointDetection,
    itvcvAnalyzerUnknown
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
    itvcvModeBalanced,
    itvcvModeHuaweiNPU
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum itvcvError
{
    itvcvErrorSuccess,                 // нет ошибок
    itvcvErrorFrameSize,               // неверно переданы размеры кадра
    itvcvErrorSensitivity,             // неверно передана чувствительность
    itvcvErrorColorScheme,             // неверно передана цветовая схема
    itvcvErrorNMeasure,                // неверно передана количество "замеров" для анализа огня
    itvcvErrorFrame,                   // передан "плохой" кадр
    itvcvErrorFramesNumber,            // ошибка в переданном числе кадров, которые обрабатываются единовременно
    itvcvErrorLoadANN,                 // ошибка при загрузке сети #
    itvcvErrorCaffeInitialization,     // caffe не удалось инициализироваться
    itvcvErrorNoCapableGPUDevice,      // не найдено ни одно подходящего GPU ресурса
    itvcvErrorInvalidGPUNumber,        // указан неверно номер используемого GPU
    itvcvErrorInconsistentGPUNumber,   // указан номер GPU, не соответсвующего заявленным требованиям
    itvcvErrorCUDADeviceCount,         // ошибка при попытке получить информацию об устройстве GPU
    itvcvErrorCUDASetDevice,           // ошибка при установке опеделенного устройства GPU
    itvcvErrorNoCapableMovidiusDevice, // не найдено ни одно подходящего Movidius ресурса
    itvcvErrorInvalidMovidiusNumber,   // указан неверно номер используемого Movidius устройства
    itvcvErrorMovidiusGetDeviceName,   // ошибка при получении имени устройства Movidius
    itvcvErrorMovidiusOpenDevice,      // ошибка при открытии устройства Movidius
    itvcvErrorMovidiusCloseDevice,     // ошибка при закрытии устройства Movidius
    itvcvErrorMovidiusCreateGraph,     // ошибка при создании графа для Movidius
    itvcvErrorMovidiusDeallocateGraph, // ошибка при удалении графа для Movidius
    itvcvErrorMovidiusLoadTensor,      // ошибка при загрузке тензора для Movidius
    itvcvErrorMovidiusGetResult,       // ошибка при получении результата от Movidius
    itvcvErrorOther,                   // другие возникшие ошибки
    itvcvErrorChoosingGPU,
    itvcvErrorLabelsValues,
    itvcvErrorParametersEmpty,    //нет вшитых параметров в сеть
    itvcvErrorParametersNotFound, // выбранный параметр не найден
    itvcvErrorGpuIsBusyBuildingNewEngine,
    itvcvErrorFullInferenceQueue, // достигнуто максимальное количество одновременных инференсов
    itvcvErrorNetworkAndDeviceMismatch, // неверный тип сети для выбранного устройства
    itvcvErrorSubFrameWrongSize, // неправильный размер окна для сканирующего детектора
    itvcvErrorInference, // ошибка во время выполнения инференса
    itvcvErrorAnalyzerTypeMismatch // ошибка несоответствия типа сети с создаваемым типом
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! уровень логирования
enum itvcvLogSeverity
{
    itvcvLogError,
    itvcvLogInfo,
    itvcvLogWarning,
    itvcvLogDebug
};

//! Описание типа callBack-a для логгирования
typedef void(*itvcvLogCallback_t) (void* userData, const char* str, itvcvLogSeverity severity);
typedef void(*itvcvCreationCallback_t) (void* userData, void* hEngine, itvcvError error);

namespace ItvCv
{
namespace Utils
{
ITVCV_UTILS_API std::shared_ptr<ITV8::ILogger> CreateLogCallbackWrapper(
    itvcvLogCallback_t callback,
    void* userData,
    itvcvLogSeverity logSeverity);
}
}
#endif
