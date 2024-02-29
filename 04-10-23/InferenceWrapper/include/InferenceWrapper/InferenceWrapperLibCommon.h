#ifndef _INFERENCEWRAPPERLIBCOMMON_H_
#define _INFERENCEWRAPPERLIBCOMMON_H_

#include <ItvCvUtils/PoseC.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>

#ifndef BATCHSIZE_ASYNC
#define BATCHSIZE_ASYNC 1
#endif

#ifndef SSD_RESULT_VECTOR_SIZE
#define SSD_RESULT_VECTOR_SIZE 7
#endif

struct SParametersHPE
{
    float* OutBoxSize;
    int* OutStride;
    float* OutMinPeaksDistance;
    float* OutMidPointsScoreThreshold;
    float* OutFoundMidPointsRatioThreshold;
    float* OutMinSubsetScore;
    int* OutUpsampleRatio;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! Описание типа callBack-a для получения результата асинхронно
typedef void(*iwResultCallback_t)(void* userData, float* result, int detectionSize);
typedef void(*iwSSDResultCallback_t)(void* userData,  const float** result, int resultCount, int detectionSize);
typedef void(*iwHPEResultCallback_t)(void* userData, int resultCount, ItvCvUtils::PoseC* result);
typedef void(*iwMASKResultCallback_t) (void* userData, int *outMaskDims, float *OutMask, itvcvError *error);
#endif
