#ifndef COMPUTERVISION_ANALYZERFACTORY_H
#define COMPUTERVISION_ANALYZERFACTORY_H

#include "IAnalyzer.h"
#include "InferenceHelperFunctions.h"

#ifdef USE_INFERENCE_ENGINE_BACKEND
#include "AnalyzerIntelIE.h"
#endif
#ifdef USE_TENSORRT_BACKEND
#include "AnalyzerTensorRT.h"
#endif
#ifdef USE_ATLAS300_BACKEND
#include "AnalyzerAtlas300.h"
#endif

#include <ItvCvUtils/NeuroAnalyticsApi.h>


namespace InferenceWrapper
{

struct AnalyzerFactory
{
    template<itvcvAnalyzerType analyzerType>
    static std::shared_ptr<IAnalyzer<analyzerType>> Create(
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters)
    {
        switch (parameters->mode)
        {
        case itvcvModeGPU:
        {
#ifdef USE_TENSORRT_BACKEND
            return CreateTensorRTAnalyzer<analyzerType>(
                error,
                parameters);
#else
            throw std::runtime_error("GPU backend is not supported");
#endif
        }
        case itvcvModeCPU:
        case itvcvModeHDDL:
        case itvcvModeMULTI:
        case itvcvModeHetero:
        case itvcvModeBalanced:
        case itvcvModeFPGAMovidius:
        case itvcvModeIntelGPU:
        {
#ifdef USE_INFERENCE_ENGINE_BACKEND
            return CreateIntelIEAnalyzer<analyzerType>(
                error,
                parameters);
#else
            throw std::runtime_error("Intel InferenceEngine backend is not supported");
#endif
        }
        case itvcvModeHuaweiNPU:
        {
#ifdef USE_ATLAS300_BACKEND
            return CreateAtlas300Analyzer<analyzerType>(
                error,
                parameters);
        break;
#else
            throw std::runtime_error("Huawei NPU backend is not supported");
#endif
        }
        default:
            throw std::invalid_argument("Unexpected analyzer backed specified: " + std::to_string(parameters->mode));
        }
    }
};

}

#endif // COMPUTERVISION_ANALYZERFACTORY_H
