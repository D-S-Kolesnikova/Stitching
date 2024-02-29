#ifndef _ANALYZER_POOL_H
#define _ANALYZER_POOL_H

#include "AnalyzerFactory.h"

#include <InferenceWrapper/InferenceEngine.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>

#include <memory>


namespace InferenceWrapper
{

std::shared_ptr<void> GetSharedAnalyzer(
    const EngineCreationParams& parameters,
    const std::function<std::shared_ptr<void>()>& creationFunc);

template<itvcvAnalyzerType analyzerType>
std::shared_ptr<IAnalyzer<analyzerType>> GetSharedAnalyzer(
    const EngineCreationParams& parameters,
    const std::function<std::shared_ptr<void>()>& creationFunc)
{
    return std::static_pointer_cast<IAnalyzer<analyzerType>>(GetSharedAnalyzer(parameters, creationFunc));
}

}
#endif //_ENGINE_POOL_H
