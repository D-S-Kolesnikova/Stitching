#ifndef _ANALYZERINTELIE_H_
#define _ANALYZERINTELIE_H_

#include "IAnalyzer.h"

#include <memory>

namespace InferenceWrapper
{
template<itvcvAnalyzerType analyzerType>
std::unique_ptr<IAnalyzer<analyzerType>> CreateIntelIEAnalyzer(
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters);
} // InferenceWrapper
#endif
