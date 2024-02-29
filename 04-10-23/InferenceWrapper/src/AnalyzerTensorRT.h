#ifndef _ANALYZERTENSORRT_H_
#define _ANALYZERTENSORRT_H_

#include "IAnalyzer.h"

#include <memory>

namespace InferenceWrapper
{
template<itvcvAnalyzerType analyzerType>
std::unique_ptr<IAnalyzer<analyzerType>> CreateTensorRTAnalyzer(
    itvcvError& error,
    std::shared_ptr<EngineCreationParams> parameters);
} // InferenceWrapper
#endif
