#ifndef _ANALYZERATLAS300_H_
#define _ANALYZERATLAS300_H_

#include "IAnalyzer.h"

#include <string>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <algorithm>
#include <libgen.h>
#include <atomic>

class Ascend;

namespace InferenceWrapper
{
    template<itvcvAnalyzerType analyzerType>
    std::unique_ptr<IAnalyzer<analyzerType>> CreateAtlas300Analyzer (
        itvcvError& error,
        std::shared_ptr<EngineCreationParams> parameters
    );
}

#endif //_ANALYZERATLAS300_H_
