#ifndef TRTENGINEFACTORY_H
#define TRTENGINEFACTORY_H

#include <ItvCvUtils/Log.h>

#include <NvInferRuntime.h>

#include <vector>
#include <string>
#include <memory>

namespace InferenceWrapper
{
class FileCacheHelper
{
public:
    FileCacheHelper(ITV8::ILogger* logger);

    bool Save(
        const std::string& fileName,
        std::shared_ptr<nvinfer1::ICudaEngine> engine);

    std::shared_ptr<nvinfer1::ICudaEngine> Load(
        const std::string& fileName);

    std::string GetFileName(
        const ITV8::Size& inputSize,
        const std::string& model,
        const std::string& weights,
        const std::string& originalFileName,
        const bool int8Usage) const;

private:
    ITV8::ILogger* m_logger;
};
}

#endif