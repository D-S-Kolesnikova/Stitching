#ifndef INFERENCE_ENGINE_PARAMS_H
#define INFERENCE_ENGINE_PARAMS_H

#include <NetworkInformation/NetworkInformationLib.h>

namespace InferenceWrapper
{

struct EngineCreationParams
{
    EngineCreationParams() = delete;
    EngineCreationParams(
        ITV8::ILogger* logger,
        itvcvAnalyzerType netType,
        itvcvModeType mode,
        int colorScheme,
        int gpuDeviceNumToUse,
        std::string model_path,
        std::string weights_path,
        std::string plugin_dir = std::string(),
        bool int8 = false)
        : logger(logger)
        , netType(netType)
        , mode(mode)
        , colorScheme(colorScheme)
        , gpuDeviceNumToUse(gpuDeviceNumToUse)
        , modelFilePath(std::move(model_path))
        , weightsFilePath(std::move(weights_path))
        , net(ItvCv::GetNetworkInformation(weightsFilePath.c_str()))
        , pluginDir(std::move(plugin_dir))
        , int8(int8)
    {
    }

    ITV8::ILogger* logger;
    itvcvAnalyzerType netType;
    itvcvModeType mode;
    int colorScheme;
    int gpuDeviceNumToUse;
    std::string modelFilePath;
    std::string weightsFilePath;
    ItvCv::PNetworkInformation net;
    std::string pluginDir;
    bool int8 = false;
};

}

#endif