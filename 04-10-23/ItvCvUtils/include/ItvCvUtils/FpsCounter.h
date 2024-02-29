#ifndef COMPUTERVISION_FPSCOUNTER_H
#define COMPUTERVISION_FPSCOUNTER_H

#include <ItvCvUtils/ItvCvUtils.h>

#include <ItvSdk/include/IErrorService.h>

#include <memory>
#include <string>

namespace ItvCvUtils
{
struct IFpsCounter
{
    virtual void Reset() = 0;
    virtual void Increment() = 0;
    virtual double Fps() const = 0;
    virtual void ForceCalcFps() = 0;
    virtual ~IFpsCounter() = default;
};

ITVCV_UTILS_API std::unique_ptr<IFpsCounter> CreateFpsCounter(const std::string& name, void* id, ITV8::ILogger* logger);
}

#endif // COMPUTERVISION_FPSCOUNTER_H
