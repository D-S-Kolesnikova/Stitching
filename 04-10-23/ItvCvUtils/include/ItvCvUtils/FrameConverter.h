#ifndef FRAME_CONVERTER_H
#define FRAME_CONVERTER_H

#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvCvUtils/Frame.h>

#include <cstdint>
#include <memory>

namespace ItvCv {
namespace Utils {

struct ITVCV_UTILS_API IYuvFrameConverter
{
    virtual ~IYuvFrameConverter() = default;

    virtual bool Convert(
        const int width,
        const int height,
        const int stride[],
        const std::uint8_t* src[],
        const ItvCv::PixelFormat inputFormat,
        const int outputStride,
        std::uint8_t* const output) = 0;
};

ITVCV_UTILS_API std::unique_ptr<IYuvFrameConverter> CreateYuvConverter(ItvCv::PixelFormat targetFormat);

}}

#endif