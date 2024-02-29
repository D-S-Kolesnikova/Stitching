#include <ItvCvUtils/Gpu/FrameConverter.h>

#include <cuda_runtime.h>
#include <nppi_color_conversion.h>

#include <fmt/core.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace ItvCv {
namespace Utils {
namespace Gpu {

template <decltype(&nppiNV12ToBGR_8u_P2C3R) ConvFunc>
class YuvFrameConverter : public IYuvFrameConverter
{
public:
    bool Convert(
        const int width,
        const int height,
        const int stride[],
        const std::uint8_t* src[],
        const ItvCv::PixelFormat inputFormat,
        const int outputStride,
        std::uint8_t* const output)
    {
        NppiSize oSizeROI;
        oSizeROI.width = width;
        oSizeROI.height = height;

        return NPP_SUCCESS == ConvFunc(src, stride[0], output, outputStride, oSizeROI);
    }
};

std::unique_ptr<IYuvFrameConverter> CreateYuvConverter(ItvCv::PixelFormat targetFormat)
{
    switch (targetFormat)
    {
    case ItvCv::PixelFormat::BGR: return std::make_unique<YuvFrameConverter<nppiNV12ToBGR_8u_P2C3R>>();
    case ItvCv::PixelFormat::RGB: return std::make_unique<YuvFrameConverter<nppiNV12ToRGB_8u_P2C3R>>();
    default: throw std::runtime_error("Unsupported target format.");
    }
}

}}}