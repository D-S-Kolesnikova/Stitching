#ifndef GPU_FRAME_CONVERTER_H
#define GPU_FRAME_CONVERTER_H

#include <ItvCvUtils/FrameConverter.h>

namespace ItvCv {
namespace Utils {
namespace Gpu {

ITVCV_UTILS_API std::unique_ptr<IYuvFrameConverter> CreateYuvConverter(ItvCv::PixelFormat targetFormat);

}}}

#endif