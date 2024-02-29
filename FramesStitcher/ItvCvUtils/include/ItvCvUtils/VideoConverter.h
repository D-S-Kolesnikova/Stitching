#ifndef VIDEOCONVERTER_H
#define VIDEOCONVERTER_H

#include <ItvCvUtils/ItvCvUtils.h>
#include <stdint.h>

namespace ItvCv
{
namespace Utils
{
enum EYUVType
{
    YUV420,
    YUV420SP,
    YUV422,
    YUV444,
    Y
};

enum PixelFormat
{
    PIXEL_FORMAT_BGR24,
    PIXEL_FORMAT_RGB24,
    PIXEL_FORMAT_GRAY8
};

class ITVCV_UTILS_API IYUV2TargetSchemeConverter
{
public:
    virtual void operator()(
        int nWidth,
        int nHeight,
        const uint8_t* frameDataY,
        int nStrideY,
        const uint8_t* frameDataU,
        const uint8_t* frameDataV,
        int nStrideUV,
        EYUVType type,
        uint8_t* outFrameData,
        int nOutStride) = 0;

    virtual ~IYUV2TargetSchemeConverter() {}
};

ITVCV_UTILS_API IYUV2TargetSchemeConverter* CreateYUV2RGBConverter();

ITVCV_UTILS_API IYUV2TargetSchemeConverter* CreateYUV2BGRConverter();

ITVCV_UTILS_API IYUV2TargetSchemeConverter* CreateYUV2GrayConverter();

} // namespace Utils
} // namespace ItvCv

#endif
