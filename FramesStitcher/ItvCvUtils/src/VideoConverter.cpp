#include <ItvCvUtils/VideoConverter.h>

extern "C"
{
#include <libswscale/swscale.h>
}

namespace ItvCv {
namespace Utils {

template <AVPixelFormat outFormatType>
class CYUV2TargetSchemeConverter : public IYUV2TargetSchemeConverter
{
public:
    void operator()
        (int nWidth,
            int nHeight,
            const uint8_t* frameDataY,
            int nStrideY,
            const uint8_t* frameDataU,
            const uint8_t* frameDataV,
            int nStrideUV,
            EYUVType type,
            uint8_t *outFrameData,
            int nOutStride)
    {
        AVPixelFormat inFormat = AV_PIX_FMT_GRAY8;
        switch (type)
        {
        case EYUVType::YUV420:
            inFormat = AV_PIX_FMT_YUV420P;
            break;
        case EYUVType::YUV420SP:
            inFormat = AV_PIX_FMT_NV12;
            break;
        case EYUVType::YUV422:
            inFormat = AV_PIX_FMT_YUV422P;
            break;
        case EYUVType::YUV444:
            inFormat = AV_PIX_FMT_YUV444P;
            break;
        }

        if (m_inFormat != inFormat
            || m_frameWidth != nWidth
            || m_frameHeight != nHeight)
        {
            ctx = sws_getCachedContext(ctx, nWidth, nHeight, inFormat, nWidth, nHeight, outFormatType, 0, 0, 0, 0);
            m_inFormat = inFormat;
            m_frameWidth = nWidth;
            m_frameHeight = nHeight;
        }

        const uint8_t* inDataColor[] = { frameDataY,  frameDataU, frameDataV };
        const uint8_t* inDataGray[] = { frameDataY };
        const int inStrideColor[] = { nStrideY, nStrideUV, nStrideUV };
        const int inStrideGray[] = { nStrideY };
        uint8_t* outData[] = { outFrameData };
        const int outStride[] = { nOutStride };
        const bool grayImage = inFormat == AV_PIX_FMT_GRAY8;
        sws_scale(ctx, grayImage ? inDataGray : inDataColor, grayImage ? inStrideGray : inStrideColor, 0, nHeight, outData, outStride);
    }

    ~CYUV2TargetSchemeConverter()
    {
        if (ctx)
        {
            sws_freeContext(ctx);
        }
    }

private:
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    int m_inFormat = 0;

    SwsContext* ctx = 0;
};

IYUV2TargetSchemeConverter* CreateYUV2RGBConverter()
{
    return new CYUV2TargetSchemeConverter<AV_PIX_FMT_RGB24>();
}

IYUV2TargetSchemeConverter* CreateYUV2BGRConverter()
{
    return new CYUV2TargetSchemeConverter<AV_PIX_FMT_BGR24>();
}

IYUV2TargetSchemeConverter* CreateYUV2GrayConverter()
{
    return new CYUV2TargetSchemeConverter<AV_PIX_FMT_GRAY8>();
}

}}
