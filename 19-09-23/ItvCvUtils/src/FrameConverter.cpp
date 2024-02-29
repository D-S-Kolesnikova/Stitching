#include <ItvCvUtils/FrameConverter.h>

#include <stdexcept>

extern "C"
{
#include <libswscale/swscale.h>
}

namespace ItvCv {
namespace Utils {

class YuvFrameConverter : public IYuvFrameConverter
{
public:
    YuvFrameConverter(ItvCv::PixelFormat targetFormat)
    {
        m_outputFormat = ItvCvToAvPixelFormat(targetFormat);
    }

    ~YuvFrameConverter()
    {
        if (m_ctx)
        {
            sws_freeContext(m_ctx);
        }
    }

    bool Convert(
        const int width,
        const int height,
        const int stride[],
        const std::uint8_t* src[],
        const ItvCv::PixelFormat inputFormat,
        const int outputStride,
        std::uint8_t* const output)
    {
        auto avPixelFormat = ItvCvToAvPixelFormat(inputFormat);

        if (m_lastFormat != avPixelFormat
            || m_lastWidth != width
            || m_lastHeight != height)
        {
            m_ctx = sws_getCachedContext(m_ctx, width, height, avPixelFormat, width, height, m_outputFormat, 0, 0, 0, 0);
            m_lastFormat = avPixelFormat;
            m_lastWidth = width;
            m_lastHeight = height;
        }

        return 0 == sws_scale(m_ctx, src, stride, 0, height, &output, &outputStride);
    }

private:
    AVPixelFormat ItvCvToAvPixelFormat(ItvCv::PixelFormat format)
    {
        switch (format)
        {
        case ItvCv::PixelFormat::BGR:    return AV_PIX_FMT_BGR24;
        case ItvCv::PixelFormat::RGB:    return AV_PIX_FMT_RGB24;
        case ItvCv::PixelFormat::NV12:   return AV_PIX_FMT_NV12;
        case ItvCv::PixelFormat::YUV420: return AV_PIX_FMT_YUV420P;
        case ItvCv::PixelFormat::YUV422: return AV_PIX_FMT_YUV422P;
        case ItvCv::PixelFormat::YUV444: return AV_PIX_FMT_YUV444P;
        default:                         return AV_PIX_FMT_GRAY8;
        }
    }

private:
    int m_lastWidth{};
    int m_lastHeight{};
    int m_lastFormat{};

    SwsContext* m_ctx{};
    AVPixelFormat m_outputFormat{};
};

std::unique_ptr<IYuvFrameConverter> CreateYuvConverter(ItvCv::PixelFormat targetFormat)
{
    switch (targetFormat)
    {
    case ItvCv::PixelFormat::RGB: return std::make_unique<YuvFrameConverter>(ItvCv::PixelFormat::RGB);
    case ItvCv::PixelFormat::BGR: return std::make_unique<YuvFrameConverter>(ItvCv::PixelFormat::BGR);
    default: throw std::runtime_error("Unsupported target format.");
    }
}

}}
