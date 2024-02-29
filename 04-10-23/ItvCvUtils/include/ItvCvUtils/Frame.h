#ifndef FRAME_H
#define FRAME_H

namespace ItvCv
{

enum class MemType
{
    Host,
    GPU,
};

enum class PixelFormat
{
    Unspecified,
    BGR, // Packed
    RGB, // Packed
    Y,
    NV12,
    YUV420,
    YUV422,
    YUV444,
    GRAY,
};

struct Frame
{
    Frame(
        int width,
        int height,
        int stride,
        const unsigned char* data,
        MemType memType = MemType::Host,
        PixelFormat format = PixelFormat::BGR)
        : width(width)
        , height(height)
        , channels(format == PixelFormat::Y ? 1 : 3) // костыл
        , stride(stride)
        , data(data)
        , memType(memType)
        , format(format)
    {
    }

    int width{};
    int height{};
    int channels{};
    int stride{};
    const unsigned char* data{};
    MemType memType{};
    PixelFormat format{};
};

}

#endif