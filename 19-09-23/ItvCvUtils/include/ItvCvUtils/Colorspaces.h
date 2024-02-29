#ifndef COLORSPACES_H_18980FDE_1E39_4f6a_B3A7_D355BDFD208D
#define COLORSPACES_H_18980FDE_1E39_4f6a_B3A7_D355BDFD208D

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <tuple>

namespace ITV8
{
namespace Colorspaces
{
template<class T>
T clip(T x, T min, T max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

inline std::uint8_t clip_uint8(std::int_fast32_t x)
{
    return static_cast<std::uint8_t>(clip<std::int_fast32_t>(x, 0, 255));
};

inline std::uint8_t clip_uint8(double x)
{
    return static_cast<std::uint8_t>(clip<double>(x, 0., 255.));
};

inline void yuv_to_rgb(
    std::uint8_t y,
    std::uint8_t u,
    std::uint8_t v,
    std::uint8_t& r,
    std::uint8_t& g,
    std::uint8_t& b)
{
    const auto c = std::int_fast32_t {y} - 16;
    const auto d = std::int_fast32_t {u} - 128;
    const auto e = std::int_fast32_t {v} - 128;
    r = clip_uint8((298 * c + 409 * e + 128) >> 8);
    g = clip_uint8((298 * c - 100 * d - 208 * e + 128) >> 8);
    b = clip_uint8((298 * c + 516 * d + 128) >> 8);
}

inline void rgb_to_hsv(
    std::uint8_t r,
    std::uint8_t g,
    std::uint8_t b,
    std::uint8_t& hue,
    std::uint8_t& sat,
    std::uint8_t& val)
{
    std::uint8_t min, max;
    std::tie(min, max) = std::minmax({r, g, b});
    const auto chroma = max - min;

    val = max;

    if (chroma == 0)
    {
        hue = 0;
        sat = 0;
        return;
    }

    if (max == r)
    {
        if (g >= b)
            hue = 0u + 85u * (g - b) / chroma / 2;
        else
            hue = static_cast<uint8_t>(256u - 85u * (b - g) / chroma / 2);
    }
    else if (max == g)
    {
        if (b >= r)
            hue = 85u + 85u * (b - r) / chroma / 2;
        else
            hue = 85u - 85u * (r - b) / chroma / 2;
    }
    else // (max == b)
    {
        if (r >= g)
            hue = 170u + 85u * (r - g) / chroma / 2;
        else
            hue = 170u - 85u * (g - r) / chroma / 2;
    }
    sat = 255u * chroma / max;
}

inline void yuv_to_hsv(
    std::uint8_t y,
    std::uint8_t u,
    std::uint8_t v,
    std::uint8_t& hue,
    std::uint8_t& sat,
    std::uint8_t& val)
{
    std::uint8_t r, g, b;
    yuv_to_rgb(y, u, v, r, g, b);
    rgb_to_hsv(r, g, b, hue, sat, val);
}

// ������� �������� ����������� �� �������������� RGB [0..255, 0..255, 0..255]
// � �������������� �������� HSV ��������� [0-360, 0-1, 0-1]
// ������� �������������� �� algorithm Smith.
// ����� ��������� �������� ���� ��������� �����:
// https://ru.wikipedia.org/wiki/HSV_(%D1%86%D0%B2%D0%B5%D1%82%D0%BE%D0%B2%D0%B0%D1%8F_%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C)#RGB_%E2%86%92_HSV
inline void rgb_to_hsv(std::uint8_t r, std::uint8_t g, std::uint8_t b, double& hue, double& sat, double& val)
{
    double min, max;
    // ������� ����������� RGB � [0,1]
    const auto r_new = static_cast<double>(r) / 255.;
    const auto g_new = static_cast<double>(g) / 255.;
    const auto b_new = static_cast<double>(b) / 255.;

    // ������� ��������� ������������ (max) � ����������� (min) ������� �� R,G,B
    // ����� ������������ �������� chroma = max-min.
    // val ��������� �������� max.
    // sat ��������� �������� (chroma/max), ���� chroma �� ����� ����, ����� sat = 0;
    // ���� chroma = 0, ������������ (hue = �������������� , sat = 0, val = max)
    // � �������� hue �������� �� ���������� �������:
    //            (  60 x (G-B)/chroma + 0   , ���� max = R � G>=B
    //            |  60 x (G-B)/chroma + 360 , ���� max = R � G<B
    // hue =     <
    //            |  60 x (B-R)/chroma + 120, ���� max = G
    //            (  60 x (R-G)/chroma + 240, ���� max = B
    std::tie(min, max) = std::minmax({r_new, g_new, b_new});
    const auto chroma = max - min;
    val = max;
    if (std::fabs(chroma) < std::numeric_limits<double>::epsilon())
    {
        // �������� ���������, ��� ���������� ����� ������� �������� hue
        // �������� ��������������, �� �� ��� ������������ �����.
        hue = 0.;
        sat = 0.;
        return;
    }
    if (max == r_new)
    {
        hue = 60. * (g_new - b_new) / chroma;
        if (g_new < b_new)
            hue += 360.;
    }
    else if (max == g_new)
        hue = 120. + 60. * (b_new - r_new) / chroma;
    else // (max == b)
        hue = 240. + 60. * (r_new - g_new) / chroma;
    sat = chroma / max;
}

// ������� ����������� �� YUV � HSV (yuv->rgb->hsv)
inline void yuv_to_hsv(std::uint8_t y, std::uint8_t u, std::uint8_t v, double& hue, double& sat, double& val)
{
    std::uint8_t r, g, b;
    yuv_to_rgb(y, u, v, r, g, b);
    rgb_to_hsv(r, g, b, hue, sat, val);
}

// ������� ����������� �� h [0, 360], s [0, 1], v [0, 1] �����������
// � r [0..255], g [0..255], b [0..255] ����������.
// ��������� �������� ���� ��������� �����: https://en.wikipedia.org/wiki/HSL_and_HSV
inline void hsv_to_rgb(double hue, double sat, double val, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b)
{
    // ������� �������� �������� chroma �� �������: chroma = val x sat;
    // HSV ������� ����������� �� ����� ������, � ����������� �� ���� hue,
    // � ���������� h_i ������������ ����� ������� (�� hue), � ������� ��������� �������� hsv.
            // � ����������� �� ���������� ������� ������������ �������� (R, G, B)
    // �������� �������:
    //               (  (val, X, m) , h_i = 0
    //               |  (X, val, m) , h_i = 1
    //               |  (m, val, X) , h_i = 2
    // (R,G,B) =    <
    //               |  (m, X, val) , h_i = 3
    //               |  (X, m, val) , h_i = 4
    //               (  (val, m, X) , h_i = 5
    // ��� �������� m, X ������������ ��������� �������:
    // m = val - chroma,
    // X = chroma x (1 - |Hi mod 2 - 1|) + m,
    // ����� Hi = hue / 60.
    // � ���������� �� �������� ��� ������ ���������� RGB �������� �� [0, 1],
    // ������� � ����� �������� �� RGB ��������� �� 255 � ��������� �� ����� �����.
    const auto chroma = val * sat;
    const auto Hi = hue / 60.;
    const auto h_i = static_cast<int>(Hi) % 6;
    const auto m = val - chroma;
    const auto X = chroma * (1. - std::fabs(std::fmod(Hi, 2.) - 1.)) + m;
    double RGB[3] = { 0,0,0 };
    // ����������� �������� ��������� �������� val , X, m
    // � ������������ � ��������������� ��������.
    const auto ind_val = ((h_i + 1) >> 1) % 3;
    const auto ind_m = ((h_i >> 1) + 2) % 3;
    const auto ind_X = (4 - (h_i % 3)) % 3;

    RGB[ind_val] = std::round(val * 255.);
    RGB[ind_m] = std::round(m * 255.);
    RGB[ind_X] = std::round(X * 255.);

    r = static_cast<uint8_t>(RGB[0]);
    g = static_cast<uint8_t>(RGB[1]);
    b = static_cast<uint8_t>(RGB[2]);
}
} // namespace Colorspaces
} // namespace ITV8

#endif
