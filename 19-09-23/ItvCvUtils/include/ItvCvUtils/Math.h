#ifndef ITVCVUTILS_MATH_H
#define ITVCVUTILS_MATH_H

#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvCvUtils/PointOperators.h>
#include <ItvSdk/include/baseTypes.h>

#include <vector>
#include <array>

namespace
{
constexpr auto COUNT_DIMS = 4;
}

namespace ItvCvUtils
{
namespace Math
{
enum class DimsOrder
{
    Undef = 0,
    NCHW, //num(batch), channel, height, width
    WHCN, //width, height, channel, num(batch)
    NHWC, //num(batch),  height, width, channel
    CWHN, //channel, width, height, num(batch)
};

struct ITVCV_UTILS_API Tensor
{
public:
    Tensor() = default;

    Tensor(
        const std::array<int, 4>& dims,
        const std::vector<float>& data,
        DimsOrder dimsOrder);

    Tensor(
        std::array<int, 4>&& dims,
        std::vector<float>&& data,
        DimsOrder dimsOrder);

    ~Tensor() = default;

    int Width() const
    {
        return m_dims[m_idWidth];
    }

    int Height() const
    {
        return m_dims[m_idHeight];
    }

    int NumChannel() const
    {
        return m_dims[m_idChannel];
    }

    int NumBatch() const
    {
        return m_dims[m_idBatch];
    }

    std::vector<float>& GetData()
    {
        return m_data;
    }

private:
    uint8_t m_idWidth = 0;
    uint8_t m_idHeight = 0;
    uint8_t m_idChannel = 0;
    uint8_t m_idBatch = 0;

    DimsOrder m_dimsOrder = DimsOrder::Undef;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
    std::array<int, COUNT_DIMS> m_dims;
    std::vector<float> m_data;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
};

class ITVCV_UTILS_API ParametricLine
{
public:
    ParametricLine();
    ParametricLine(const ITV8::PointF& first, const ITV8::PointF& second);

    // получение СЛАУ  для решения
    std::vector<std::vector<float>> GetEquals(const ParametricLine& other) const;

    double CosBetweenLines(const ParametricLine& secondLine) const;

    // получение x по y
    float GetX(float y) const;
    // получение у по х
    float GetY(float x) const;
    // получение точки по lambda
    ITV8::PointF GetPoint(float lambda) const;

    ITV8::PointF GetParallelLinePoint(const ITV8::PointF& input) const;

    bool IsPointOnLine(const ITV8::PointF& point) const;

    ParametricLine GetOrthogonal(const ITV8::PointF& input);
    ParametricLine GetParallelLine(const ITV8::PointF& input);

private:
    float m_aX;
    float m_aY;
    float m_x1;
    float m_y1;
    float m_lambda;
};

class ITVCV_UTILS_API Geometry
{
public:
    enum class ETriangulationType
    {
        Simple,
        Clockwise
    };

public:
    static double Cos(const ITV8::PointF& firstVector, const ITV8::PointF& secondVector);
    static double ScalarProduct(const ITV8::PointF& first, const ITV8::PointF& second);
    static double Distance(const ITV8::PointF& vector);
    static double Distance(const ITV8::PointF& first, const ITV8::PointF& second);
    static double Projection(const ITV8::PointF& firstVector, const ITV8::PointF& secondVector);
    static bool IsPointInRect(const ITV8::RectangleF& rect, const ITV8::PointF& point);
    static double SquareOfTwoRectsIntersecection(const ITV8::RectangleF& rect1, const ITV8::RectangleF& rect2);

    static std::vector<std::vector<ITV8::PointF>> Triangulation(
            std::vector<ITV8::PointF>& points,
            ETriangulationType triangulationType = ETriangulationType::Clockwise);
    static bool IsInTriangleZone(const std::vector<ITV8::PointF>& zone, const ITV8::PointF& point);
    static double IoU(const ITV8::RectangleF& predBbox, const ITV8::RectangleF& labelBbox);
};

ITVCV_UTILS_API std::vector<double> SolveSLE(std::vector<std::vector<float>>& equation);

ITVCV_UTILS_API ParametricLine Ransac2d(const std::vector<ITV8::PointF>& hor, float delta);

} // namespace Math
} // namespace ItvCvUtils

#endif
