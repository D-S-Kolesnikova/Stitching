#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvCvUtils/Math.h>

#include <boost/polygon/voronoi.hpp>
#include <boost/polygon/voronoi_diagram.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/polygon/voronoi.hpp>
#include <boost/circular_buffer.hpp>

namespace ublas = boost::numeric::ublas;

namespace
{
float Area(std::vector<ITV8::PointF>& triangle)
{
    return (triangle[0].x - triangle[2].x) * (triangle[1].y - triangle[2].y)
        - (triangle[1].x - triangle[2].x) * (triangle[0].y - triangle[2].y);
}

//Триангуляция для вершины Воронова обхобом по часовой стрелке
//input:
//const std::vector<int>& pointsIndexes - индексы точек сформировавшие вершину Воронова
//const std::vector<ITV8::PointF>& points - все точки
//output:
//std::vector<std::vector<ITV8::PointF>> - набор треугольников
std::vector<std::vector<ITV8::PointF>> ClockwiseTriangulation(
    const std::vector<int>& pointsIndexes,
    const std::vector<ITV8::PointF>& points)
{
    boost::circular_buffer<int> poolIndexes(pointsIndexes.begin(), pointsIndexes.end());
    auto firstPointId = 0;
    auto countPoints = static_cast<int>(poolIndexes.size());

    std::vector<std::vector<ITV8::PointF>> result;
    //(countPoints - 2) * 3  кол-во треугольников которое можно составить из N(countPoints) точек
    result.reserve((countPoints - 2) * 3);
    //Нахождение текущего id когда идем по кругу
    auto CircleId = [](const int& index, const int& maxSize) { return index % maxSize; };

    while (countPoints > 2)
    {
        const auto secondPointId = CircleId(firstPointId + 1, countPoints);
        const auto thirdPointId = CircleId(firstPointId + 2, countPoints);
        result.emplace_back(
            std::vector<ITV8::PointF>{
                points[poolIndexes[firstPointId]],
                points[poolIndexes[secondPointId]],
                points[poolIndexes[thirdPointId]]});
        poolIndexes.erase(poolIndexes.begin() + secondPointId);
        --countPoints;
        firstPointId = CircleId(firstPointId + 1, countPoints);
    }
    return result;
}

//Триангуляция для вершины Воронова простой алгоритм
//input:
//const std::vector<int>& pointsIndexes - индексы точек сформировавшие вершину Воронова
//const std::vector<ITV8::PointF>& points - все точки
//output:
//std::vector<std::vector<ITV8::PointF>> - набор треугольников
std::vector<std::vector<ITV8::PointF>> SimpleTriangulation(
    const std::vector<int>& pointsIndexes,
    const std::vector<ITV8::PointF>& points)
{
    auto countPoints = static_cast<int>(pointsIndexes.size());

    std::vector<std::vector<ITV8::PointF>> result;
    //(countPoints - 2) * 3  кол-во треугольников которое можно составить из N(countPoints) точек
    result.reserve((countPoints - 2) * 3);
    for (auto i = 1; i < static_cast<int>(pointsIndexes.size() - 1); ++i)
    {
        result.emplace_back(
            std::vector<ITV8::PointF>{
                points[pointsIndexes[0]],
                points[pointsIndexes[i]],
                points[pointsIndexes[i + 1]]});
    }
    return result;
}

//Врутренняя триангуляция для одной вершины Воронова
//input:
//pointsIndexes - индексы точек сформировавшие вершину
//points - все точки
//ItvCvUtils::Math::Geometry::ETriangulationType - тип триангуляции
//output:
//std::vector<std::vector<ITV8::PointF>> - набор треугольников
std::vector<std::vector<ITV8::PointF>> InnerTriangulation(
    const std::vector<int>& pointsIndexes,
    const std::vector<ITV8::PointF>& points,
    const ItvCvUtils::Math::Geometry::ETriangulationType triangulationType)
{
    switch (triangulationType)
    {
    case ItvCvUtils::Math::Geometry::ETriangulationType::Simple:
        return SimpleTriangulation(pointsIndexes, points);
    case ItvCvUtils::Math::Geometry::ETriangulationType::Clockwise:
        return ClockwiseTriangulation(pointsIndexes, points);
    default:
        return {};
    }
}

} // namespace
namespace boost
{
namespace polygon
{
template<>
struct geometry_concept<ITV8::PointF>
{
    typedef boost::polygon::point_concept type;
};

template<>
struct point_traits<ITV8::PointF>
{
    typedef int coordinate_type;

    static coordinate_type get(const ITV8::PointF& point, boost::polygon::orientation_2d orient)
    {
        return (orient == boost::polygon::HORIZONTAL) ? point.x : point.y;
    }
};
} // namespace polygon
} // namespace boost

typedef boost::geometry::model::d2::point_xy<double_t> BgPoint_t;
typedef boost::geometry::model::polygon<BgPoint_t> BgPolygon_t;
namespace ItvCvUtils
{
namespace Math
{

Tensor::Tensor(
    const std::array<int, 4>& dims,
    const std::vector<float>& data,
    const DimsOrder dimsOrder)
    : Tensor(
        std::array<int, 4>(dims),
        std::vector<float>(data),
        dimsOrder)
{}

Tensor::Tensor(
    std::array<int, 4>&& dims,
    std::vector<float>&& data,
    const DimsOrder dimsOrder)
    : m_dimsOrder(dimsOrder)
    , m_dims(dims)
    , m_data(data)
{
    switch (m_dimsOrder)
    {
    case DimsOrder::NCHW:
        m_idBatch = 0;
        m_idChannel = 1;
        m_idHeight = 2;
        m_idWidth = 3;
        break;

    case DimsOrder::NHWC:
        m_idBatch = 0;
        m_idHeight = 1;
        m_idWidth = 2;
        m_idChannel = 3;
        break;

    case DimsOrder::WHCN:
        m_idWidth = 0;
        m_idHeight = 1;
        m_idChannel = 2;
        m_idBatch = 3;
        break;

    case DimsOrder::CWHN:
        m_idChannel = 0;
        m_idWidth = 1;
        m_idHeight = 2;
        m_idBatch = 3;
        break;

    case DimsOrder::Undef:
    default:
        m_idWidth = m_idBatch = m_idChannel = m_idHeight = 0;
        break;
    }
}

double Geometry::Distance(const ITV8::PointF& vector)
{
    return sqrt(vector.x * vector.x + vector.y * vector.y);
}

double Geometry::Distance(const ITV8::PointF& first, const ITV8::PointF& second)
{
    const auto diff = first - second;
    return Distance(diff);
}

bool Geometry::IsInTriangleZone(const std::vector<ITV8::PointF>& zone, const ITV8::PointF& point)
{
    std::vector<ITV8::PointF> tempTriangle = {point, zone[0], zone[1]};
    const auto area1 = Area(tempTriangle);

    tempTriangle[1] = zone[1];
    tempTriangle[2] = zone[2];
    const auto area2 = Area(tempTriangle);

    tempTriangle[1] = zone[2];
    tempTriangle[2] = zone[0];
    const auto area3 = Area(tempTriangle);

    const auto hasNeg = (area1 < 0) || (area2 < 0) || (area3 < 0);
    const auto hasPos = (area1 > 0) || (area2 > 0) || (area3 > 0);
    return !(hasNeg && hasPos);
}

ITVCV_UTILS_API std::vector<double> SolveSLE(std::vector<std::vector<float>>& equation)
{
    const auto numArg = static_cast<int>(equation[0].size() - 1);
    const auto countEquals = static_cast<int>(equation.size());
    boost::numeric::ublas::compressed_matrix<double, ublas::column_major, 0> equals(numArg, numArg, countEquals);
    ublas::vector<double> y(countEquals);

    for (int i = 0; i < countEquals; ++i)
    {
        for (int j = 0; j <= numArg; ++j)
        {
            if (j == numArg)
            {
                y[i] = equation[i][j];
            }
            else
            {
                equals(i, j) = equation[i][j];
            }
        }
    }

    std::vector<double> returnValue(countEquals, 0);
    ublas::permutation_matrix<size_t> pm(equals.size1());
    lu_factorize(equals, pm);
    lu_substitute(equals, pm, y);
    std::copy(y.begin(), y.end(), returnValue.begin());
    return returnValue;
}

bool Geometry::IsPointInRect(const ITV8::RectangleF& rect, const ITV8::PointF& point)
{
    return rect.left <= point.x && (rect.left + rect.width) >= point.x && rect.top <= point.y
        && (rect.top + rect.height) >= point.y;
}

double Geometry::SquareOfTwoRectsIntersecection(const ITV8::RectangleF& rect1, const ITV8::RectangleF& rect2)
{
    double result = 0;
    boost::polygon::rectangle_data<double> r1 (rect1.left, rect1.top, rect1.right(), rect1.bottom());
    boost::polygon::rectangle_data<double> r2(rect2.left, rect2.top, rect2.right(), rect2.bottom());
    if (boost::polygon::intersect(r1, r2)) result = boost::polygon::area(r1);
    return result;
}

std::vector<std::vector<ITV8::PointF>> Geometry::Triangulation(
    std::vector<ITV8::PointF>& points,
    const ETriangulationType triangulationType)
{
    boost::polygon::voronoi_diagram<double> vd;
    boost::polygon::construct_voronoi(points.begin(), points.end(), &vd);

    std::vector<std::vector<ITV8::PointF>> ret;
    for (const auto& vertex : vd.vertices())
    {
        auto edge = vertex.incident_edge();
        // индексы точек сформировавшие вершину Воронова
        std::vector<int> pointIndexes;
        do
        {
            pointIndexes.emplace_back(edge->cell()->source_index());
            edge = edge->rot_next();
        } while (edge != vertex.incident_edge());

        auto triangles = InnerTriangulation(pointIndexes, points, triangulationType);
        ret.insert(ret.end(), triangles.begin(), triangles.end());
    }

    return ret;
}

double Geometry::Projection(const ITV8::PointF& first, const ITV8::PointF& second)
{
    return abs(ScalarProduct(first, second) / Distance(second));
}

ParametricLine::ParametricLine(): m_aX(0), m_aY(0), m_x1(0), m_y1(0), m_lambda(1) {}

ParametricLine::ParametricLine(const ITV8::PointF& first, const ITV8::PointF& second)
{
    m_lambda = 1;
    m_aY = second.y - first.y;
    m_aX = second.x - first.x;
    m_x1 = first.x;
    m_y1 = first.y;
}

std::vector<std::vector<float>> ParametricLine::GetEquals(const ParametricLine& other) const
{
    return {
        {m_aX, -other.m_aX, other.m_x1 - m_x1},
        {m_aY, -other.m_aY, other.m_y1 - m_y1},
    };
}

double ParametricLine::CosBetweenLines(const ParametricLine& secondLine) const
{
    return Geometry::Cos(GetPoint(0) - GetPoint(1), secondLine.GetPoint(0) - secondLine.GetPoint(1));
}

float ParametricLine::GetX(float y) const
{
    const auto tempLambda = (y - m_y1) / (m_aY);
    return tempLambda * m_aX + m_x1;
}

float ParametricLine::GetY(float x) const
{
    const auto tempLambda = (x - m_x1) / (m_aX);
    return tempLambda * m_aY + m_y1;
}

ITV8::PointF ParametricLine::GetPoint(float lambda) const
{
    return {m_aX * lambda + m_x1, m_aY * lambda + m_y1};
}

ITV8::PointF ParametricLine::GetParallelLinePoint(const ITV8::PointF& input) const
{
    return {m_aX * m_lambda + input.x, m_aY * m_lambda + input.y};
}

ParametricLine ParametricLine::GetOrthogonal(const ITV8::PointF& input)
{
    ParametricLine line;
    line.m_aX = m_aY;
    line.m_aY = -m_aX;
    line.m_lambda = 1;
    line.m_x1 = input.x;
    line.m_y1 = input.y;
    return line;
}

ParametricLine ParametricLine::GetParallelLine(const ITV8::PointF& input)
{
    ParametricLine line;
    line.m_aX = m_aX;
    line.m_aY = m_aY;
    line.m_lambda = m_lambda;
    line.m_x1 = input.x;
    line.m_y1 = input.y;
    return line;
}

bool ParametricLine::IsPointOnLine(const ITV8::PointF& point) const
{
    const auto tempLambdaX = (point.x - m_x1) / m_aX;
    const auto tempLambdaY = (point.y - m_y1) / m_aY;
    return tempLambdaX == tempLambdaY;
}

double Geometry::Cos(const ITV8::PointF& first, const ITV8::PointF& second)
{
    return (ScalarProduct(first, second) / (Distance(first) * Distance(second)));
}

double Geometry::ScalarProduct(const ITV8::PointF& first, const ITV8::PointF& second)
{
    return first.x * second.x + first.y * second.y;
}

//Intersection over union
double Geometry::IoU(const ITV8::RectangleF& predBbox, const ITV8::RectangleF& labelBbox)
{
    const auto left = std::max(predBbox.left, labelBbox.left);
    const auto top = std::max(predBbox.top, labelBbox.top);
    const auto right = std::min(predBbox.right(), labelBbox.right());
    const auto bottom = std::min(predBbox.bottom(), labelBbox.bottom());

    if (right < left || bottom < top)
    {
        return 0.0;
    }
    const auto intersectionArea = (right - left) * (bottom - top);
    const auto predBboxArea = predBbox.width * predBbox.height;
    const auto labelBboxArea = labelBbox.width * labelBbox.height;
    return intersectionArea / (predBboxArea + labelBboxArea - intersectionArea);
}

ITVCV_UTILS_API ParametricLine Ransac2d(const std::vector<ITV8::PointF>& hor, const float delta)
{
    auto maxM = 0;
    ParametricLine result;
    for (auto i = 0; i < hor.size(); ++i)
    {
        for (auto j = i + 1; j < hor.size(); ++j)
        {
            ParametricLine temp(hor[i], hor[j]);
            auto onLine = 0;
            for (auto k = 0; k < hor.size(); ++k)
            {
                if (k != i && k != j)
                {
                    const auto cos = Geometry::Cos(hor[k] - hor[i], hor[j] - hor[i]);

                    const auto sin = std::sqrt(1 - cos * cos);
                    const auto dis = Geometry::Distance(hor[i], hor[k]) * sin;

                    if (dis <= delta || temp.IsPointOnLine(hor[k]))
                    {
                        ++onLine;
                    }
                }
            }
            if (onLine >= maxM)
            {
                maxM = onLine;
                result = temp;
            }
        }
    }
    return result;
}
} // namespace Math
} // namespace ItvCvUtils
