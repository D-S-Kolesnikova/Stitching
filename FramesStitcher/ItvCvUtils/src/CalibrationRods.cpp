#include <cassert>
#include <tuple>

#include <ItvCvUtils/CalibrationRods.h>
#include <ItvCvUtils/PointOperators.h>

#include "ItvCvUtils/Pose.h"

namespace
{
constexpr auto COUNT_STEPS_FOR_DISTANCE = 30;
} // namespace
namespace ItvCvUtils
{
namespace CalibrationTools
{
CalibrationRods::CalibrationRods(const std::vector<Math::SLine>& rodsRaw, const ITV8::Size& imgSize)
{
    Clear();
    SetSize(imgSize);
    for (const auto& rod : rodsRaw)
    {
        AddRod(rod.first, rod.second);
    }
    Calibrate();
}

CalibrationRods::CalibrationRods(const CalibrationRods& rods)
    : m_zones(rods.m_zones)
    , m_levelingRods(rods.m_levelingRods)
    , m_horizonPole(rods.m_horizonPole)
    , m_width(rods.m_width)
    , m_height(rods.m_height)
{}

void CalibrationRods::Clear()
{
    m_horizonPole.clear();
    m_levelingRods.clear();
    m_zones.clear();
}
double CalibrationRods::GetDistanceToPoint(ITV8::PointF& firstBottomPoint, ITV8::PointF& secondBottomPoint, const double humanWidthCoefficient, double humanHeight) const
{
    auto firstTopPoint = GetTopPoint(firstBottomPoint);
    auto secondTopPoint = GetTopPoint(secondBottomPoint);
    //нахождение минимального столба по высоте
    if (Math::Geometry::Distance(firstTopPoint, firstBottomPoint)
        > Math::Geometry::Distance(secondTopPoint, secondBottomPoint))
    {
        std::swap(firstBottomPoint, secondBottomPoint);
        std::swap(firstTopPoint, secondTopPoint);
    }
    // линия по нижним точкам столбов. oт минимального столба до максимального столба
    const Math::ParametricLine botLine(firstBottomPoint, secondBottomPoint);

    //общее растояние
    float distance = 0;

    //предпологаемая ширина в метрах основанное на пропорции человека,
    const auto propWidth = humanWidthCoefficient * humanHeight;

    // расстояние до текущей точки
    auto disBotPoint = Math::Geometry::Distance(firstBottomPoint, secondBottomPoint);
    // последняя найденная высота
    auto lastHeight = Math::Geometry::Distance(firstTopPoint, firstBottomPoint);
    // средний коэффициент изменения столба по высоте(от минимального до максимального)
    double delta = 0;
    double lastH = 0;
    for (auto i = 0; i < COUNT_STEPS_FOR_DISTANCE; ++i)
    {
        auto lambda = i / static_cast<float>(COUNT_STEPS_FOR_DISTANCE);
        auto bot = botLine.GetPoint(lambda);
        auto top = GetTopPoint(bot);
        auto height = Math::Geometry::Distance(top, bot);
        if (lastH > 1)
        {
            delta += height / lastH;
        }
        lastH = height;
    }
    delta /= (COUNT_STEPS_FOR_DISTANCE - 1);
    //обратный коэффициентк к delta, изменение в противоположную (от максимального к минимальному)

    //коээфициент пути
    auto coefLengthPath = 1.f;
    // lambda для нахождения точки на прямой botLine
    float lambdaCoefficient = 0.0f;
    //расстояние от начало до конца в пикселях
    double dis = 0;
    //последняя нижняя точка
    ITV8::PointF lastBotPoint = firstBottomPoint;

    while (lambdaCoefficient <= 1)
    {
        // текущая нижняя точка
        auto bot = botLine.GetPoint(lambdaCoefficient);
        // текущая верхняя точка
        auto top = GetTopPoint(bot);
        // высота столба в пикселях
        auto height = Math::Geometry::Distance(top, bot);
        //предпологаемая ширина отступа
        auto width = height * humanWidthCoefficient;

        dis = Math::Geometry::Distance(bot, firstBottomPoint) / disBotPoint;
        coefLengthPath = dis > 1 ? float(2 - dis) : 1.f;

        const auto heightCoefficient = height / lastHeight;
        distance += float(propWidth * coefLengthPath * heightCoefficient);

        if (width == 0)
        {
            //выход из цикла в случае если width ==0 образуется бесконечный цикл
            break;
        }
        auto normWidth = width / disBotPoint;
        coefLengthPath = 1;
        lambdaCoefficient += float(normWidth * delta);
        lastBotPoint = bot;
        lastHeight = height;
    }
    return distance;
}

ITV8::PointF CalibrationRods::GetTopPoint(const ITV8::PointF& bodyPoint) const
{
    if (!IsValid())
    {
        return {-1, -1};
    }
    std::list<int> minPole;
    float avgX = 0;
    float avgY = 0;

    ITV8::PointF normalizePoint(bodyPoint.x / m_width, bodyPoint.y / m_height);

    NearestPoles(normalizePoint, minPole);
    for (auto idPole : minPole)
    {
        const Math::ParametricLine bottomLine(normalizePoint, m_levelingRods[idPole].bottom);
        auto equals(m_horizonPole[idPole].GetEquals(bottomLine));
        auto lambdas(Math::SolveSLE(equals));
        const auto tempIntersection(m_horizonPole[idPole].GetPoint(lambdas.front()));
        const Math::ParametricLine topLine(m_levelingRods[idPole].top, tempIntersection);
        const Math::ParametricLine pole(m_levelingRods[idPole].bottom, m_levelingRods[idPole].top);
        const auto p(pole.GetParallelLinePoint(normalizePoint));
        Math::ParametricLine bodyLine(normalizePoint, p);
        equals = bodyLine.GetEquals(topLine);
        lambdas = Math::SolveSLE(equals);
        const auto tempPoint = bodyLine.GetPoint(lambdas.front());
        avgX += float(tempPoint.x);
        avgY += float(tempPoint.y);
    }

    const auto n = minPole.size();
    //тк минимальное кол-во столбов при калибровке 3.
    //те возможный минимальный размер у minPole может быть 1
    //(если зона состоит из двух точек  размера кадра
    //(0,0) или (frameWidth,0) или (0,frameHeight)или (frameWidth,frameHeight)
    // и столба)
    assert(n != 0);
    avgX = avgX / static_cast<float>(n);
    avgY = avgY / static_cast<float>(n);

    return {avgX * m_width, avgY * m_height};
}

void CalibrationRods::AddRod(const ITV8::PointF& first, const ITV8::PointF& second)
{
    auto bottomPoint = first;
    auto topPoint = second;
    if (topPoint.y > bottomPoint.y)
    {
        std::swap(bottomPoint, topPoint);
    }
    m_levelingRods.emplace_back(bottomPoint, topPoint);
    m_levelingRodsChanged = true;
}

void CalibrationRods::Calibrate()
{
    if (m_levelingRodsChanged)
    {
        FindHorizontalEquation();
        SetupZone();
    }
    if (m_incorrectZone && !m_levelingRodsChanged)
    {
        SetupZone();
    }
}

void CalibrationRods::SetSize(const ITV8::Size& imgSize)
{
    if (imgSize.width != m_width || imgSize.height != m_height)
    {
        m_height = imgSize.height;
        m_width = imgSize.width;
        m_incorrectZone = true;
    }
    if (m_incorrectZone)
    {
        SetupZone();
    }
}

void CalibrationRods::NormalizeZone()
{
    for (auto& zone : m_zones)
    {
        for (auto& point : zone)
        {
            point.x /= m_width;
            point.y /= m_height;
        }
    }
}

void CalibrationRods::SetupZone()
{
    if (!IsValid())
    {
        m_incorrectZone = true;
        return;
    }
    std::vector<ITV8::PointF> bottomPoints;
    bottomPoints.reserve(m_levelingRods.size() + 4);
    for (const auto& rod : m_levelingRods)
    {
        ITV8::PointF bottomPoint(rod.bottom.x * m_width, rod.bottom.y * m_height);
        bottomPoints.emplace_back(bottomPoint);
    }
    bottomPoints.emplace_back(0, 0);
    bottomPoints.emplace_back(0, m_height);
    bottomPoints.emplace_back(m_width, 0);
    bottomPoints.emplace_back(m_width, m_height);
    m_zones = Math::Geometry::Triangulation(bottomPoints);
    NormalizeZone();
    m_incorrectZone = false;
}

void CalibrationRods::FindHorizontalEquation()
{
    if (!IsValid())
    {
        return;
    }
    const auto rodCount = static_cast<int>(m_levelingRods.size());
    m_horizonPole.clear();
    for (auto i = 0; i < rodCount; ++i)
    {
        std::vector<ITV8::PointF> allPoints; //все точки горизонта
        allPoints.reserve(rodCount);
        for (auto j = 0; j < rodCount; ++j)
        {
            if (j == i)
            {
                continue;
            }
            const Math::ParametricLine line(m_levelingRods[j].top, m_levelingRods[i].top);
            const Math::ParametricLine line2(m_levelingRods[j].bottom, m_levelingRods[i].bottom);
            auto equals(line.GetEquals(line2));
            const auto lambdas(Math::SolveSLE(equals));
            const auto horizonPoint(line.GetPoint(lambdas.front()));
            allPoints.emplace_back(horizonPoint);
        }
        // 20 погрешность относительно линии
        m_horizonPole.emplace_back(Math::Ransac2d(allPoints, 0.02f));
    }
}

int CalibrationRods::GetUpperPointPole(const ITV8::PointF& downPoint) const
{
    if (!IsValid())
    {
        return -1;
    }

    for (auto rodIt = m_levelingRods.begin(); rodIt != m_levelingRods.end(); ++rodIt)
    {
        if (rodIt->bottom == downPoint)
        {
            return std::distance(m_levelingRods.begin(), rodIt);
        }
    }
    return -1;
}

void CalibrationRods::NearestPoles(const ITV8::PointF& normalizedDownPoint, std::list<int>& indPole) const
{
    indPole.clear();
    // проход по зонам
    for (const auto& iterZone : m_zones)
    {
        // проверка вхождения в зону
        if (!Math::Geometry::IsInTriangleZone(iterZone, normalizedDownPoint))
        {
            continue;
        }
        int countPoint = 0;
        for (const auto& point : iterZone)
        {
            //проверка точки
            const auto flagBorderPoint = point.x == 0 || point.y == 0 || point.x == m_width || point.y == m_height;

            if (!flagBorderPoint)
            {
                const auto tempUp = GetUpperPointPole(point);
                if (tempUp != -1)
                {
                    indPole.push_back(tempUp);
                }
                ++countPoint;
            }
        }
        if (countPoint > 0)
        {
            break;
        }
    }
}

} // namespace CalibrationTools
} // namespace ItvCvUtils
