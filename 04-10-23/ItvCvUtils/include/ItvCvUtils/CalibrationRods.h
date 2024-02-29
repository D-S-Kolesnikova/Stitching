#ifndef ITVCVUTILS_CALIBRATION_RODS_H
#define ITVCVUTILS_CALIBRATION_RODS_H

#include <list>
#include <vector>

#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvSdk/include/baseTypes.h>
#include <ItvCvUtils/GeneralStructuresC.h>
#include <ItvCvUtils/Math.h>

namespace ItvCvUtils
{
struct Pose;
}

namespace ItvCvUtils
{
namespace CalibrationTools
{
class ITVCV_UTILS_API CalibrationRods
{
    struct SLevelingRod
    {
        SLevelingRod() = default;
        SLevelingRod(const ITV8::PointF& bottom, const ITV8::PointF& top): bottom(bottom), top(top) {}

        ITV8::PointF bottom;
        ITV8::PointF top;
    };

public:
    ~CalibrationRods() = default;
    CalibrationRods() = default;
    CalibrationRods(const std::vector<Math::SLine>& rodsRaw, const ITV8::Size& imgSize);
    CalibrationRods(const CalibrationRods& rods);
    void Clear();
    // add normalized rod  with coordinates in range  (0..1)
    double GetDistanceToPoint(ITV8::PointF& firstBottomPoint, ITV8::PointF& secondBottomPoint, double humanWidthCoefficient, double humanHeight) const;

    void AddRod(const ITV8::PointF& first, const ITV8::PointF& second);
    void Calibrate();
    void SetSize(const ITV8::Size& imgSize);

    bool IsValid() const
    {
        return m_levelingRods.size() >= 3 && m_width > 0 && m_height > 0;
    };

    ITV8::PointF GetTopPoint(const ITV8::PointF& bottomPoint) const;

    const std::vector<SLevelingRod>& GetRods() const
    {
        return m_levelingRods;
    }

    const std::vector<std::vector<ITV8::PointF>>& GetZones() const
    {
        return m_zones;
    }

private: // function
    void NormalizeZone();
    void SetupZone();
    void FindHorizontalEquation();
    int GetUpperPointPole(const ITV8::PointF& normalizedDownPoint) const;
    void NearestPoles(const ITV8::PointF& normalizedDownPoint, std::list<int>& indPole) const;

private: // attribute
    bool m_levelingRodsChanged = false;
    bool m_incorrectZone = true;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
    std::vector<std::vector<ITV8::PointF>> m_zones;
    // normalize leveling rods
    std::vector<SLevelingRod> m_levelingRods;
    std::vector<ItvCvUtils::Math::ParametricLine> m_horizonPole;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    int32_t m_width = 0;
    int32_t m_height = 0;
};

} // namespace CalibrationTools
} // namespace ItvCvUtils

#endif
