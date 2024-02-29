#ifndef ITVCVUTILS_GENERAL_STRUCTURES_H
#define ITVCVUTILS_GENERAL_STRUCTURES_H

#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvSdk/include/baseTypes.h>

namespace ItvCvUtils
{
namespace Color
{
struct HsvColor
{
    float H = 0;
    float S = 0;
    float V = 0;
};
}
namespace Math
{
#pragma pack(push, 1)
struct SLine
{
    ITV8::PointF first;
    ITV8::PointF second;
};

struct SPolygon
{
    ITV8::PointF* points;
    int countPoint;
};

struct SPolygons
{
    SPolygon* polygons;
    int count;
};

struct SLines
{
    SLine* lines;
    int count;
};
#pragma pack(pop)
} // namespace Math
} // namespace ItvCvUtils

#endif
