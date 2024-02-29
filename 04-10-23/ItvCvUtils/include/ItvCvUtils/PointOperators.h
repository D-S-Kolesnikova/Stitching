#ifndef COMPUTERVISION_POINTOPERATORS_H
#define COMPUTERVISION_POINTOPERATORS_H

#include <ItvSdk/include/baseTypes.h>

inline ITV8::PointF operator-(const ITV8::PointF& left,
                              const ITV8::PointF& right)
{
    return {left.x - right.x, left.y - right.y};
}

inline ITV8::PointF operator-(const ITV8::PointF& left)
{
    return {-left.x, -left.y};
}

inline ITV8::PointF operator+(const ITV8::PointF& left,
                              const ITV8::PointF& right)
{
    return {left.x + right.x, left.y + right.y};
}

inline ITV8::PointF operator*(const ITV8::PointF& left,
                              const ITV8::PointF& right)
{
    return {left.x * right.x, left.y * right.y};
}

inline ITV8::PointF operator/(const ITV8::PointF& left,
                              const ITV8::PointF& right)
{
    return {left.x / right.x, left.y / right.y};
}

inline ITV8::PointF operator/(const ITV8::PointF& left, const int& right)
{
    return {left.x / right, left.y / right};
}

inline void operator-=(ITV8::PointF& left, const ITV8::PointF& right)
{
    left.x -= right.x;
    left.y -= right.y;
}

inline void operator+=(ITV8::PointF& left, const ITV8::PointF& right)
{
    left.x += right.x;
    left.y += right.y;
}

#endif // COMPUTERVISION_POINTOPERATORS_H
