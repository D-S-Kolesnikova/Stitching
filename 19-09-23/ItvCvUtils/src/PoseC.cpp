
#include <ItvCvUtils/PoseC.h>
#include <ItvCvUtils/Pose.h>

#include <tuple>

namespace ItvCvUtils
{
void PoseFunction::BoundingBox(const PoseC& outPoses, double maxWidth, double maxHeight, ITV8::RectangleF& returnRectangle, double paddingX, double paddingY)
{
    returnRectangle = Pose(outPoses).BoundingBox(maxWidth, maxHeight, paddingX, paddingY);
}

float PoseFunction::GetHumanHeight(const PoseC& outPoses)
{
    return Pose(outPoses).GetHumanHeight();
}

void PoseFunction::GetBottomMiddlePoint(const PoseC& outPoses, ITV8::PointF& returnPoint)
{
    returnPoint = Pose(outPoses).GetBottomMiddlePoint();
}

PointAndType PoseFunction::GetBottomMiddlePointAndType(const PoseC& outPoses)
{
    PointAndType result;
    std::tie(result.point, result.type) = Pose(outPoses).GetBottomMiddlePointAndType();
    return result;
}

void PoseFunction::GetBodyCenter(const PoseC& outPoses, ITV8::PointF& returnPoint)
{
    returnPoint = Pose(outPoses).GetBodyCenter();
}

void PoseFunction::GetMiddlePointByType(
    const PoseC& outPoses,
    BottomPosePointType type,
    ITV8::PointF& returnPoint)
{
    returnPoint = Pose(outPoses).GetMiddlePointByType(type);
}

bool PoseFunction::HasLegs(
    const PoseC& outPoses,
    const BodyPartOrientation orient)
{
    return Pose(outPoses).HasLegs(orient);
}

bool PoseFunction::HasHands(
    const PoseC& outPoses,
    const BodyPartOrientation orient)
{
    return Pose(outPoses).HasHands(orient);
}

bool PoseFunction::HasBody(const PoseC& outPoses)
{
    return Pose(outPoses).HasBody();
}

bool PoseFunction::IsValid(const PoseC& outPoses)
{
    return Pose(outPoses).IsValid();
}
}
