#ifndef COMPUTERVISION_POSE_C_STYLE_H
#define COMPUTERVISION_POSE_C_STYLE_H

#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvSdk/include/baseTypes.h>

#include <utility>

#define ITVCV_UTILS_API_C extern "C" ITVCV_UTILS_API

namespace ItvCvUtils
{
enum BottomPosePointType
{
    Hip,
    Knee,
    Ankle,
    Undef
};
enum BodyPartOrientation
{
    Left = 0,
    Right = 1,
    Both = 2
};

#pragma pack(push, 1)
struct PoseC
{
    ITV8::PointF keypointData[18];

    ITV8::PointF Nose() const
    {
        return keypointData[0];
    }
    const ITV8::PointF& Neck() const
    {
        return keypointData[1];
    }
    const ITV8::PointF& RightShoulder() const
    {
        return keypointData[2];
    }
    const ITV8::PointF& RightElbow() const
    {
        return keypointData[3];
    }
    const ITV8::PointF& RightWrist() const
    {
        return keypointData[4];
    }
    const ITV8::PointF& LeftShoulder() const
    {
        return keypointData[5];
    }
    const ITV8::PointF& LeftElbow() const
    {
        return keypointData[6];
    }
    const ITV8::PointF& LeftWrist() const
    {
        return keypointData[7];
    }
    const ITV8::PointF& RightHip() const
    {
        return keypointData[8];
    }
    const ITV8::PointF& RightKnee() const
    {
        return keypointData[9];
    }
    const ITV8::PointF& RightAnkle() const
    {
        return keypointData[10];
    }
    const ITV8::PointF& LeftHip() const
    {
        return keypointData[11];
    }
    const ITV8::PointF& LeftKnee() const
    {
        return keypointData[12];
    }
    const ITV8::PointF& LeftAnkle() const
    {
        return keypointData[13];
    }
    const ITV8::PointF& LeftEye() const
    {
        return keypointData[14];
    }
    const ITV8::PointF& RightEye() const
    {
        return keypointData[15];
    }
    const ITV8::PointF& RightEar() const
    {
        return keypointData[16];
    }
    const ITV8::PointF& LeftEar() const
    {
        return keypointData[17];
    }
};

struct PointAndType
{
    BottomPosePointType type;
    ITV8::PointF point;
};

namespace PoseFunction
{
    ITVCV_UTILS_API_C void BoundingBox(const PoseC& outPoses, double maxWidth, double maxHeight, ITV8::RectangleF& returnRectangle, double paddingX = 0.1, double paddingY = 0.1);

    ITVCV_UTILS_API_C float GetHumanHeight(const PoseC& outPoses);

    ITVCV_UTILS_API_C void GetBottomMiddlePoint(const PoseC& outPoses, ITV8::PointF& returnPoint);

    ITVCV_UTILS_API_C PointAndType GetBottomMiddlePointAndType(const PoseC& outPoses);

    ITVCV_UTILS_API_C void GetBodyCenter(const PoseC& outPoses, ITV8::PointF& returnPoint);

    ITVCV_UTILS_API_C void GetMiddlePointByType(const PoseC& outPoses, BottomPosePointType type, ITV8::PointF& returnPoint);

    ITVCV_UTILS_API_C bool HasLegs(const PoseC& outPoses, const BodyPartOrientation orient);

    ITVCV_UTILS_API_C bool HasHands(const PoseC& outPoses, const BodyPartOrientation orient);

    ITVCV_UTILS_API_C bool HasBody(const PoseC& outPoses);

    ITVCV_UTILS_API_C bool IsValid(const PoseC& outPoses);
};
#pragma pack(pop)
}
#endif // COMPUTERVISION_POSE_H
