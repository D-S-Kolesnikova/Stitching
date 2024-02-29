#ifndef COMPUTERVISION_POSE_H
#define COMPUTERVISION_POSE_H

#include <ItvCvUtils/PoseC.h>
#include <ItvCvUtils/ItvCvUtils.h>
#include <ItvCvUtils/Math.h>

#include <ItvSdk/include/baseTypes.h>

#include <vector>
#include <string>

namespace ItvCvUtils
{

constexpr auto COUNTS_POSE_KEYPOINT = 18;

struct ITVCV_UTILS_API RawPoseData
{
    RawPoseData() = default;

    RawPoseData(
        const Math::Tensor& heatmap,
        const Math::Tensor& paf,
        const std::pair<float, float>& padding);

    RawPoseData(
        Math::Tensor&& heatmap,
        Math::Tensor&& paf,
        std::pair<float, float>&& padding);

    Math::Tensor heatmap;
    Math::Tensor paf;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
    std::pair<float, float> padding;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
};

struct ITVCV_UTILS_API Pose
{
    Pose(): m_keypoints(18, {-1, -1}) {}
    Pose(std::vector<ITV8::PointF> keypoints);
    Pose(const PoseC& pose);

    const std::vector<ITV8::PointF>& keypoints() const
    {
        return m_keypoints;
    }

    std::vector<ITV8::PointF>& keypoints()
    {
        return m_keypoints;
    }

    ITV8::PointF& nose()
    {
        return m_keypoints[0];
    }
    ITV8::PointF& neck()
    {
        return m_keypoints[1];
    }
    ITV8::PointF& rightShoulder()
    {
        return m_keypoints[2];
    }
    ITV8::PointF& rightElbow()
    {
        return m_keypoints[3];
    }
    ITV8::PointF& rightWrist()
    {
        return m_keypoints[4];
    }
    ITV8::PointF& leftShoulder()
    {
        return m_keypoints[5];
    }
    ITV8::PointF& leftElbow()
    {
        return m_keypoints[6];
    }
    ITV8::PointF& leftWrist()
    {
        return m_keypoints[7];
    }
    ITV8::PointF& rightHip()
    {
        return m_keypoints[8];
    }
    ITV8::PointF& rightKnee()
    {
        return m_keypoints[9];
    }
    ITV8::PointF& rightAnkle()
    {
        return m_keypoints[10];
    }
    ITV8::PointF& leftHip()
    {
        return m_keypoints[11];
    }
    ITV8::PointF& leftKnee()
    {
        return m_keypoints[12];
    }
    ITV8::PointF& leftAnkle()
    {
        return m_keypoints[13];
    }
    ITV8::PointF& leftEye()
    {
        return m_keypoints[14];
    }
    ITV8::PointF& rightEye()
    {
        return m_keypoints[15];
    }
    ITV8::PointF& rightEar()
    {
        return m_keypoints[16];
    }
    ITV8::PointF& leftEar()
    {
        return m_keypoints[17];
    }

    const ITV8::PointF& nose() const
    {
        return m_keypoints[0];
    }
    const ITV8::PointF& neck() const
    {
        return m_keypoints[1];
    }
    const ITV8::PointF& rightShoulder() const
    {
        return m_keypoints[2];
    }
    const ITV8::PointF& rightElbow() const
    {
        return m_keypoints[3];
    }
    const ITV8::PointF& rightWrist() const
    {
        return m_keypoints[4];
    }
    const ITV8::PointF& leftShoulder() const
    {
        return m_keypoints[5];
    }
    const ITV8::PointF& leftElbow() const
    {
        return m_keypoints[6];
    }
    const ITV8::PointF& leftWrist() const
    {
        return m_keypoints[7];
    }
    const ITV8::PointF& rightHip() const
    {
        return m_keypoints[8];
    }
    const ITV8::PointF& rightKnee() const
    {
        return m_keypoints[9];
    }
    const ITV8::PointF& rightAnkle() const
    {
        return m_keypoints[10];
    }
    const ITV8::PointF& leftHip() const
    {
        return m_keypoints[11];
    }
    const ITV8::PointF& leftKnee() const
    {
        return m_keypoints[12];
    }
    const ITV8::PointF& leftAnkle() const
    {
        return m_keypoints[13];
    }
    const ITV8::PointF& leftEye() const
    {
        return m_keypoints[14];
    }
    const ITV8::PointF& rightEye() const
    {
        return m_keypoints[15];
    }
    const ITV8::PointF& rightEar() const
    {
        return m_keypoints[16];
    }
    const ITV8::PointF& leftEar() const
    {
        return m_keypoints[17];
    }

    PoseC GetDataC() const;

    ITV8::RectangleF BoundingBox() const;

    ITV8::RectangleF BoundingBox(double maxWidth, double maxHeight, double paddingX = 0.1, double paddingY = 0.1) const;

    float GetHumanHeight() const;
    ITV8::PointF GetBottomMiddlePoint() const;
    std::pair<ITV8::PointF, BottomPosePointType> GetBottomMiddlePointAndType() const;
    ITV8::PointF GetBodyCenter() const;
    ITV8::PointF GetMiddlePointByType(BottomPosePointType type) const;
    bool HasLegs(const BodyPartOrientation orient) const;
    bool HasHands(const BodyPartOrientation orient) const;
    bool HasBody() const;
    bool IsValid() const;

    static bool IsValidPoint(const ITV8::PointF& p)
    {
        return p.x >= 0 && p.y >= 0;
    }

private:
    void FindNeckByShoulder();
    std::pair<ITV8::PointF, ITV8::PointF> GetLeftRightPoints(const BottomPosePointType& type) const;
    std::pair<bool, bool> CheckValidLeftRightPoints(const ITV8::PointF& leftPoint, const ITV8::PointF& rightPoint)
        const;

private:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
    std::vector<ITV8::PointF> m_keypoints;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
};
}

#endif // COMPUTERVISION_POSE_H
