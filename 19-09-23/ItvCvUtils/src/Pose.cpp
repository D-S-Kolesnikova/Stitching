
#include <ItvCvUtils/Pose.h>

#include <ItvCvUtils/PointOperators.h>

#include <ItvDetectorSdk/include/DetectorConstants.h>
#include <boost/algorithm/string.hpp>

namespace
{
float Distance(const ITV8::PointF& point)
{
    return std::sqrt(static_cast<float>(point.x * point.x + point.y * point.y));
}

float Distance(const ITV8::PointF& first, const ITV8::PointF& second)
{
    return Distance(first - second);
}
} // namespace

namespace ItvCvUtils
{
RawPoseData::RawPoseData(
    ItvCvUtils::Math::Tensor&& heatmap,
    ItvCvUtils::Math::Tensor&& paf,
    std::pair<float, float>&& padding)
    : heatmap(heatmap)
    , paf(paf)
    , padding(padding)
{}

RawPoseData::RawPoseData(
    const ItvCvUtils::Math::Tensor& heatmap,
    const ItvCvUtils::Math::Tensor& paf,
    const std::pair<float, float>& padding)
    : RawPoseData(
        ItvCvUtils::Math::Tensor(heatmap),
        ItvCvUtils::Math::Tensor(paf),
        std::pair<float, float>(padding))
{}

Pose::Pose(std::vector<ITV8::PointF> keypoints)
: m_keypoints(std::move(keypoints))
{
    if (!IsValidPoint(this->neck()))
    {
        FindNeckByShoulder();
    }
}

Pose::Pose(const PoseC& pose)
    : m_keypoints(COUNTS_POSE_KEYPOINT, {-1,-1})
{
    for(size_t i = 0; i < m_keypoints.size(); ++i)
    {
        m_keypoints[i] = pose.keypointData[i];
    }

    if(!IsValidPoint(this->neck()))
    {
        FindNeckByShoulder();
    }
}

PoseC Pose::GetDataC() const
{
    PoseC result;
    std::copy(m_keypoints.begin(), m_keypoints.end(), result.keypointData);
    return result;
}

ITV8::RectangleF Pose::BoundingBox() const
{
    auto left = std::numeric_limits<double>::max();
    auto top = std::numeric_limits<double>::max();
    auto right = 0.0;
    auto bottom = 0.0;

    for (const auto& point : m_keypoints)
    {
        if (point.x >= 0 && point.y >= 0)
        {
            left = std::min(left, point.x);
            top = std::min(top, point.y);
            right = std::max(right, point.x);
            bottom = std::max(bottom, point.y);
        }
    }

    return ITV8::RectangleF{ left, top, right - left, bottom - top};
}

ITV8::RectangleF Pose::BoundingBox(double maxWidth, double maxHeight, double paddingX, double paddingY) const
{
    const auto bbox = BoundingBox();

    paddingX *= bbox.right() - bbox.left;
    paddingY *= bbox.bottom() - bbox.top;

    const auto left = std::max(0., bbox.left - paddingX);
    const auto right = std::min(maxWidth, bbox.right() + paddingX);
    const auto top = std::max(0., bbox.top - paddingY);
    const auto bottom = std::min(maxHeight, bbox.bottom() + paddingY);

    return ITV8::RectangleF {left, top, right - left, bottom - top};
}

bool Pose::IsValid() const
{
    return std::any_of(
        m_keypoints.begin(), m_keypoints.end(), [](const ITV8::PointF& p) { return p.x >= 0 && p.y >= 0; });
}

float Pose::GetHumanHeight() const
{
    ITV8::PointF predPoint(-1, -1);
    float returnValue = 0;

    for (auto typeEnd : {BottomPosePointType::Ankle, BottomPosePointType::Knee, BottomPosePointType::Hip})
    {
        ITV8::PointF leftPoint(-1, -1);
        ITV8::PointF rightPoint(-1, -1);
        std::tie(leftPoint, rightPoint) = GetLeftRightPoints(typeEnd);
        ITV8::PointF currentPoint(-1, -1);
        auto leftExist = IsValidPoint(leftPoint);
        auto rightExist = IsValidPoint(rightPoint);

        if (leftExist && rightExist)
        {
            currentPoint = GetMiddlePointByType(typeEnd);
        }
        else if (leftExist)
        {
            currentPoint = leftPoint;
        }
        else if (rightExist)
        {
            currentPoint = rightPoint;
        }
        if (IsValidPoint(predPoint))
        {
            returnValue += Distance(predPoint, currentPoint);
        }
        predPoint = currentPoint;
    }
    if (IsValidPoint(predPoint))
    {
        returnValue += Distance(predPoint, neck());
    }
    return returnValue;
}

void Pose::FindNeckByShoulder()
{
    const auto leftShoulderExist = IsValidPoint(this->leftShoulder());
    const auto rightShoulderExist = IsValidPoint(this->rightShoulder());
    if(
        leftShoulderExist
        && rightShoulderExist)
    {
        this->neck() = (this->leftShoulder() + this->rightShoulder()) / 2;
    }
}

// return left,right point by type
std::pair<ITV8::PointF, ITV8::PointF> Pose::GetLeftRightPoints(const BottomPosePointType& type) const
{
    switch (type)
    {
    case BottomPosePointType::Ankle:
        return {leftAnkle(), rightAnkle()};
    case BottomPosePointType::Knee:
        return {leftKnee(), rightKnee()};
    case BottomPosePointType::Hip:
        return {leftHip(), rightHip()};
    default:
        return {{-1, -1}, {-1, -1}};
    }
}

std::pair<bool, bool> Pose::CheckValidLeftRightPoints(const ITV8::PointF& leftPoint, const ITV8::PointF& rightPoint)
    const
{
    return {IsValidPoint(leftPoint), IsValidPoint(rightPoint)};
}

ITV8::PointF Pose::GetBodyCenter() const
{
    if (HasBody())
    {
        return (leftShoulder() + rightShoulder() + leftHip() + rightHip()) / 4;
    }

    return {-1, -1};
}

ITV8::PointF Pose::GetMiddlePointByType(BottomPosePointType type) const
{
    ITV8::PointF returnPoint(-1, -1);
    switch (type)
    {
    case BottomPosePointType::Hip:
        if (IsValidPoint(leftHip()) && IsValidPoint(rightHip()))
            return (leftHip() + rightHip()) / 2;
    case BottomPosePointType::Ankle:
        if (IsValidPoint(leftAnkle()) && IsValidPoint(rightAnkle()))
            return (leftAnkle() + rightAnkle()) / 2;
    case BottomPosePointType::Knee:
        if (IsValidPoint(leftKnee()) && IsValidPoint(rightKnee()))
            return (leftKnee() + rightKnee()) / 2;
    default:
        break;
    }
    return returnPoint;
}

bool Pose::HasBody() const
{
    std::vector<size_t> bodyPointIds {1, 8, 11};
    const auto isValidPoints = std::all_of(
        bodyPointIds.begin(), bodyPointIds.end(), [this](size_t i) { return IsValidPoint(m_keypoints[i]); });
    const auto existShoulder = (IsValidPoint(m_keypoints[2]) || IsValidPoint(m_keypoints[5]));
    return isValidPoints && existShoulder;
}

bool Pose::HasLegs(const BodyPartOrientation orientation) const
{
    std::vector<int> legsPointIds;
    switch (orientation)
    {
    case BodyPartOrientation::Both:
        legsPointIds = {8, 9, 10, 11, 12, 13};
        break;
    case BodyPartOrientation::Left:
        legsPointIds = {11, 12, 13};
        break;
    case BodyPartOrientation::Right:
        legsPointIds = {8, 9, 10};
        break;
    }
    return std::all_of(
        legsPointIds.begin(), legsPointIds.end(), [this](size_t i) { return IsValidPoint(m_keypoints[i]); });
}

bool Pose::HasHands(const BodyPartOrientation orientation) const
{
    std::vector<int> handKeyPointIds;
    switch (orientation)
    {
    case BodyPartOrientation::Both:
        handKeyPointIds = {2, 3, 4, 5, 6, 7};
        break;
    case BodyPartOrientation::Left:
        handKeyPointIds = {5, 6, 7};
        break;
    case BodyPartOrientation::Right:
        handKeyPointIds = {2, 3, 4};
        break;
    }
    return std::all_of(
        handKeyPointIds.begin(), handKeyPointIds.end(), [this](size_t i) { return IsValidPoint(m_keypoints[i]); });
}

ITV8::PointF Pose::GetBottomMiddlePoint() const
{
    return GetBottomMiddlePointAndType().first;
}

std::pair<ITV8::PointF, BottomPosePointType> Pose::GetBottomMiddlePointAndType() const
{
    if (IsValidPoint(leftAnkle()) && IsValidPoint(rightAnkle()))
    {
        return std::make_pair((leftAnkle() + rightAnkle()) / 2, BottomPosePointType::Ankle);
    }
    else if (IsValidPoint(leftKnee()) && IsValidPoint(rightKnee()))
    {
        return std::make_pair((leftKnee() + rightKnee()) / 2, BottomPosePointType::Knee);
    }
    else if (IsValidPoint(leftHip()) && IsValidPoint(rightHip()))
    {
        return std::make_pair((leftHip() + rightHip()) / 2, BottomPosePointType::Hip);
    }
    else
    {
        return std::make_pair(ITV8::PointF(-1, -1), BottomPosePointType::Undef);
    }
}
}
