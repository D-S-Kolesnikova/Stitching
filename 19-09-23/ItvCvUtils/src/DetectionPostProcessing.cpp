// orig https://github.com/Nuzhny007/Non-Maximum-Suppression

#include <algorithm>
#include <ItvCvUtils//DetectionPostProcessing.h>


namespace {
constexpr float INNER_CONFIDENCE_THRESHOLD = 0.01;
}

namespace ItvCv
{
namespace Utils
{
std::vector<float> GetIntersectionRect(const std::vector<float> &rect1, const std::vector<float> &rect2)
{
    // xmin, ymin, xmax, ymax
    std::vector<float> result = {
         std::max(rect1[0], rect2[0]),
         std::max(rect1[1], rect2[1]),
         std::min(rect1[2], rect2[2]),
         std::min(rect1[3], rect2[3]) };

    if (result[2] - result[0] < 0 || result[3] - result[1] < 0)
        result = { 0,0,0,0 };

    return result;
}

float GetRectArea(const std::vector<float> &rect)
{
    // xmin, ymin, xmax, ymax
    return (rect[2] - rect[0]) * (rect[3] - rect[1]);
}

/**
 * @brief nms
 * Non maximum suppression with detection scores
 * @param detections
 * @param thresh
 * @param neighbors
 * @param minScoresSum
 */

 /**
  * Main idea is to loop through
  * existing rects and if
  * any two have overlap in IoU (intersection over union)
  * more than threshold, then
  * we suppress the one with less confidence
  * with the one with bigger confidence
  */

ITVCV_UTILS_API std::vector<int> NonMaxSuppression(
    const std::vector<std::vector<float>>& detections,
    const float thresh,
    const int neighbors,
    const float minScoresSum)
{
    // detections - inner vector format
    // {batchid, label, score, xmin, ymin, xmax, ymax}

    // resIndices is vector to return -
    // indexes of rects that are not suppressed
    std::vector<int> resIndices(0);

    const auto size = detections.size();
    if (!size)
    {
        return resIndices;
    }

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(detections[i][2], i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (!idxs.empty())
    {
        // grab the last rectangle (biggest confidence)
        const auto lastElemIter = std::rbegin(idxs);
        const auto lastElemIndex = lastElemIter->second;
        const auto rect1 = std::vector<float>{ detections[lastElemIndex].begin() + 3, detections[lastElemIndex].end() };
        const auto label1 = detections[lastElemIndex][1];

        auto neighborsCount = 0;
        auto scoresSum = lastElemIter->first;

        // delete the chosen one
        idxs.erase(std::next(lastElemIter).base());

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle (with smallest confidence)
            const auto& rect2 = std::vector<float>{ detections[pos->second].begin() + 3, detections[pos->second].end() };
            const auto label2 = detections[pos->second][1];
            // intersection
            const auto intArea = GetRectArea(GetIntersectionRect(rect1, rect2));
            // union
            const auto unionArea = GetRectArea(rect1) + GetRectArea(rect2) - intArea;
            // intersection over union
            const auto overlap = intArea / unionArea;

            // if there is sufficient overlap and they are the same class
            // suppress the current bounding box
            if (overlap > thresh && label2 == label1)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neighborsCount;
            }
            else
            {
                ++pos;
            }
        }
        // thresholding with number of neighbors and scoresSum
        if (neighborsCount >= neighbors &&
            scoresSum >= minScoresSum)
        {
            resIndices.emplace_back(lastElemIndex);
        }
    }
    return resIndices;
}

ITVCV_UTILS_API std::vector<int> FilterOutBigFalseDetection(
    const std::vector<std::vector<float>>& detections,
    const std::vector<int>& remainingIndices,
    const int maxInnerRects)
{
    // resIndices is vector to return -
    // indexes of rects that are not suppressed
    std::vector<int> goodIndices(0);

    const auto size = remainingIndices.size();
    if (!size)
    {
        return goodIndices;
    }

    std::vector<int> innerRectsNum(size, 0);
    for (size_t i = 0; i < size - 1; ++i)
    {
        const auto idxI = remainingIndices[i];
        const auto rectI = std::vector<float>{ detections[idxI].begin() + 3, detections[idxI].end() };
        const auto rectIArea = GetRectArea(rectI);
        for (size_t j = i + 1; j < size; ++j)
        {
            const auto idxJ = remainingIndices[j];
            const auto rectJ = std::vector<float>{ detections[idxJ].begin() + 3, detections[idxJ].end() };
            const auto rectJArea = GetRectArea(rectJ);

            auto smallerRectArea = rectIArea;
            auto biggerRectArea = rectJArea;
            auto biggerRectIndex = j;

            if (rectIArea > rectJArea)
            {
                smallerRectArea = rectJArea;
                biggerRectArea = rectIArea;
                biggerRectIndex = i;
            }

            const auto isSmallerCoveredByBigger = GetRectArea(GetIntersectionRect(rectI, rectJ)) / smallerRectArea > 0.95;

            if (detections[idxI][1] == detections[idxJ][1] && // if labels are the same
                isSmallerCoveredByBigger &&
                biggerRectArea > smallerRectArea * 2)
            {
                ++innerRectsNum[biggerRectIndex];
            }
        }
    }
    size_t index = 0;
    for (auto currentInnerRectsNum : innerRectsNum)
    {
        if (currentInnerRectsNum < maxInnerRects)
        {
            goodIndices.emplace_back(remainingIndices[index]);
        }
        ++index;
    }

    return goodIndices;
}

ITVCV_UTILS_API std::vector<float> CalcWindowCoordinates(const float& imageSize, const float& windowSize, const float& step)
{
    std::vector<float> result(0);
    for (auto i = 0; i < std::ceil((imageSize - windowSize) / step); ++i)
    {
        result.emplace_back(step * i / imageSize);
    }
    if (imageSize / windowSize - std::floor(imageSize / windowSize) >= 0)
    {
        result.emplace_back((imageSize - windowSize) / imageSize);
    }
    return result;
}

ITVCV_UTILS_API std::vector<std::vector<float>> ConcatenateDetections(
    std::vector<AsyncInferenceResult_t> &ssdResult,
    const size_t& width,
    const size_t& height,
    const std::pair<size_t, size_t>& windowSize,
    const std::pair<size_t, size_t>& stepSize,
    itvcvError* error)
{
    auto windowXes = ItvCv::Utils::CalcWindowCoordinates(width, windowSize.first, stepSize.first);
    auto windowYs = ItvCv::Utils::CalcWindowCoordinates(height, windowSize.second, stepSize.second);

    std::vector<std::vector<float>> resultsConcatenated(0);

    const auto windowCoefficientX = static_cast<float>(windowSize.first) / width;
    const auto windowCoefficientY = static_cast<float>(windowSize.second) / height;

    size_t windowIndexX = 0;
    size_t windowIndexY = 0;
    for (auto& res : ssdResult)
    {
        auto result = res.get();

        if (result.first != itvcvErrorSuccess)
        {
            *error = result.first;
            return resultsConcatenated;
        }

        for (auto& det : result.second)
        {
            if (det[2] < INNER_CONFIDENCE_THRESHOLD) continue;
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            std::vector<float> resultTransformed({
                det[0],
                det[1],
                det[2],
                det[3] * windowCoefficientX + windowXes[windowIndexX],
                det[4] * windowCoefficientY + windowYs[windowIndexY],
                det[5] * windowCoefficientX + windowXes[windowIndexX],
                det[6] * windowCoefficientY + windowYs[windowIndexY],
                });
            resultsConcatenated.emplace_back(resultTransformed);
        }
        ++windowIndexX;
        if (windowIndexX == windowXes.size())
        {
            ++windowIndexY;
            windowIndexX = 0;
        }
    }
    return resultsConcatenated;
}
}

}