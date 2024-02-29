#ifndef DetectionPostProcessing_hpp
#define DetectionPostProcessing_hpp

#include "ItvCvUtils.h"
#include "NeuroAnalyticsApi.h"

#if defined(BOOST_THREAD_FUTURE_HPP) && !(defined(BOOST_THREAD_PROVIDES_FUTURE) && defined(BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION) && defined(BOOST_THREAD_PROVIDES_EXECUTORS))
#error "boost/thread/future.hpp should be included only after defining BOOST_THREAD_PROVIDES_FUTURE, BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION, BOOST_THREAD_PROVIDES_EXECUTORS."
#endif
#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_EXECUTORS
#include <boost/thread/future.hpp>

#include <map>
#include <vector>

namespace ItvCv
{
namespace Utils
{
    using ResultType = std::vector<std::vector<float>>;
    using InferenceResult_t = std::pair<itvcvError, ResultType>;
    using AsyncInferenceResult_t = boost::future<InferenceResult_t>;

    std::vector<float> GetIntersectionRect(
        const std::vector<float> &rect1,
        const std::vector<float> &rect2);

    float GetRectArea(const std::vector<float> &rect);

    ITVCV_UTILS_API std::vector<int> NonMaxSuppression(
        const std::vector<std::vector<float>>& detections,
        const float thresh = 0.4f,
        const int neighbors = 0,
        const float minScoresSum = 0.f);

    ITVCV_UTILS_API std::vector<int> FilterOutBigFalseDetection(
        const std::vector<std::vector<float>>& detections,
        const std::vector<int>& remainingIndices,
        const int maxInnerRects = 3);

    ITVCV_UTILS_API std::vector<float> CalcWindowCoordinates(
        const float& imageSize,
        const float& windowSize,
        const float& step);

    ITVCV_UTILS_API std::vector<std::vector<float>> ConcatenateDetections(
        std::vector<AsyncInferenceResult_t> &ssdResult,
        const size_t& width,
        const size_t& height,
        const std::pair<size_t, size_t>& windowSize,
        const std::pair<size_t, size_t>& stepSize,
        itvcvError* error);
}
}

#endif //DetectionPostProcessing_hpp