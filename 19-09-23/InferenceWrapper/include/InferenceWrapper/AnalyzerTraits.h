#ifndef COMPUTERVISION_ANALYZERTRAITS_H
#define COMPUTERVISION_ANALYZERTRAITS_H


#include <ItvCvUtils/NeuroAnalyticsApi.h>
#include <ItvCvUtils/ItvCvDefs.h>
#include <ItvCvUtils/Pose.h>
#include <ItvCvUtils/PointDetection.h>

#include <InferenceWrapper/InferenceWrapperLib.h>

#if defined(BOOST_THREAD_FUTURE_HPP) && !(defined(BOOST_THREAD_PROVIDES_FUTURE) && defined(BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION) && defined(BOOST_THREAD_PROVIDES_EXECUTORS))
#error "boost/thread/future.hpp should be included only after defining BOOST_THREAD_PROVIDES_FUTURE, BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION, BOOST_THREAD_PROVIDES_EXECUTORS."
#endif
#define BOOST_THREAD_PROVIDES_FUTURE
#define BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION
#define BOOST_THREAD_PROVIDES_EXECUTORS
#include <boost/thread/future.hpp>

#include <vector>

namespace InferenceWrapper
{

template<itvcvAnalyzerType analyzerType>
struct AnalyzerTraits
{
    using ResultType = int;
    static const char* Name();
};

template<>
struct AnalyzerTraits<itvcvAnalyzerClassification>
{
    using ResultType = std::vector<float>;
    static constexpr const char* Name() { return "Classification"; }
};

template<>
struct AnalyzerTraits<itvcvAnalyzerSSD>
{
    using ResultType = std::vector<std::vector<float>>;
    static constexpr const char* Name() { return "SSD"; }
};

template<>
struct AnalyzerTraits<itvcvAnalyzerHumanPoseEstimator>
{
    using ResultType = ItvCvUtils::RawPoseData;
    static constexpr const char* Name() { return "PoseEstimator"; }
};

template<>
struct AnalyzerTraits<itvcvAnalyzerMaskSegments>
{
    struct MaskData
    {
        std::vector<int> dims;
        std::vector<float> data;
    };

    using ResultType = MaskData;
    static constexpr const char* Name() { return "MaskSegments"; }
};

template<>
struct AnalyzerTraits<itvcvAnalyzerSiamese>
{
    struct ReIdOutput
    {
        std::vector<float> features;
        float qualityScore;
    };

    using ResultType = ReIdOutput;
    static constexpr const char* Name() { return "Siamese"; }
};

template<>
struct AnalyzerTraits<itvcvAnalyzerPointDetection>
{
    using ResultType = PointDetection::PointDetectionData;
    static constexpr const char* Name() { return "PointDetection"; }
};

template<itvcvAnalyzerType analyzerType>
struct InferenceResultTraits
{
    using ResultType = typename AnalyzerTraits<analyzerType>::ResultType;
    using InferenceResult_t = std::pair<itvcvError, ResultType>;
    using AsyncInferenceResult_t = boost::future<InferenceResult_t>;
};
}
#endif
