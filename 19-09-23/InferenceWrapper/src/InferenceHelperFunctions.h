#ifndef INFERENCEHELPERFUNCTIONS_H_
#define INFERENCEHELPERFUNCTIONS_H_

#include <opencv2/opencv.hpp>
#include <functional>

#ifdef USE_INFERENCE_ENGINE_BACKEND
#include <ie/ie_common.h>
#include <openvino/openvino.hpp>
#endif
#ifdef USE_TENSORRT_BACKEND
#include "TensorRT/Buffers.h"
#ifndef __aarch64__
#include <opencv2/cudaimgproc.hpp>
#endif
#endif
#ifdef USE_ATLAS300_BACKEND
#include <HuaweiAscend/Ascend.h>
#endif

#include <InferenceWrapper/AnalyzerTraits.h>
#include <NetworkInformation/NetworkInformationLib.h>

namespace InferenceWrapper
{
struct PreprocessContext
{
    std::vector<int> widths;
    std::vector<int> heights;
    float hpePaddingFillValue;
    cv::Vec2f hpePaddingFromRightBottom;
    ItvCv::PoseNetworkParams hpeParams;
    cv::Mat* resizeBufferCache = nullptr;
};

typedef std::function<void(const cv::Mat&, cv::Mat&, PreprocessContext&)> SampleResizerFunc;
SampleResizerFunc DefineHostResizeFunction(ItvCv::ResizePolicy& resizePolicy);

#ifdef USE_INFERENCE_ENGINE_BACKEND
template<itvcvAnalyzerType analyzerType>
typename std::vector<typename AnalyzerTraits<analyzerType>::ResultType> ProcessIntelIEOutput(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation);
#endif

#ifdef USE_TENSORRT_BACKEND
#ifndef __aarch64__
typedef std::function<void(const cv::cuda::GpuMat&, cv::cuda::GpuMat&, PreprocessContext&, cv::cuda::Stream&)> GpuSampleResizerFunc;
GpuSampleResizerFunc DefineGpuResizeFunction(ItvCv::ResizePolicy& resizePolicy);
#endif

template<itvcvAnalyzerType analyzerType>
std::vector<typename AnalyzerTraits<analyzerType>::ResultType> ProcessTensorRTOutput(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation);
#endif

#ifdef USE_ATLAS300_BACKEND
template<itvcvAnalyzerType analyzerType>
typename std::vector<typename AnalyzerTraits<analyzerType>::ResultType> ProcessAtlas300Output(
    const ResultData& res,
    PreprocessContext const&,
    ItvCv::PNetworkInformation const& net);
#endif

} // namespace InferenceWrapper
#endif
