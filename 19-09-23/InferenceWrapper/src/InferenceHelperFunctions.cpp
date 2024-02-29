
#include "InferenceHelperFunctions.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/constants.hpp>

#if (USE_TENSORRT_BACKEND && !__aarch64__)
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include <cctype>
#include <cmath>

namespace
{

using HpeResultType_t = InferenceWrapper::AnalyzerTraits<itvcvAnalyzerHumanPoseEstimator>::ResultType;

constexpr auto NUM_DIMS = 4;

constexpr auto YOLO_CONF_THRESHOLD = 0.1f;
constexpr auto YOLO_PREDICTIONS_NUM = 6300;

constexpr auto SSD_RESNET34_CONF_THRESHOLD = YOLO_CONF_THRESHOLD;
constexpr auto SSD_RESNET34_PREDICTIONS_NUM = 8732;

constexpr auto REID_OPTIONAL_OUTPUT_NAME = "quality_score";

constexpr auto POINT_DETECTION_OUTPUT_KPTS = "0keypoints";
constexpr auto POINT_DETECTION_OUTPUT_DESC = "0descriptors";
constexpr auto POINT_DETECTION_OUTPUT_MATCHES = "matches";
constexpr auto POINT_DETECTION_OUTPUT_SCORES = "mscores";
}

namespace InferenceWrapper
{

void ResizeProportionallyToFit(const cv::Mat& img, cv::Mat &out, const cv::Size& size)
{
    auto scale = std::min(size.width * 1.0 / img.cols, size.height * 1.0 / img.rows);
    cv::resize(img, out, cv::Size(), scale, scale);
}

void ResizeProportionallyWithConstantBorders(const cv::Mat& img, cv::Mat& out, PreprocessContext& pp)
{
    if (img.size().aspectRatio() == out.size().aspectRatio())
    {
        ResizeProportionallyToFit(img, out, out.size());
        pp.hpePaddingFromRightBottom[0] = 0;
        pp.hpePaddingFromRightBottom[1] = 0;
        return;
    }

    cv::Mat tmp;
    const cv::Mat* resizedImage;

    if ((img.size().width == out.size().width && img.size().height < out.size().height) ||
        (img.size().height == out.size().height && img.size().width < out.size().width))
    {
        // the input is already of the size, that proportional resize would return
        resizedImage = &img;
    }
    else
    {
        cv::Mat* resizeBuffer = &tmp;
        if (pp.resizeBufferCache)
        {
            resizeBuffer = pp.resizeBufferCache;
        }

        ResizeProportionallyToFit(img, *resizeBuffer, out.size());
        resizedImage = resizeBuffer;
    }

    const auto difHeight = out.rows - resizedImage->rows;
    const auto difWidth = out.cols - resizedImage->cols;
    pp.hpePaddingFromRightBottom[0] = difWidth / static_cast<float>(out.cols);
    pp.hpePaddingFromRightBottom[1] = difHeight / static_cast<float>(out.rows);

    cv::copyMakeBorder(
        *resizedImage, out, 0, difHeight, 0, difWidth, cv::BORDER_CONSTANT, cv::Scalar::all(pp.hpePaddingFillValue));
}

void ResizeProportionallyCenteredWithReplicatedBorders(const cv::Mat& in, cv::Mat& out, PreprocessContext& ctx)
{
    if (in.size().aspectRatio() == out.size().aspectRatio())
    {
        ResizeProportionallyToFit(in, out, out.size());
        return;
    }

    cv::Mat tmp;
    const cv::Mat* resizedImage;

    if ((in.size().width == out.size().width && in.size().height < out.size().height) ||
        (in.size().height == out.size().height && in.size().width < out.size().width))
    {
        // the input is already of the size, that proportional resize would return
        resizedImage = &in;
    }
    else
    {
        cv::Mat* resizeBuffer = &tmp;
        if (ctx.resizeBufferCache)
        {
            resizeBuffer = ctx.resizeBufferCache;
        }

        ResizeProportionallyToFit(in, *resizeBuffer, out.size());
        resizedImage = resizeBuffer;
    }

    const auto hPadding = out.cols - resizedImage->cols;
    const auto vPadding = out.rows - resizedImage->rows;
    const auto lPadding = hPadding / 2;
    const auto rPadding = hPadding - lPadding;
    const auto tPadding = vPadding / 2;
    const auto bPadding = vPadding - tPadding;
    cv::copyMakeBorder(
        *resizedImage, out, tPadding, bPadding, lPadding, rPadding, cv::BORDER_REPLICATE);
}

void ResizeSample(const cv::Mat& in, cv::Mat& out, PreprocessContext&)
{
    cv::resize(in, out, out.size());
}

SampleResizerFunc DefineHostResizeFunction(ItvCv::ResizePolicy& resizePolicy)
{
    switch (resizePolicy)
    {
        case ItvCv::ResizePolicy::Unspecified:
        case ItvCv::ResizePolicy::AsIs:
            return ResizeSample;
        case ItvCv::ResizePolicy::ProportionalWithRepeatedBorders:
            return ResizeProportionallyCenteredWithReplicatedBorders;
        case ItvCv::ResizePolicy::ProportionalWithPaddedBorders:
            return ResizeProportionallyWithConstantBorders;
        default:
            throw std::runtime_error("Unknown resize policy");
    }
}

#ifdef USE_INFERENCE_ENGINE_BACKEND
template<>
std::vector<AnalyzerTraits<itvcvAnalyzerClassification>::ResultType> ProcessIntelIEOutput<itvcvAnalyzerClassification>(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    const auto outputLayer = inferRequest.get_output_tensor();
    const auto outputData = outputLayer.data<float>();
    const auto resultCount = outputLayer.get_size() / batchSize;

    std::vector<AnalyzerTraits<itvcvAnalyzerClassification>::ResultType> results;
    results.reserve(batchSize);
    for (size_t i = 0; i < batchSize; ++i)
    {
        int offset = i * resultCount;
        results.emplace_back(outputData + offset, outputData + offset + resultCount);
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> ProcessIntelIEOutput<itvcvAnalyzerSSD>(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> results;

    const auto outputLayer = inferRequest.get_output_tensor();

    const auto resultCount = outputLayer.get_size() / batchSize;
    const auto outputData = outputLayer.data<float>();

    const auto outputDims = outputLayer.get_shape();

    int num_det = outputDims[2];
    int objectSize = outputDims[3]; // should be 7

    // pretty much regular SSD post-processing
    results.reserve(num_det);
    float lastImageId = -1;
    for (int i = 0; i < num_det; ++i)
    {
        int detection_pos = i * objectSize;
        float image_id = outputData[detection_pos];
        if (image_id < 0)
        {
            // indicates end of detections
            break;
        }

        if (image_id != lastImageId)
        {
            results.emplace_back();
            lastImageId = image_id;
        }
        results.back().emplace_back(outputData + detection_pos, outputData + detection_pos + SSD_RESULT_VECTOR_SIZE);
    }

    // FIXME: temporary. Either get rid of batches or rewrite to be clear with intentions
    // The calling code expects the results to contain at least one element
    if (results.empty())
    {
        results.emplace_back();
    }

    return results;
}

float ReIdGetQualityScoreIE(const boost::optional<ov::Tensor>& outputTensor, int batchPosition, int batchSize)
{
    if(
        !outputTensor
        || outputTensor.get().get_size() != batchSize)
    {
        return std::nanf("");
    }

    // offset = batchNum, because qualityScore size == 1
    return { *(outputTensor.get().data<float>() + batchPosition) };
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType> ProcessIntelIEOutput<itvcvAnalyzerSiamese>(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    using Result_t = AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType;
    const auto outputsNets = inferRequest.get_compiled_model().outputs();

    ov::Tensor featureLayer;
    boost::optional<ov::Tensor> qualityScoreLayer;

    for(auto iter = outputsNets.begin(); iter != outputsNets.end(); ++iter)
    {
        auto position = static_cast<int>(std::distance(outputsNets.begin(), iter));
        const auto& outputNames = iter->get_names();
        if(iter->get_shape()[1] == boost::get<ItvCv::ReidParams>(networkInformation->networkParams).vectorSize)
        {
            featureLayer = inferRequest.get_output_tensor(position);
        }
        else if(outputNames.find(REID_OPTIONAL_OUTPUT_NAME) != outputNames.end())
        {
            qualityScoreLayer = inferRequest.get_output_tensor(position);
        }
    }

    const auto* outputData = featureLayer.data<float>();
    const auto resultCount = featureLayer.get_size() / batchSize;

    std::vector<AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType> results;
    results.reserve(batchSize);
    for (size_t i = 0; i < batchSize; ++i)
    {
        int offset = i * resultCount;
        results.emplace_back(
            Result_t{
                {outputData + offset, outputData + offset + resultCount},
                ReIdGetQualityScoreIE(qualityScoreLayer, i, batchSize)});
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerHumanPoseEstimator>::ResultType> ProcessIntelIEOutput<
    itvcvAnalyzerHumanPoseEstimator>(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    const ov::Layout tensorOrder("NCHW");
    const auto outputsNets = inferRequest.get_compiled_model().outputs();
    int pafIndex = 0;
    int heatmapIndex = 0;

    for(auto iter = outputsNets.begin(); iter != outputsNets.end(); ++iter)
    {
        auto tests = iter->get_names();
        auto name = iter->get_any_name();
        std::transform(
            name.begin(),
            name.end(),
            name.begin(),
            [](auto c) { return std::tolower(c); });

        if(name.find("paf") != std::string::npos
            || name.find("tags") != std::string::npos)
        {
            pafIndex = std::distance(outputsNets.begin(), iter);
        }
        else if (name.find("heatmap") != std::string::npos)
        {
            heatmapIndex = std::distance(outputsNets.begin(), iter);
        }
    }

    const auto pafsTensor = inferRequest.get_output_tensor(pafIndex);
    const auto heatmapsTensor = inferRequest.get_output_tensor(heatmapIndex);

    const auto& dimsHeatmap = heatmapsTensor.get_shape();
    const auto& dimsPaf = pafsTensor.get_shape();
    std::vector<HpeResultType_t> results;
    results.reserve(batchSize);
    const auto batchIndex = ov::layout::batch_idx(tensorOrder);
    const auto channelIndex = ov::layout::channels_idx(tensorOrder);
    const auto heightIndex = ov::layout::height_idx(tensorOrder);
    const auto widthIndex = ov::layout::width_idx(tensorOrder);

    results.emplace_back(
        HpeResultType_t(
            ItvCvUtils::Math::Tensor(
                {
                    static_cast<int>(dimsHeatmap[batchIndex]),
                    static_cast<int>(dimsHeatmap[channelIndex]),
                    static_cast<int>(dimsHeatmap[heightIndex]),
                    static_cast<int>(dimsHeatmap[widthIndex])},
                std::vector<float>(heatmapsTensor.data<float>(), heatmapsTensor.data<float>() + heatmapsTensor.get_size()),
                ItvCvUtils::Math::DimsOrder::NCHW),
            ItvCvUtils::Math::Tensor(
                {
                    static_cast<int>(dimsPaf[batchIndex]),
                    static_cast<int>(dimsPaf[channelIndex]),
                    static_cast<int>(dimsPaf[heightIndex]),
                    static_cast<int>(dimsPaf[widthIndex])},
                std::vector<float>(pafsTensor.data<float>(), pafsTensor.data<float>() + pafsTensor.get_size()),
                ItvCvUtils::Math::DimsOrder::NCHW),
            { pp.hpePaddingFromRightBottom[0], pp.hpePaddingFromRightBottom[1] }));
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerMaskSegments>::ResultType> ProcessIntelIEOutput<itvcvAnalyzerMaskSegments>(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    const auto& outputTensor = inferRequest.get_output_tensor();
    auto dims = outputTensor.get_shape();

    std::vector<AnalyzerTraits<itvcvAnalyzerMaskSegments>::ResultType> results;
    results.emplace_back();
    results.back().dims = std::vector<int>(dims.begin(), dims.end());
    results.back().data = std::vector<float>(outputTensor.data<float>(), outputTensor.data<float>() + outputTensor.get_size());
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerPointDetection>::ResultType> ProcessIntelIEOutput<itvcvAnalyzerPointDetection>(
    ov::InferRequest inferRequest,
    const std::size_t batchSize,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    throw std::logic_error("Unimplemented");
}
#endif

#ifdef USE_TENSORRT_BACKEND
#ifndef __aarch64__
void GpuResizeProportionallyToFit(const cv::cuda::GpuMat& img, cv::cuda::GpuMat &out, const cv::Size& size, cv::cuda::Stream& stream)
{
    auto scale = std::min(size.width * 1.0 / img.cols, size.height * 1.0 / img.rows);
    cv::cuda::resize(img, out, cv::Size(), scale, scale, cv::InterpolationFlags::INTER_LINEAR, stream);
}

void GpuResizeProportionallyCenteredWithReplicatedBorders(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& out, PreprocessContext& ctx, cv::cuda::Stream& stream)
{
    if (in.size().aspectRatio() == out.size().aspectRatio())
    {
        GpuResizeProportionallyToFit(in, out, out.size(), stream);
        return;
    }

    cv::cuda::GpuMat resizedImage;

    if ((in.size().width == out.size().width && in.size().height < out.size().height) ||
        (in.size().height == out.size().height && in.size().width < out.size().width))
    {
        // the input is already of the size, that proportional resize would return
        resizedImage = in;
    }
    else
    {
        GpuResizeProportionallyToFit(in, resizedImage, out.size(), stream);
    }

    const auto hPadding = out.cols - resizedImage.cols;
    const auto vPadding = out.rows - resizedImage.rows;
    const auto lPadding = hPadding / 2;
    const auto rPadding = hPadding - lPadding;
    const auto tPadding = vPadding / 2;
    const auto bPadding = vPadding - tPadding;
    cv::cuda::copyMakeBorder(
        resizedImage, out, tPadding, bPadding, lPadding, rPadding, cv::BORDER_REPLICATE, cv::Scalar(), stream);
}

void GpuResizeProportionallyWithConstantBorders(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& out, PreprocessContext& ctx, cv::cuda::Stream& stream)
{
    if (in.size().aspectRatio() == out.size().aspectRatio())
    {
        GpuResizeProportionallyToFit(in, out, out.size(), stream);
        ctx.hpePaddingFromRightBottom[0] = 0;
        ctx.hpePaddingFromRightBottom[1] = 0;
        return;
    }

    cv::cuda::GpuMat resizedImage;

    if ((in.size().width == out.size().width && in.size().height < out.size().height) ||
        (in.size().height == out.size().height && in.size().width < out.size().width))
    {
        // the input is already of the size, that proportional resize would return
        resizedImage = in;
    }
    else
    {
        GpuResizeProportionallyToFit(in, resizedImage, out.size(), stream);
    }

    const auto rPadding = out.cols - resizedImage.cols;
    const auto bPadding = out.rows - resizedImage.rows;
    cv::cuda::copyMakeBorder(
        resizedImage, out, 0, bPadding, 0, rPadding, cv::BORDER_CONSTANT, cv::Scalar::all(ctx.hpePaddingFillValue), stream);
}

void GpuResizeSample(const cv::cuda::GpuMat& in, cv::cuda::GpuMat& out, PreprocessContext&, cv::cuda::Stream& stream)
{
    cv::cuda::resize(in, out, out.size(), 1, 1, cv::InterpolationFlags::INTER_LINEAR, stream);
}

GpuSampleResizerFunc DefineGpuResizeFunction(ItvCv::ResizePolicy& resizePolicy)
{
    switch (resizePolicy)
    {
        case ItvCv::ResizePolicy::Unspecified:
        case ItvCv::ResizePolicy::AsIs:
            return GpuResizeSample;
        case ItvCv::ResizePolicy::ProportionalWithRepeatedBorders:
            return GpuResizeProportionallyCenteredWithReplicatedBorders;
        case ItvCv::ResizePolicy::ProportionalWithPaddedBorders:
            return GpuResizeProportionallyWithConstantBorders;
        default:
            throw std::runtime_error("Unknown resize policy");
    }
}
#endif

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerClassification>::ResultType> ProcessTensorRTOutput<itvcvAnalyzerClassification>(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    const auto outputData = static_cast<const float*>(bufMgr.GetHostBuffer(outputLayerName));
    const auto outputLayerSize = bufMgr.size(outputLayerName);
    const auto resultCount = outputLayerSize / batchSize;

    std::vector<AnalyzerTraits<itvcvAnalyzerClassification>::ResultType> results;
    results.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i)
    {
        std::size_t offset = i * resultCount;
        results.emplace_back(outputData + offset, outputData + offset + resultCount);
    }
    return results;
}


std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> ProcessTensorRTOutputYOLO(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName)
{
    std::vector<int> numDetData = { YOLO_PREDICTIONS_NUM };

    const auto outputData = static_cast<const float*>(bufMgr.GetHostBuffer(outputLayerName));
    const auto batchVolume = bufMgr.size(outputLayerName) / batchSize;

    const auto resultVectorSize = batchVolume / numDetData[0];
    const auto classesNum = resultVectorSize - 4;

    std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> results;
    results.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i)
    {
        std::vector<cv::Rect_<float>> rects;
        std::vector<float> scores;
        std::vector<float> labels;

        auto batchResultsIterator = outputData;
        std::advance(batchResultsIterator, i * batchVolume);

        auto scoresArrayIterator = batchResultsIterator;
        std::advance(scoresArrayIterator, 4);

        for (std::size_t j = 0; j < numDetData[i]; ++j)
        {
            // converting Yolo output format to SSD
            // Yolo : [xc, yc, width, height, class_confs(without noise) ......]
            // SSD  : [image_id, label, score, xmin, ymin, xmax, ymax]
            auto xc = batchResultsIterator[0];
            auto yc = batchResultsIterator[1];
            auto width = batchResultsIterator[2];
            auto height = batchResultsIterator[3];
            float xmin = xc - width / 2.;
            float ymin = yc - height / 2.;


            const auto maxElem = std::max_element(scoresArrayIterator, scoresArrayIterator + classesNum);
            auto score = *maxElem;
            auto label = static_cast<int>(std::distance(scoresArrayIterator, maxElem)) + 1;

            //threshold by confidence
            if (score > YOLO_CONF_THRESHOLD)
            {
                rects.emplace_back(xmin, ymin, width, height);
                labels.emplace_back(label);
                scores.emplace_back(score);
            }
            std::advance(scoresArrayIterator, resultVectorSize);
            std::advance(batchResultsIterator, resultVectorSize);
        }

        results.emplace_back();
        results.back().reserve(rects.size());
        for (size_t i = 0; i < rects.size(); ++i)
        {
            const auto& rect = rects[i];
            results.back().emplace_back(std::initializer_list<float>{1., labels[i], scores[i], rect.x, rect.y, rect.x + rect.width, rect.y + rect.height});
        }
    }
    return results;
}

std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> ProcessTensorRTOutputSSDResNet34(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName)
{
    std::vector<int> numDetData = { SSD_RESNET34_PREDICTIONS_NUM };

    const auto outputData = static_cast<const float*>(bufMgr.GetHostBuffer(outputLayerName));
    const auto batchVolume = bufMgr.size(outputLayerName) / batchSize;

    const auto resultVectorSize = batchVolume / numDetData[0];

    // without noise
    const auto classesNum = resultVectorSize - 4 - 1;

    std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> results;
    results.reserve(batchSize);
    for (std::size_t batchNum = 0; batchNum < batchSize; ++batchNum)
    {
        std::size_t offset = batchNum * batchVolume;

        std::vector<std::vector<float>> rects;
        std::vector<float> scores;
        std::vector<float> labels;

        auto batchResultsIterator = outputData;
        std::advance(batchResultsIterator, batchNum * batchVolume);

        auto scoresArrayIterator = batchResultsIterator;
        std::advance(scoresArrayIterator, 4);

        // for ssd_resnet34 class confs contain noise
        // and also class confs are normalized by softmax
        // so their sum == 1

        for (size_t j = 0; j < numDetData[batchNum]; ++j)
        {
            // converting SSD_RESNET34 output format to SSD
            // SSD_RESNET34 : [x1, y1, x2, y2, class_confs(with noise!) ......]
            // SSD_MOBILENET  : [image_id, label, score, xmin, ymin, xmax, ymax]

            const auto maxElem = std::max_element(scoresArrayIterator + 1, scoresArrayIterator + classesNum + 1);
            auto score = *maxElem;
            auto label = static_cast<int>(std::distance(scoresArrayIterator, maxElem));

            //threshold by confidence and skip noise
            if (label != 0 && score > SSD_RESNET34_CONF_THRESHOLD)
            {
                rects.emplace_back(batchResultsIterator, batchResultsIterator + 4);
                labels.emplace_back(label);
                scores.emplace_back(score);
            }
            std::advance(scoresArrayIterator, resultVectorSize);
            std::advance(batchResultsIterator, resultVectorSize);
        }
        results.emplace_back();
        results.back().reserve(rects.size());
        for (size_t i = 0; i < rects.size(); ++i)
        {
            const auto& rect = rects[i];
            results.back().emplace_back(std::initializer_list<float>{1., labels[i], scores[i], rect[0], rect[1], rect[2], rect[3]});
        }
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> ProcessTensorRTOutput<itvcvAnalyzerSSD>(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    if (networkInformation->commonParams.architecture == ItvCv::ArchitectureType::Yolo)
    {
        return ProcessTensorRTOutputYOLO(bufMgr, batchSize, outputLayerName);
    }
    if (networkInformation->commonParams.architecture == ItvCv::ArchitectureType::SSD_ResNet34)
    {
        return ProcessTensorRTOutputSSDResNet34(bufMgr, batchSize, outputLayerName);
    }

    const auto numDetData = static_cast<const int*>(bufMgr.GetHostBuffer("keep_count"));

    const auto outputData = static_cast<const float*>(bufMgr.GetHostBuffer(outputLayerName));
    const auto batchVolume = bufMgr.size(outputLayerName) / batchSize;

    std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> results;
    results.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i)
    {
        std::size_t offset = i * batchVolume;
        results.emplace_back();
        results.back().reserve(numDetData[i]);
        for (std::size_t j = 0; j < numDetData[i]; ++j)
        {
            results.back().emplace_back(outputData + offset, outputData + offset + SSD_RESULT_VECTOR_SIZE);
            offset += SSD_RESULT_VECTOR_SIZE;
        }
    }
    return results;
}
//holder outputData first - data ptr, second - size
using ReIdOutputData_t = std::pair<float*, std::uint64_t>;

float ReIdGetQualityScoreRT(const boost::optional<ReIdOutputData_t>& qualityData, int batchPosition, int batchSize)
{
    if(
        !qualityData
        || qualityData.get().second != batchSize)
    {
        return std::nanf("");
    }
    auto* dataPtr = qualityData.get().first;

    // offset = batchNum, because qualityScore size == 1
    return { *(dataPtr + batchPosition) };
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType> ProcessTensorRTOutput<itvcvAnalyzerSiamese>(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    using Result_t = AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType;

    boost::optional<ReIdOutputData_t> qualityScoreOutputData;
    ReIdOutputData_t featureOutputData;

    const auto featuresSize = boost::get<ItvCv::ReidParams>(networkInformation->networkParams).vectorSize;

    std::vector<std::string> names;
    boost::split(names, outputLayerName, boost::algorithm::is_any_of(","));
    for(const auto& nodeName: names)
    {
        auto* dataPtr = static_cast<float*>(bufMgr.GetHostBuffer(nodeName));
        const auto elementSize = bufMgr.size(nodeName) / batchSize;
        if(static_cast<int>(elementSize)  == featuresSize)
        {
            featureOutputData = { dataPtr, elementSize };
        }
        else if(nodeName == REID_OPTIONAL_OUTPUT_NAME)
        {
            qualityScoreOutputData = {{ dataPtr, elementSize }};
        }
    }

    std::vector<AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType> results;
    results.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i)
    {
        // add image result
        auto offset = i * featureOutputData.second;
        auto* featureDataPtr = featureOutputData.first;

        results.emplace_back(
            Result_t{
                { featureDataPtr + offset, featureDataPtr + offset + featureOutputData.second },
                ReIdGetQualityScoreRT(qualityScoreOutputData, i, batchSize) });
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerHumanPoseEstimator>::ResultType> ProcessTensorRTOutput<
    itvcvAnalyzerHumanPoseEstimator>(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    std::vector<std::string> namesOut;
    std::string firstName;
    std::string secondName;
    boost::split(namesOut, outputLayerName, boost::algorithm::is_any_of(","));
    if (namesOut.size() % 2 != 0)
        return {{}};
    else
    {
        int lastValidId = namesOut.size() - 1;
        firstName = namesOut[lastValidId - 1];
        secondName = namesOut[lastValidId];
        if (firstName.find("L1") == std::string::npos && firstName.find("paf") == std::string::npos)
        {
            firstName.swap(secondName);
        }
    }

    const auto outputDataPaf = static_cast<const float*>(bufMgr.GetHostBuffer(firstName));
    const auto outputDataHeatmap = static_cast<const float*>(bufMgr.GetHostBuffer(secondName));
    const auto dimsPaf = bufMgr.GetDims(firstName);
    const auto dimsHeatmap = bufMgr.GetDims(secondName);

    const auto sizePaf = bufMgr.size(firstName) / batchSize;
    const auto sizeHeatmap = bufMgr.size(secondName) / batchSize;

    std::vector<HpeResultType_t> results;
    results.reserve(batchSize);
    for (int i = 0; i < batchSize; ++i)
    {
        HpeResultType_t rawData;
        if (dimsHeatmap.nbDims == 3)
        {
            rawData = {
                ItvCvUtils::Math::Tensor(
                    { 1, dimsHeatmap.d[0], dimsHeatmap.d[1], dimsHeatmap.d[2] },
                    std::vector<float>(outputDataHeatmap + sizeHeatmap * i, outputDataHeatmap + sizeHeatmap * (i + 1)),
                    ItvCvUtils::Math::DimsOrder::NCHW),
                ItvCvUtils::Math::Tensor(
                    { 1, dimsPaf.d[0], dimsPaf.d[1], dimsPaf.d[2] },
                    std::vector<float>(outputDataPaf + sizePaf * i, outputDataPaf + sizePaf * (i + 1)),
                    ItvCvUtils::Math::DimsOrder::NCHW),
                { pp.hpePaddingFromRightBottom[0], pp.hpePaddingFromRightBottom[1] }};
        }
        else
        {
            rawData = {
                ItvCvUtils::Math::Tensor(
                    { dimsHeatmap.d[0], dimsHeatmap.d[1], dimsHeatmap.d[2], dimsHeatmap.d[3] },
                    std::vector<float>(outputDataHeatmap + sizeHeatmap * i, outputDataHeatmap + sizeHeatmap * (i + 1)),
                    ItvCvUtils::Math::DimsOrder::NCHW),
                ItvCvUtils::Math::Tensor(
                    { dimsPaf.d[0], dimsPaf.d[1], dimsPaf.d[2], dimsPaf.d[3] },
                    std::vector<float>(outputDataPaf + sizePaf * i, outputDataPaf + sizePaf * (i + 1)),
                    ItvCvUtils::Math::DimsOrder::NCHW),
                { pp.hpePaddingFromRightBottom[0], pp.hpePaddingFromRightBottom[1] }};
        }
        results.emplace_back(std::move(rawData));
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerMaskSegments>::ResultType> ProcessTensorRTOutput<itvcvAnalyzerMaskSegments>(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    const auto outputData = static_cast<const float*>(bufMgr.GetHostBuffer(outputLayerName));
    const auto outputLayerSize = bufMgr.size(outputLayerName);
    const auto dimsOut = bufMgr.GetDims(outputLayerName);
    const auto resultCount = outputLayerSize / batchSize;

    std::vector<AnalyzerTraits<itvcvAnalyzerMaskSegments>::ResultType> results;
    results.emplace_back();
    results.back().dims = std::vector<int>(dimsOut.d, dimsOut.d + dimsOut.nbDims);
    results.back().data = std::vector<float>(outputData, outputData + outputLayerSize);

    return results;
}

using PointDetectionData_t = InferenceWrapper::AnalyzerTraits<itvcvAnalyzerPointDetection>::ResultType;

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerPointDetection>::ResultType> ProcessTensorRTOutput<itvcvAnalyzerPointDetection>(
    const CBufferManager& bufMgr,
    const int& batchSize,
    const std::string& outputLayerName,
    const PreprocessContext& pp,
    const ItvCv::PNetworkInformation& networkInformation)
{
    PointDetectionData_t featureOutputData;

    std::vector<std::string> names;
    boost::split(names, outputLayerName, boost::algorithm::is_any_of(","));
    for (const auto& nodeName : names)
    {
        auto* dataPtr = static_cast<float*>(bufMgr.GetHostBuffer(nodeName));
        const auto elementSize = bufMgr.size(nodeName);
        auto elementDims = bufMgr.GetDims(nodeName);
        std::vector<int64_t> dimsArray;
        dimsArray.resize(elementDims.nbDims);
        for (int i = 0; i < elementDims.nbDims; i++)
        {
            dimsArray[i] = elementDims.d[i];
        }
        if (boost::algorithm::contains(nodeName, POINT_DETECTION_OUTPUT_KPTS))
        {
            uint64_t id =(*(nodeName.end() - 1)) - '0';
            featureOutputData.keypoints[id] = { {dataPtr, dataPtr + elementSize}, dimsArray };
        }
        else if (boost::algorithm::contains(nodeName, POINT_DETECTION_OUTPUT_DESC))
        {
            uint64_t id = (*(nodeName.end() - 1)) - '0';
            featureOutputData.descriptors[id] = { {dataPtr, dataPtr + elementSize}, dimsArray };
        }
        else if (boost::algorithm::contains(nodeName, POINT_DETECTION_OUTPUT_MATCHES))
        {
            uint64_t id = (*(nodeName.end() - 1)) - '0';
            featureOutputData.matches[id] = { {dataPtr, dataPtr + elementSize}, dimsArray };
        }
        else if (boost::algorithm::contains(nodeName, POINT_DETECTION_OUTPUT_SCORES))
        {
            uint64_t id = (*(nodeName.end() - 1)) - '0';
            featureOutputData.scores[id] = { {dataPtr, dataPtr + elementSize}, dimsArray };
        }
    }

    /*for (auto i = 0; i < 2; i++)
    {
        std::string nameOutput = POINT_DETECTION_OUTPUT_KPTS + std::to_string(i);
        auto outputDataKpts = static_cast<float*>(bufMgr.GetHostBuffer(nameOutput));
        auto dimsKpts = bufMgr.GetDims(nameOutput);
        std::vector<int> dimsArray;
        dimsArray.resize(dimsKpts.nbDims);
        for (int i = 0; i < dimsKpts.nbDims; i++)
        {
            dimsArray[i] = dimsKpts.d[i];
        }
        featureOutputData.keypoints[i] = { outputDataKpts , dimsArray };
        nameOutput = POINT_DETECTION_OUTPUT_DESC + std::to_string(i);
        auto outputDataDesc = static_cast<float*>(bufMgr.GetHostBuffer(nameOutput));
        dimsKpts = bufMgr.GetDims(nameOutput);
        dimsArray.resize(dimsKpts.nbDims);
        for (int i = 0; i < dimsKpts.nbDims; i++)
        {
            dimsArray[i] = dimsKpts.d[i];
        }
        featureOutputData.descriptors[i] = { outputDataDesc, dimsArray };
        nameOutput = POINT_DETECTION_OUTPUT_MATCHES + std::to_string(i);
        auto outputDataMatch = static_cast<float*>(bufMgr.GetHostBuffer(nameOutput));
        dimsKpts = bufMgr.GetDims(nameOutput);
        dimsArray.resize(dimsKpts.nbDims);
        for (int i = 0; i < dimsKpts.nbDims; i++)
        {
            dimsArray[i] = dimsKpts.d[i];
        }
        featureOutputData.matches[i] = { outputDataMatch, dimsArray };
    }*/
    std::vector<PointDetectionData_t> results;
    results.emplace_back(std::move(featureOutputData));

    /*const auto outputDataKpts = static_cast<const float*>(bufMgr.GetHostBuffer("0keypoints0"));
    auto dimsKpts = bufMgr.GetDims("keypoints");
    int kpSize = dimsKpts.d[1];
    featureOutputData.keypoints.resize(kpSize);
    for (int i = 0; i < kpSize; i++)
    {
        cv::KeyPoint p;
        int index = i * 2;
        p.pt.x = ((float)outputDataKpts[index]);
        p.pt.y = ((float)outputDataKpts[index + 1]);
        featureOutputData.keypoints[i] = p;
    }
    const auto outputDataDesc = static_cast<const float*>(bufMgr.GetHostBuffer("descriptors"));
    auto dimsDesc = bufMgr.GetDims("descriptors");
    cv::Mat desmat;
    desmat.create(cv::Size(dimsDesc.d[2], dimsDesc.d[1]), CV_32FC1);
    for (int h = 0; h < dimsDesc.d[1]; h++)
    {
        for (int w = 0; w < dimsDesc.d[2]; w++)
        {
            int index = h * dimsDesc.d[2] + w;
            desmat.at<float>(h, w) = outputDataDesc[index];
        }
    }
    desmat.copyTo(featureOutputData.descriptors);*/

    return results;
}
#endif


#ifdef USE_ATLAS300_BACKEND
ResultData::Blob const* find_blob(std::vector<ResultData::Blob> const& blobs, std::initializer_list<const char*> patterns)
{
    for (auto const& b : blobs)
    {
        for (auto pattern : patterns)
            if (std::string::npos != b.name.find(pattern))
                return &b;
    }
    return nullptr;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerClassification>::ResultType> ProcessAtlas300Output<itvcvAnalyzerClassification>(
    const ResultData& res,
    const PreprocessContext&,
    ItvCv::PNetworkInformation const&)
{
    auto blobs = res.GetBlobs();
    auto prob = find_blob(blobs, {"prob"});
    if (!prob)
    {
        return {};
    }

    std::vector<AnalyzerTraits<itvcvAnalyzerClassification>::ResultType> results;
    results.emplace_back(prob->data.first, prob->data.first + prob->data.second);

    return results;
}

std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> ProcessAtlas300OutputSSDResNet34(
    const ResultData& res)
{
    auto blobs = res.GetBlobs();
    if (blobs.empty())
        return {};

    auto const& out = blobs[0];
    const auto outputData = out.data.first;
    auto const batchSize = 1;
    auto predictionsNum = out.h;
    auto resultVectorSize = out.w;
    if (0 == resultVectorSize || 0 == predictionsNum)
    {
        predictionsNum = out.dims[0];
        resultVectorSize = out.dims[1];
    }
    assert(predictionsNum == SSD_RESNET34_PREDICTIONS_NUM);
    const auto batchVolume = out.data.second / batchSize;

    // without noise
    const auto classesNum = resultVectorSize - 4 - 1;

    std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> results;
    results.reserve(out.n);
    for (std::size_t batchNum = 0; batchNum < batchSize; ++batchNum)
    {
        std::vector<std::vector<float>> rects;
        std::vector<float> scores;
        std::vector<float> labels;

        auto batchResultsIterator = outputData;
        std::advance(batchResultsIterator, batchNum * batchVolume);

        auto scoresArrayIterator = batchResultsIterator;
        std::advance(scoresArrayIterator, 4);

        // for ssd_resnet34 class confs contain noise
        // and also class confs are normalized by softmax
        // so their sum == 1

        for (std::size_t j = 0; j < predictionsNum; ++j)
        {
            // converting SSD_RESNET34 output format to SSD
            // SSD_RESNET34 : [x1, y1, x2, y2, class_confs(with noise!) ......]
            // SSD_MOBILENET  : [image_id, label, score, xmin, ymin, xmax, ymax]

            const auto maxElem = std::max_element(scoresArrayIterator + 1, scoresArrayIterator + classesNum + 1);
            auto score = *maxElem;
            auto label = static_cast<int>(std::distance(scoresArrayIterator, maxElem));

            //threshold by confidence and skip noise
            if (label != 0 && score > SSD_RESNET34_CONF_THRESHOLD)
            {
                rects.emplace_back(batchResultsIterator, batchResultsIterator + 4);
                labels.emplace_back(label);
                scores.emplace_back(score);
            }
            std::advance(scoresArrayIterator, resultVectorSize);
            std::advance(batchResultsIterator, resultVectorSize);
        }

        results.emplace_back();
        results.back().reserve(rects.size());
        for (size_t i = 0; i < rects.size(); ++i)
        {
            const auto& rect = rects[i];
            results.back().emplace_back(std::initializer_list<float>{1., labels[i], scores[i], rect[0], rect[1], rect[2], rect[3]});
        }
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> ProcessAtlas300Output<itvcvAnalyzerSSD>(
    const ResultData& res,
    const PreprocessContext&,
    ItvCv::PNetworkInformation const& net)
{
    if (net->commonParams.architecture == ItvCv::ArchitectureType::SSD_ResNet34)
    {
        return ProcessAtlas300OutputSSDResNet34(res);
    }

    auto blobs = res.GetBlobs();
    if (blobs.size() < 2)
        return {};
    auto const* out_bbox_count = find_blob(blobs, {"detection_out_0"});
    if (!out_bbox_count)
    {
        return {};
    }

    auto const* out_bbox_data = find_blob(blobs, {"detection_out_2"});
    if (!out_bbox_data)
    {
        return {};
    }

    auto actual_ssd_result_vector_size = out_bbox_data->w ? out_bbox_data->w : out_bbox_data->dims[2];
    if (actual_ssd_result_vector_size < SSD_RESULT_VECTOR_SIZE)
    {
        assert(false && "Unexpected shape for SSD bbox data output");
        return {};
    }

    int bbox_count = static_cast<int>(*(out_bbox_count->data.first));
    if (bbox_count < 0)
    {
        return {};
    }

    int bbox_buffer_size = bbox_count * actual_ssd_result_vector_size;
    if (bbox_buffer_size > static_cast<int>(out_bbox_data->data.second))
    {
        assert(false && "SSD bbox data is too short for reported bbox_count");
        return {};
    }

    std::vector<AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> results;
    results.emplace_back();
    results.back().reserve(bbox_count);

    for (int bbox_pos = 0; bbox_pos < bbox_buffer_size; bbox_pos += actual_ssd_result_vector_size)
    {
        results.back().emplace_back(out_bbox_data->data.first + bbox_pos, out_bbox_data->data.first + bbox_pos + SSD_RESULT_VECTOR_SIZE);
    }

    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerHumanPoseEstimator>::ResultType> ProcessAtlas300Output<
    itvcvAnalyzerHumanPoseEstimator>(const ResultData& res, const PreprocessContext& pp,
    ItvCv::PNetworkInformation const&)
{
    const int batchSize = 1;
    auto blobs = res.GetBlobs();
    if (blobs.size() < 2)
        return {};
    auto const* pafs = find_blob(blobs, {"L1", "paf"});
    auto const* heatmaps = find_blob(blobs, {"L2"});
    if (!pafs || !heatmaps)
    {
        return {};
    }

    std::vector<HpeResultType_t> results;
    results.reserve(batchSize);

    for (int i = 0; i < batchSize; ++i)
    {
        results.emplace_back(HpeResultType_t(
            ItvCvUtils::Math::Tensor(
                { heatmaps->w, heatmaps->h, heatmaps->c, 1 },
                std::vector<float>(heatmaps->data.first, heatmaps->data.first + heatmaps->data.second),
                ItvCvUtils::Math::DimsOrder::WHCN),
            ItvCvUtils::Math::Tensor(
                { pafs->w, pafs->h, pafs->c, 1 },
                std::vector<float>(pafs->data.first, pafs->data.first + pafs->data.second),
                ItvCvUtils::Math::DimsOrder::WHCN),
            { pp.hpePaddingFromRightBottom[0], pp.hpePaddingFromRightBottom[1] }));
    }
    return results;
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerSiamese>::ResultType> ProcessAtlas300Output<itvcvAnalyzerSiamese>(
        const ResultData&, const PreprocessContext&,
        ItvCv::PNetworkInformation const&)
{
    throw std::logic_error("Unimplemented");
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerMaskSegments>::ResultType> ProcessAtlas300Output<itvcvAnalyzerMaskSegments>(
        const ResultData&, const PreprocessContext&,
        ItvCv::PNetworkInformation const&)
{
    throw std::logic_error("Unimplemented");
}

template<>
std::vector<AnalyzerTraits<itvcvAnalyzerPointDetection>::ResultType> ProcessAtlas300Output<itvcvAnalyzerPointDetection>(
    const ResultData& res,
    const PreprocessContext&,
    ItvCv::PNetworkInformation const&)
{
    throw std::logic_error("Unimplemented");
}
#endif

} // namespace InferenceWrapper
