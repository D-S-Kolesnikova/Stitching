#include <InferenceWrapper/InferenceWrapperLib.h>
#include <InferenceWrapper/InferenceEngine.h>

#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/DynamicThreadPool.h>
#include <ItvCvUtils/DetectionPostProcessing.h>
#include <NetworkInformation/NetworkInformationLib.h>
#include <HpePostProcessing/HpePostProcessing.h>

#include <boost/asio.hpp>
// WARNING: don't delete the following include as it is necessary for inline_executor on linux
#include <boost/thread/sync_bounded_queue.hpp>
#include <boost/thread/executors/inline_executor.hpp>
#include <boost/range/combine.hpp>

#include <cstring>
#include <memory>

namespace
{

std::vector<std::vector<float>> ApplyNMS(const std::vector<std::vector<float>>& ssdResults)
{
    std::vector<std::vector<float>> filteredResults(0);
    auto indices = ItvCv::Utils::NonMaxSuppression(ssdResults);
    for (const auto& idx : indices)
    {
        filteredResults.emplace_back(ssdResults[idx]);
    }

    return filteredResults;
}

class CombinedPoses
{
public:
    CombinedPoses(const std::vector<ItvCvUtils::Pose>& poses): m_poses(poses)
    {
    }

    std::vector<ItvCvUtils::PoseC> GetView()
    {
        std::vector<ItvCvUtils::PoseC> resultVector;
        resultVector.reserve(m_poses.size());
        for(const auto& pose : m_poses)
        {
            resultVector.emplace_back(pose.GetDataC());
        }

        return resultVector;
    }

private:
    const std::vector<ItvCvUtils::Pose>& m_poses;
};

void warpCountPeople(int* const OutConteiner, std::vector<int> inConteiner)
{
    for (int i = 0; static_cast<size_t>(i) < inConteiner.size(); ++i)
        OutConteiner[i] = inConteiner.at(i);
};

void warpPosePoints(ItvCvUtils::PoseC** const outConteiner, std::vector<std::vector<ItvCvUtils::Pose>> inConteiner)
{
    for (int k = 0; static_cast<size_t>(k) < inConteiner.size(); ++k)
    {
        std::map<int, float*> COCO_Key_Points;
        int i = 0;
        for (const auto& pose : inConteiner[k])
        {
            outConteiner[k][i] = pose.GetDataC();
            ++i;
        }
    }
};
}

namespace InferenceWrapper
{

struct ICachingInferenceEngine
{
    virtual ~ICachingInferenceEngine() = default;
    virtual ITV8::Size GetInputGeometry() = 0;
    virtual void TakeStats(std::chrono::milliseconds period) = 0;
};

template<itvcvAnalyzerType analyzerType>
class CachingInferenceEngine: public ICachingInferenceEngine
{
public:
    using ResultType = typename AnalyzerTraits<analyzerType>::ResultType;
    struct BatchInferenceResult
    {
        itvcvError error;
        std::vector<ResultType> results;
        std::vector<ITV8::Size> imgSizes;
        std::vector<std::vector<ItvCvUtils::Pose>> processedPoses;
    };

    using AsyncInferenceResult_t = typename InferenceResultTraits<analyzerType>::AsyncInferenceResult_t;

    CachingInferenceEngine(
        itvcvError& error,
        const EngineCreationParams& engineParams,
        const InferenceChannelParams& channelParams,
        int resultVectorSize)
        : m_engine(IInferenceEngine<analyzerType>::Create(error, engineParams, channelParams))
        , m_resultVectorSize(resultVectorSize)
        , m_net(engineParams.net)
    {
    }

    const BatchInferenceResult& ProcessFrames(
        const int nFrames,
        const int* widths,
        const int* heights,
        const int* strides,
        const unsigned char** framesDataRGB)
    {
        m_lastBatchResult.error = itvcvErrorSuccess;
        m_lastBatchResult.results.clear();
        m_lastBatchResult.processedPoses.clear();
        m_lastBatchResult.imgSizes.clear();

        std::vector<AsyncInferenceResult_t> asyncResults;
        for (int i = 0; i < nFrames; ++i)
        {
            asyncResults.emplace_back(AsyncProcessFrame(widths[i], heights[i], strides[i], framesDataRGB[i]));
        }

        for (int i = 0; i < nFrames; ++i)
        {
            auto result = asyncResults[i].get();
            if (result.first != itvcvErrorSuccess)
            {
                m_lastBatchResult.error = result.first;
                m_lastBatchResult.results.clear();

                //TODO временное решение
                if (m_net->commonParams.architecture == ItvCv::ArchitectureType::Openpose18_MobileNet)
                {
                    m_lastBatchResult.processedPoses.clear();
                }
                m_lastBatchResult.imgSizes.clear();
                //end
                return m_lastBatchResult;
            }
            m_lastBatchResult.imgSizes.emplace_back(widths[i], heights[i]);
            m_lastBatchResult.results.emplace_back(std::move(result.second));
        }
        return m_lastBatchResult;
    }

    BatchInferenceResult& ProcessSubFrame(
        const int width,
        const int height,
        const int stride,
        const unsigned char* frameDataRGB,
        const std::pair<ITV8::int32_t, ITV8::int32_t> window,
        const std::pair<ITV8::int32_t, ITV8::int32_t> steps)
    {
        m_lastBatchSubResults.error = itvcvErrorSuccess;
        m_lastBatchSubResults.results.clear();
        m_lastBatchSubResults.processedPoses.clear();
        m_lastBatchSubResults.imgSizes.clear();

        if (analyzerType != itvcvAnalyzerType::itvcvAnalyzerSSD)
        {
            m_lastBatchSubResults.error = itvcvErrorOther;
            return m_lastBatchSubResults;
        }

        std::vector<AsyncInferenceResult_t> asyncSubResults;
        asyncSubResults = std::move(AsyncProcessSubFrame(
            width,
            height,
            stride,
            frameDataRGB,
            window,
            steps));
        
        auto error = itvcvErrorSuccess;
        auto resultsConcatenated = ItvCv::Utils::ConcatenateDetections(
            asyncSubResults,
            width,
            height,
            window,
            steps,
            &error);

        m_lastBatchSubResults.error = error;
        if (m_lastBatchSubResults.error == itvcvErrorSuccess)
        {
            m_lastBatchSubResults.results.emplace_back(std::move(resultsConcatenated));
        }
        return m_lastBatchSubResults;
    }

    const BatchInferenceResult& GetLastBatchSubResults() const
    {
        return m_lastBatchSubResults;
    }

    const BatchInferenceResult& GetLastBatchResult() const
    {
        return m_lastBatchResult;
    }

    AsyncInferenceResult_t AsyncProcessFrame(
        const int width,
        const int height,
        const int stride,
        const unsigned char* frameDataRGB)
    {
        return m_engine->AsyncProcessFrame(ItvCv::Frame(width, height, stride, frameDataRGB));
    }

    std::vector<AsyncInferenceResult_t> AsyncProcessSubFrame(
        const int width,
        const int height,
        const int stride,
        const unsigned char* frameDataRGB,
        const std::pair<ITV8::int32_t, ITV8::int32_t> window,
        const std::pair<ITV8::int32_t, ITV8::int32_t> steps)
    {
       return m_engine->AsyncProcessSubFrame(ItvCv::Frame(width, height, stride, frameDataRGB), window, steps);
    }

    ITV8::Size GetInputGeometry()
    {
        return m_engine->GetInputGeometry();
    }

    int GetResultVectorSize() const
    {
        return m_resultVectorSize;
    }

    void TakeStats(std::chrono::milliseconds period) override
    {
        return m_engine->TakeStats(period);
    }

    //TODO временное решение
    ItvCv::PNetworkInformation GetNet() const
    {
        return m_net;
    }

    ~CachingInferenceEngine() override = default;

private:
    using PInferenceEngine = std::unique_ptr<IInferenceEngine<analyzerType>>;
    PInferenceEngine m_engine;
    BatchInferenceResult m_lastBatchResult;
    BatchInferenceResult m_lastBatchSubResults;
    int m_resultVectorSize;
    ItvCv::PNetworkInformation m_net;
};

template<typename ResultProcessor, typename ResultType>
void HandleAsyncResult(
    itvcvError& readyError,
    ResultProcessor&& resultProcessor,
    boost::future<std::pair<itvcvError, ResultType>>&& resultFuture)
{
    if (resultFuture.is_ready())
    {
        auto result = resultFuture.get();
        readyError = result.first;

        if (readyError == itvcvErrorSuccess)
        {
            resultProcessor(std::move(result));
        }
    }
    else
    {
        readyError = itvcvErrorSuccess;
        static boost::executors::inline_executor executor;
        resultFuture.then(
            executor,
            [resultProcessor](auto&& resultFuture)
            {
                resultProcessor(resultFuture.get());
            });
    }
}

} // InferenceWrapper

// ##########################################
// >>>>>>>>>>>>>>>>>> CREATION AND DELETION
// ##########################################

inline InferenceWrapper::ICachingInferenceEngine* createAnalyzer(
    itvcvAnalyzerType netType,
    itvcvError& error,
    const InferenceWrapper::EngineCreationParams& engineParams,
    const InferenceWrapper::InferenceChannelParams& channelParams,
    int resultVectorSize)
{
    switch (netType)
    {
    case itvcvAnalyzerClassification:
        return new InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerClassification>(error, engineParams, channelParams, resultVectorSize);
    case itvcvAnalyzerSSD:
        return new InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerSSD>(error, engineParams, channelParams, resultVectorSize);
    case itvcvAnalyzerSiamese:
        return new InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerSiamese>(error, engineParams, channelParams, resultVectorSize);
    case itvcvAnalyzerHumanPoseEstimator:
        return new InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerHumanPoseEstimator>(error, engineParams, channelParams, resultVectorSize);
    case itvcvAnalyzerMaskSegments:
        return new InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerMaskSegments>(error, engineParams, channelParams, resultVectorSize);
    case itvcvAnalyzerPointDetection:
        return new InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerPointDetection>(error, engineParams, channelParams, resultVectorSize);
    default:
        break;
    }
    throw std::logic_error("Unknown analyzer type");
}

template <itvcvAnalyzerType netType> inline
InferenceWrapper::CachingInferenceEngine<netType>* analyzer_cast(void* iwObject)
{
    using namespace InferenceWrapper;
    return static_cast<CachingInferenceEngine<netType>*>(static_cast<ICachingInferenceEngine*>(iwObject));
}

IW_API_C void* iwCreate(
    itvcvError* error,
    ITV8::ILogger* logger,
    itvcvAnalyzerType netType,
    itvcvModeType mode,
    const int colorScheme,
    int nClasses,
    const int gpuDeviceNumToUse,
    const char* modelFilePath,
    const char* weightsFilePath,
    const char* pluginDir,
    bool int8)
{
    try
    {
        auto engineParams = InferenceWrapper::EngineCreationParams(
            logger,
            netType,
            mode,
            colorScheme,
            gpuDeviceNumToUse,
            modelFilePath,
            weightsFilePath,
            pluginDir,
            int8);

        if (nClasses == 0)
        {
            if (netType == itvcvAnalyzerClassification)
            {
                nClasses = boost::get<std::vector<ItvCv::Label>>(engineParams.net->networkParams).size();
            }
            else if (netType == itvcvAnalyzerSiamese)
            {
                auto siameseParams = boost::get<ItvCv::ReidParams>(&engineParams.net->networkParams);
                nClasses = siameseParams->vectorSize;
            }
        }

        return createAnalyzer(netType, *error, engineParams, InferenceWrapper::InferenceChannelParams{}, nClasses);
    }
    catch (const std::exception &x)
    {
        ITVCV_LOG(logger, ITV8::LOG_ERROR, "Error during creation. Got std::exception: " << x.what());
        if (*error == itvcvErrorSuccess)
        {
            *error = itvcvErrorOther;
        }
        return nullptr;
    }
    catch (...)
    {
        ITVCV_LOG(logger, ITV8::LOG_ERROR, "Error during creation. Got unknown exception");
        if (*error == itvcvErrorSuccess)
        {
            *error = itvcvErrorOther;
        }
        return nullptr;
    }
}

IW_API_C void iwCreateAsync(
    void* userData,
    itvcvCreationCallback_t creationCallback,
    itvcvError* error,
    ITV8::ILogger* logger,
    itvcvAnalyzerType netType,
    itvcvModeType mode,
    const int colorScheme,
    const int nClasses,
    const int gpuDeviceNumToUse,
    const char* modelFilePath,
    const char* weightsFilePath,
    const char* pluginDir,
    bool int8)
{
    static auto threadPool = ItvCvUtils::CreateDynamicThreadPool(
        logger,
        "iwCreateAsync",
        10,
        0,
        1);

    if (creationCallback == nullptr)
    {
        *error = itvcvErrorOther;
        return;
    }

    auto createFunc =
        [
            =,
            modelFilePath=std::string(modelFilePath),
            weightsFilePath=std::string(weightsFilePath),
            pluginDir=std::string(pluginDir)
        ]()
        {
            itvcvError e;
            auto h = iwCreate(
                &e,
                logger,
                netType,
                mode,
                colorScheme,
                nClasses,
                gpuDeviceNumToUse,
                modelFilePath.c_str(),
                weightsFilePath.c_str(),
                pluginDir.c_str(),
                int8);
            creationCallback(userData, h, e);
            return;
        };

    if (!threadPool->Post(createFunc))
    {
        *error = itvcvErrorOther;
    }
}

IW_API_C void iwDestroy(void* iwObject)
{
    delete static_cast<InferenceWrapper::ICachingInferenceEngine*>(iwObject);
}

// ##########################################
// >>>>>>>>>>>>>>>>>> CLASSIFICATION
// ##########################################
IW_API_C void iwProcessFrames(
    void* iwObject,
    itvcvError* error,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB,
    float** const results)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerClassification>(iwObject);

    auto inferenceResult = objectAnalyzer->ProcessFrames(
        nFrames,
        widths,
        heights,
        strides,
        framesDataRGB);

    *error = inferenceResult.error;
    if (inferenceResult.error != itvcvErrorSuccess)
    {
        return;
    }

    auto resultsVec = inferenceResult.results;
    auto resultVectorSize = objectAnalyzer->GetResultVectorSize();
    for (std::size_t i = 0; i < resultsVec.size(); ++i)
    {
        auto resultVec = resultsVec[i];
        int size = std::min<int>(resultVectorSize, (int)resultVec.size());

        std::memset(results[i], 0, resultVectorSize * sizeof(float));
        std::memcpy(results[i], resultVec.data(), size * sizeof(float));
    }
}

void iwAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerClassification>(iwObject);
    int resultVectorSize = objectAnalyzer->GetResultVectorSize();
    auto resultFuture = objectAnalyzer->AsyncProcessFrame(
        width,
        height,
        stride,
        frameDataRGB);

    auto resultProcessor = [userData, resultCallBack, resultVectorSize](auto&& result)
    {
        if (result.first != itvcvErrorSuccess)
        {
          resultCallBack(userData, nullptr, 0);
          return;
        }

        auto inferenceResult = result.second;
        const int vecSize = std::min<int>(
          resultVectorSize,
          inferenceResult.size());
        resultCallBack(userData, inferenceResult.data(), vecSize);
    };

    InferenceWrapper::HandleAsyncResult(*error, std::move(resultProcessor), std::move(resultFuture));
}

// ##########################################
// >>>>>>>>>>>>>>>>>> SSD
// ##########################################
IW_API_C void iwssdProcessFrames(
    void* iwObject,
    itvcvError* error,
    const char*,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB,
    int* nResults,
    float*** const results)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerSSD>(iwObject);

    InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerSSD>::BatchInferenceResult inferenceResults;

    if (!results)
    {
        inferenceResults = objectAnalyzer->ProcessFrames(nFrames, widths, heights, strides, framesDataRGB);
    }
    else
    {
        inferenceResults = objectAnalyzer->GetLastBatchResult();
    }

    *error = inferenceResults.error;
    if (inferenceResults.error != itvcvErrorSuccess)
    {
        return;
    }

    std::vector<InferenceWrapper::AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> processedResults;
    if (objectAnalyzer->GetNet()->commonParams.architecture == ItvCv::ArchitectureType::SSD_ResNet34)
    {
        for (std::size_t resultsSize = 0; resultsSize < inferenceResults.results.size(); resultsSize++)
        {
            processedResults.emplace_back(ApplyNMS(inferenceResults.results[resultsSize]));
        }
    }
    else
    {
        processedResults = std::move(inferenceResults.results);
    }

    // write num of results
    if (results == NULL)
    {
        const auto& imagesVec = processedResults;
        for (std::size_t i = 0; i < imagesVec.size(); ++i)
        {
            nResults[i] = imagesVec[i].size();
        }
        return;
    }
        // write asked results
    else
    {
        const auto& imagesVec = processedResults;
        for (std::size_t i = 0; i < imagesVec.size(); ++i)
        {
            const auto& detectionsVec = imagesVec[i];
            for (std::size_t d = 0; d < static_cast<std::size_t>(std::min(
                     int(detectionsVec.size()),
                     nResults[i])); ++d)
            {
                std::memcpy(results[i][d], detectionsVec[d].data(), SSD_RESULT_VECTOR_SIZE * sizeof(float));
            }
        }
    }
}

IW_API_C void iwssdProcessSubFrame(
    void* iwObject,
    itvcvError* error,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB,
    const int windowW,
    const int windowH,
    const int windowStepW,
    const int windowStepH,
    int* nResults,
    float*** const results)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerSSD>(iwObject);

    InferenceWrapper::CachingInferenceEngine<itvcvAnalyzerSSD>::BatchInferenceResult inferenceSubResults;

    if (!results)
    {
        const std::pair<size_t, size_t> window = std::make_pair(windowW, windowH);
        const std::pair<size_t, size_t> steps = std::make_pair(windowStepW, windowStepH);

        inferenceSubResults = objectAnalyzer->ProcessSubFrame(width, height, stride, frameDataRGB, window, steps);
    }
    else
    {
        inferenceSubResults = objectAnalyzer->GetLastBatchSubResults();
    }

    *error = inferenceSubResults.error;
    if (inferenceSubResults.error != itvcvErrorSuccess)
    {
        return;
    }

    std::vector<InferenceWrapper::AnalyzerTraits<itvcvAnalyzerSSD>::ResultType> processedSubResults;
    if (objectAnalyzer->GetNet()->commonParams.architecture == ItvCv::ArchitectureType::SSD_ResNet34)
    {
        for (std::size_t resultsSize = 0; resultsSize < inferenceSubResults.results.size(); resultsSize++)
        {
            processedSubResults.emplace_back(ApplyNMS(inferenceSubResults.results[resultsSize]));
        }
    }
    else
    {
        processedSubResults = std::move(inferenceSubResults.results);
    }

    // write num of results
    if (results == NULL)
    {
        const auto& ssdResultsVec = processedSubResults;
        const auto resultsCount = ssdResultsVec.size();
        for (auto i = 0; i < resultsCount; ++i)
        {
            nResults[i] = ssdResultsVec[i].size();
        }
        return;
    }
    // write asked results
    else
    {
        const auto& ssdResultsVec = processedSubResults;
        const auto resultsCount = ssdResultsVec.size();
        for (auto i = 0; static_cast<size_t>(i) < resultsCount; ++i)
        {
            const auto& detectionsVec = ssdResultsVec[i];
            const int resultsCountToCopy = static_cast<int>(std::min(int(detectionsVec.size()), nResults[i]));
            for (auto d = 0; d < resultsCountToCopy; ++d)
            {
                std::memcpy(results[i][d], detectionsVec[d].data(), SSD_RESULT_VECTOR_SIZE * sizeof(float));
            }
        }
    }
}

IW_API_C void iwssdAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwSSDResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerSSD>(iwObject);
    auto resultFuture = objectAnalyzer->AsyncProcessFrame(
        width,
        height,
        stride,
        frameDataRGB);

    auto resultProcessor = [userData, resultCallBack](decltype(resultFuture.get()) && result)
    {
         if (result.first != itvcvErrorSuccess)
         {
             resultCallBack(userData, nullptr, 0, 0);
             return;
         }

         const auto& ssdResults = result.second;
         const int resultsCount = ssdResults.size();
         std::vector<const float *> data;
         data.resize(resultsCount);
         for (int i = 0; i < resultsCount; ++i)
         {
             data[i] = ssdResults[i].data();
         }
         resultCallBack(
             userData,
             data.data(),
             resultsCount,
             SSD_RESULT_VECTOR_SIZE);
    };

    InferenceWrapper::HandleAsyncResult(*error, std::move(resultProcessor), std::move(resultFuture));
}

// ##########################################
// >>>>>>>>>>>>>>>>>> SIEMESE
// ##########################################
IW_API_C void iwsiameseProcessFrames(
    void* iwObject,
    itvcvError* error,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB,
    float** const results)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerSiamese>(iwObject);

    auto inferenceResult = objectAnalyzer->ProcessFrames(
        nFrames,
        widths,
        heights,
        strides,
        framesDataRGB);

    *error = inferenceResult.error;
    if (inferenceResult.error != itvcvErrorSuccess)
    {
        return;
    }

    auto& resultsVec = inferenceResult.results;
    auto resultVectorSize = objectAnalyzer->GetResultVectorSize();
    for (std::size_t i = 0; i < resultsVec.size(); ++i)
    {
        auto& resultVec = resultsVec[i];
        int size = std::min<int>(resultVectorSize, (int)resultVec.features.size());

        std::memset(results[i], 0, resultVectorSize * sizeof(float));
        std::memcpy(results[i], resultVec.features.data(), size * sizeof(float));
    }
}

IW_API_C float iwsiameseCompareVectors(
    float* const vector1,
    float* const vector2,
    const int vectorSize)
{
    // cosine distance
    double mul, denomA, denomB, A, B;
    mul = denomA = denomB = A = B = 0;
    for (int i = 0; i < vectorSize; ++i)
    {
        A = vector1[i];
        B = vector2[i];
        mul += A * B;
        denomA += A * A;
        denomB += B * B;
    }

    return mul / (sqrt(denomA * denomB));
}

// ##########################################
// >>>>>>>>>>>>>>>>>> HPE
// ##########################################
IW_API_C void ihpeGetResult(
    void* iwObject,
    itvcvError* error,
    const char*,
    int* const countsPersonPose,
    ItvCvUtils::PoseC** const result)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerHumanPoseEstimator>(iwObject);

    auto iResult = objectAnalyzer->GetLastBatchResult();

    *error = iResult.error;
    if (iResult.error != itvcvErrorSuccess)
    {
        return;
    }

    if(iResult.processedPoses.empty() && !iResult.results.empty())
    {
        if(iResult.results.size() != iResult.imgSizes.size())
        {
            iResult.error = itvcvErrorOther;
            return;
        }
        for (auto i = 0 ; static_cast<size_t>(i) < iResult.results.size(); ++i)
        {
            const auto& poseParameters = boost::get<ItvCv::PoseNetworkParams>(objectAnalyzer->GetNet()->networkParams);

            iResult.processedPoses.emplace_back(
                PoseAnalysis::GetHumanPoseByRawData(
                    iResult.results[i],
                    iResult.imgSizes[i],
                    poseParameters));
        }
    }
    if (result)
    {
        warpPosePoints(result, iResult.processedPoses);
        iResult.processedPoses.clear();
    }
    else
    {
        std::vector<int> numPersons;
        for (auto& pose:iResult.processedPoses)
        {
            auto numPersonByFrame = static_cast<int>(pose.size());
            numPersons.emplace_back(numPersonByFrame);
        }
        warpCountPeople(countsPersonPose, numPersons);
    }

}

IW_API_C void ihpeProcessFrame(
    void* iwObject,
    itvcvError* error,
    const char*,
    const int nFrames,
    const int* widths,
    const int* heights,
    const int* strides,
    const unsigned char** framesDataRGB)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerHumanPoseEstimator>(iwObject);
    const auto iResults = objectAnalyzer->ProcessFrames(
        nFrames,
        widths,
        heights,
        strides,
        framesDataRGB);

    *error = iResults.error;
}

IW_API_C void ihpeAsyncProcessFrame(
    void* iwObject,
    itvcvError* error,
    void* userData,
    iwHPEResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerHumanPoseEstimator>(iwObject);

    auto resultFuture = objectAnalyzer->AsyncProcessFrame(
        width,
        height,
        stride,
        frameDataRGB);

    const auto poseParameters = boost::get<ItvCv::PoseNetworkParams>(objectAnalyzer->GetNet()->networkParams);
    auto resultProcessor = [userData, resultCallBack, poseParameters, width, height](decltype(resultFuture.get()) && result)
    {
        if (result.first != itvcvErrorSuccess)
        {
            resultCallBack(
                userData,
                0,
                nullptr);
            return;
        }

        auto& hpeResults = result.second;
        ITV8::Size imgSize(width, height);

        auto poses = PoseAnalysis::GetHumanPoseByRawData(
            hpeResults,
            imgSize,
            poseParameters);
        ItvCvUtils::PoseC poseC;
        CombinedPoses combinedPoses(poses);
        auto poseOut = combinedPoses.GetView();
        auto countPose = static_cast<int>(poses.size());
        resultCallBack(
            userData,
            countPose,
            poseOut.data());
    };

    InferenceWrapper::HandleAsyncResult(*error, std::move(resultProcessor), std::move(resultFuture));
}

IW_API_C void iwmaskAsyncProcessFrame(void* iwObject,
    itvcvError* error,
    void* userData,
    iwMASKResultCallback_t resultCallBack,
    const int width,
    const int height,
    const int stride,
    const unsigned char* frameDataRGB)
{
    auto objectAnalyzer = analyzer_cast<itvcvAnalyzerMaskSegments>(iwObject);
    auto resultFuture = objectAnalyzer->AsyncProcessFrame(
        width,
        height,
        stride,
        frameDataRGB);

    auto resultProcessor = [userData, resultCallBack](decltype(resultFuture.get()) && result)
    {
        if (result.first != itvcvErrorSuccess)
        {
            resultCallBack(
                userData,
                nullptr,
                nullptr,
                &result.first);
            return;
        }

        auto& maskResults = result.second;
        resultCallBack(
            userData,
            maskResults.dims.data(),
            maskResults.data.data(),
            &result.first);

    };

    InferenceWrapper::HandleAsyncResult(*error, std::move(resultProcessor), std::move(resultFuture));
}

itvcvError iwGetInputGeometry(void* iwObject, ITV8::Size* out)
{
    auto objectAnalyzer = static_cast<InferenceWrapper::ICachingInferenceEngine*>(iwObject);
    auto size = objectAnalyzer->GetInputGeometry();
    if (out)
        *out = ITV8::Size(size.width, size.height);
    return itvcvErrorSuccess;
}

void iwTakeStats(void* iwObject, std::uint32_t period_ms)
{
    auto objectAnalyzer = static_cast<InferenceWrapper::ICachingInferenceEngine*>(iwObject);
    objectAnalyzer->TakeStats(std::chrono::milliseconds(period_ms));
}
