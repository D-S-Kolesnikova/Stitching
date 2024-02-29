#include <InferenceWrapper/InferenceEngine.h>
#include <NetworkInformation/NetworkInformationLib.h>

#include <opencv2/stitching.hpp>
#include <opencv2/calib3d.hpp>

#include <vector>
#include <string>
#include <memory>

using PointEngine_t = InferenceWrapper::IInferenceEngine<itvcvAnalyzerPointMatcher>;
using PointEngineResult_t = InferenceWrapper::InferenceResultTraits<itvcvAnalyzerPointMatcher>::InferenceResult_t;

class LightGlue;

class SuperPoint :public cv::Feature2D
{
public:
	SuperPoint(const std::string& modelPath, const std::vector<cv::Mat>& frames);
	virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors,
		bool useProvidedKeypoints = false);
	virtual void detect(cv::InputArray image,
		std::vector<cv::KeyPoint>& keypoints,
		cv::InputArray mask = cv::noArray());
	virtual void compute(cv::InputArray image,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors);
	cv::Ptr<LightGlue> getMatcher() { return m_matcher; }

private:
	std::string m_modelPath;
	cv::Ptr<LightGlue> m_matcher;
	std::shared_ptr<ITV8::ILogger> m_logger;
	std::unique_ptr<PointEngine_t> m_engine;
	PointEngineResult_t m_results;
	std::vector<cv::Mat> m_framesBuffer;
};

class LightGlue :public cv::detail::FeaturesMatcher
{
public:
	LightGlue(std::string modelPath, cv::Stitcher::Mode mode, float matchThresh);
	void match(const cv::detail::ImageFeatures& features1, const cv::detail::ImageFeatures& features2, cv::detail::MatchesInfo& matches_info);
	void operator ()(const std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches,
		const cv::UMat& mask = cv::UMat());
	void SetResultFromBuffer(const PointEngineResult_t& results);

private:
	cv::Stitcher::Mode m_mode = cv::Stitcher::Mode::PANORAMA;
	std::string m_modelPath;
	std::vector<cv::detail::ImageFeatures> m_features;
	std::vector<cv::detail::MatchesInfo> m_pairwise_matches;
	float m_matchThresh = 0.0;
	PointEngineResult_t m_results;

	void AddFeature(const cv::detail::ImageFeatures& features);
	void AddMatcheinfo(const cv::detail::MatchesInfo& matches_info);
};
