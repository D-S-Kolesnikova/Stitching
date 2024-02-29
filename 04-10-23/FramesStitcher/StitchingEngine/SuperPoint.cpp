#include"SuperPoint.h"

#include <ItvCvUtils/Log.h>

#include <opencv2/stitching.hpp>

#include <iostream>

constexpr float match_conf = 0.1f;

SuperPoint::SuperPoint(const std::string& modelPath, const std::vector<cv::Mat>& frames):
	m_modelPath(modelPath),
	m_framesBuffer(frames)
{
	m_logger = ItvCv::Utils::CreateStreamLogger(std::cout, 0);

	InferenceWrapper::EngineCreationParams engineParams(
		m_logger.get(),
		itvcvAnalyzerType::itvcvAnalyzerPointMatcher,
		itvcvModeType::itvcvModeGPU,
		16,
		0,
		"",
		modelPath,
		"",
		false
	);

	itvcvError error{ itvcvErrorSuccess };
	m_engine = InferenceWrapper::IInferenceEngine<itvcvAnalyzerPointMatcher>::Create(error, engineParams);
	if (error != itvcvErrorSuccess)
	{
		throw std::logic_error("unable to create engine");
	}
	m_results.first = itvcvErrorInference;
	m_matcher = cv::makePtr<LightGlue>(m_modelPath, cv::Stitcher::PANORAMA, match_conf);
}

bool MatIsEqual(const cv::Mat mat1, const cv::Mat mat2)
{
	if (mat1.empty() && mat2.empty())
	{
		return true;
	}
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims)
	{
		return false;
	}
	bool areIdentical = !cv::norm(mat1, mat2, cv::NORM_L1);
	return areIdentical;
}

void SuperPoint::detectAndCompute(cv::InputArray image, cv::InputArray mask,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors,
	bool useProvidedKeypoints)
{
	// to avoid repeated inference
	if (!(m_results.first == itvcvErrorSuccess))
	{
		auto iwFrame1 = ItvCv::Frame{ m_framesBuffer[0].cols, m_framesBuffer[0].rows, (int)m_framesBuffer[0].step, m_framesBuffer[0].data};
		auto iwFrame2 = ItvCv::Frame{ m_framesBuffer[1].cols, m_framesBuffer[1].rows, (int)m_framesBuffer[1].step,  m_framesBuffer[1].data};
		std::vector< ItvCv::Frame> iwFrames = { iwFrame1, iwFrame2 };
		auto future = m_engine->AsyncProcessFrame(iwFrames);
		future.wait();
		m_results = future.get();
	}
	if (m_results.first == itvcvErrorSuccess)
	{
		cv::Mat inputImage = image.getMat();
		int imgIndx = 0;
		for (int i = 0; i < 2; i++)
		{
			if (MatIsEqual(inputImage, m_framesBuffer[i]))
			{
				imgIndx = i;
			}
		}
		auto kp = m_results.second.keypoints[imgIndx].first.data();
		int kpCount = m_results.second.keypoints[imgIndx].second[1];

		auto netWidth = m_engine->GetInputGeometry().width;
		auto netHeight = m_engine->GetInputGeometry().height;
		double fx = float(netWidth) / float(m_framesBuffer[imgIndx].cols);
		double fy = float(netHeight) / float(m_framesBuffer[imgIndx].rows);

		keypoints.resize(kpCount);
		for (int i = 0; i < kpCount; i++)
		{
			cv::KeyPoint p;
			int index = i * 2;
			p.pt.x = ((float)kp[index]* netWidth * 0.5) + netWidth * 0.5;
			p.pt.y = ((float)kp[index + 1] * netHeight * 0.5) + netHeight * 0.5;

			//normalization of points relative to the frame with the original aspect ratio
			p.pt.x = (p.pt.x + .5) / fx - .5;
			p.pt.y = (p.pt.y + .5) / fy - .5;
			keypoints[i] = p;
		}
		auto desshape = m_results.second.descriptors[imgIndx].second;
		float* des = m_results.second.descriptors[imgIndx].first.data();
		cv::Mat desmat = descriptors.getMat();
		desmat.create(cv::Size(desshape[2], desshape[1]), CV_32FC1);
		for (int h = 0; h < desshape[1]; h++)
		{
			for (int w = 0; w < desshape[2]; w++)
			{
				int index = h * desshape[2] + w;
				desmat.at<float>(h, w) = des[index];
			}
		}
		desmat.copyTo(descriptors);
	}
	else
	{
		return;
	}
	m_matcher->SetResultFromBuffer(m_results);
}
void SuperPoint::detect(cv::InputArray image,
	std::vector<cv::KeyPoint>& keypoints,
	cv::InputArray mask)
{
	// to avoid repeated inference
	if (!(m_results.first == itvcvErrorSuccess))
	{
		auto iwFrame1 = ItvCv::Frame{ m_framesBuffer[0].cols, m_framesBuffer[0].rows, (int)m_framesBuffer[0].step, m_framesBuffer[0].data };
		auto iwFrame2 = ItvCv::Frame{ m_framesBuffer[1].cols, m_framesBuffer[1].rows, (int)m_framesBuffer[1].step, m_framesBuffer[1].data };
		std::vector< ItvCv::Frame> iwFrames = { iwFrame1, iwFrame2 };
		auto future = m_engine->AsyncProcessFrame(iwFrames);
		future.wait();
		m_results = future.get();
	}
	if (m_results.first == itvcvErrorSuccess)
	{
		cv::Mat inputImage = image.getMat();
		int imgIndx = 0;
		for (int i = 0; i < 2; i++)
		{
			if (MatIsEqual(inputImage, m_framesBuffer[i]))
			{
				imgIndx = i;
			}
		}
		auto kp = m_results.second.keypoints[imgIndx].first.data();
		int kpCount = m_results.second.keypoints[imgIndx].second[1];

		auto netWidth = m_engine->GetInputGeometry().width;
		auto netHeight = m_engine->GetInputGeometry().height;
		double fx = netWidth / m_framesBuffer[imgIndx].cols;
		double fy = netHeight / m_framesBuffer[imgIndx].rows;

		keypoints.resize(kpCount);
		for (int i = 0; i < kpCount; i++)
		{
			cv::KeyPoint p;
			int index = i * 2;
			p.pt.x = ((float)kp[index] * netWidth * 0.5) + netWidth * 0.5;
			p.pt.y = ((float)kp[index + 1] * netHeight * 0.5) + netHeight * 0.5;

			//normalization of points relative to the frame with the original aspect ratio
			p.pt.x = (p.pt.x + .5) / fx - .5;
			p.pt.y = (p.pt.y + .5) / fy - .5;
			keypoints[i] = p;
		}
	}
	else
	{
		return;
	}
}
void SuperPoint::compute(cv::InputArray image,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors)
{
	// to avoid repeated inference
	if (!(m_results.first == itvcvErrorSuccess))
	{
		auto iwFrame1 = ItvCv::Frame{ m_framesBuffer[0].cols, m_framesBuffer[0].rows, (int)m_framesBuffer[0].step, m_framesBuffer[0].data };
		auto iwFrame2 = ItvCv::Frame{ m_framesBuffer[1].cols, m_framesBuffer[1].rows, (int)m_framesBuffer[1].step, m_framesBuffer[1].data };
		std::vector< ItvCv::Frame> iwFrames = { iwFrame1, iwFrame2 };
		auto future = m_engine->AsyncProcessFrame(iwFrames);
		future.wait();
		m_results = future.get();
	}
	if (m_results.first == itvcvErrorSuccess)
	{
		cv::Mat inputImage = image.getMat();
		int imgIndx = 0;
		for (int i = 0; i < 2; i++)
		{
			if (MatIsEqual(inputImage, m_framesBuffer[i]))
			{
				imgIndx = i;
			}
		}
		auto kp = m_results.second.keypoints[imgIndx].first.data();
		int kpCount = m_results.second.keypoints[imgIndx].second[1];

		auto netWidth = m_engine->GetInputGeometry().width;
		auto netHeight = m_engine->GetInputGeometry().height;
		double fx = netWidth / m_framesBuffer[imgIndx].cols;
		double fy = netHeight / m_framesBuffer[imgIndx].rows;

		keypoints.resize(kpCount);
		for (int i = 0; i < kpCount; i++)
		{
			cv::KeyPoint p;
			int index = i * 2;
			p.pt.x = ((float)kp[index] * netWidth * 0.5) + netWidth * 0.5;
			p.pt.y = ((float)kp[index + 1] * netHeight * 0.5) + netHeight * 0.5;

			//normalization of points relative to the frame with the original aspect ratio
			p.pt.x = (p.pt.x + .5) / fx - .5;
			p.pt.y = (p.pt.y + .5) / fy - .5;
			keypoints[i] = p;
		}
		auto desshape = m_results.second.descriptors[imgIndx].second;
		float* des = m_results.second.descriptors[imgIndx].first.data();
		cv::Mat desmat = descriptors.getMat();
		desmat.create(cv::Size(desshape[2], desshape[1]), CV_32FC1);
		for (int h = 0; h < desshape[1]; h++)
		{
			for (int w = 0; w < desshape[2]; w++)
			{
				int index = h * desshape[2] + w;
				desmat.at<float>(h, w) = des[index];
			}
		}
		desmat.copyTo(descriptors);
	}
	else
	{
		return;
	}
	m_matcher->SetResultFromBuffer(m_results);
}

LightGlue::LightGlue(std::string modelPath, cv::Stitcher::Mode mode, float matchThresh)
	: m_matchThresh(matchThresh)
	, m_mode(mode)
	, m_modelPath(modelPath)
{
}

void LightGlue::SetResultFromBuffer(const PointEngineResult_t& results)
{
	m_results = results;
}

void LightGlue::match(const cv::detail::ImageFeatures& features1, const cv::detail::ImageFeatures& features2,
	cv::detail::MatchesInfo& matches_info)
{
	std::vector<int64_t> match1shape = m_results.second.matches[0].second;

	float* match1 = (float*)(m_results.second.matches[0].first.data());
	int match1counts = match1shape[1];

	std::vector<int64_t> mscoreshape1 = m_results.second.scores[0].second;
	float* mscore1 = m_results.second.scores[0].first.data();

	std::vector<int64_t> match2shape = m_results.second.matches[1].second;
	float* match2 = (float*)(m_results.second.matches[1].first.data());
	int match2counts = match2shape[1];

	std::vector<int64_t> mscoreshape2 = m_results.second.scores[1].second;
	float* mscore2 = m_results.second.scores[1].first.data();

	matches_info.src_img_idx = features1.img_idx;
	matches_info.dst_img_idx = features2.img_idx;

	std::set<std::pair<int, int> > matches;
	for (int i = 0; i < match1counts; i++)
	{
		auto index = int64_t(match1[i]);
		if (match1[i] > -1 && mscore1[i] > this->m_matchThresh && match2[index] == i)
		{
			cv::DMatch mt;
			mt.queryIdx = i;
			mt.trainIdx = match1[i];
			matches_info.matches.push_back(mt);
			matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));
		}
	}

	// Construct point-point correspondences for transform estimation
	cv::Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	cv::Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);

	if (this->m_mode == cv::Stitcher::SCANS)
	{
		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			src_points.at<cv::Point2f>(0, static_cast<int>(i)) = features1.keypoints[matches_info.matches[i].queryIdx].pt;
			dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = features2.keypoints[matches_info.matches[i].trainIdx].pt;
		}

		// Find pair-wise motion
		matches_info.H = estimateAffine2D(src_points, dst_points, matches_info.inliers_mask);

		if (matches_info.H.empty()) {
			// could not find transformation
			matches_info.confidence = 0;
			matches_info.num_inliers = 0;
			return;
		}

		// Find number of inliers
		matches_info.num_inliers = 0;
		for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
			if (matches_info.inliers_mask[i])
				matches_info.num_inliers++;

		// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic
		// Image Stitching using Invariant Features"
		matches_info.confidence =
			matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

		/* should we remove matches between too close images? */
		// matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

		// extend H to represent linear transformation in homogeneous coordinates
		matches_info.H.push_back(cv::Mat::zeros(1, 3, CV_64F));
		matches_info.H.at<double>(2, 2) = 1;
	}
	else if (this->m_mode == cv::Stitcher::PANORAMA)
	{
		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			const cv::DMatch& m = matches_info.matches[i];

			cv::Point2f p = features1.keypoints[m.queryIdx].pt;
			src_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;

			p = features2.keypoints[m.trainIdx].pt;
			dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;
		}

		// Find pair-wise motion
		matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, cv::RANSAC);
		std::cout << matches_info.H << std::endl;
		if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
			return;

		// Find number of inliers
		matches_info.num_inliers = 0;
		for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
			if (matches_info.inliers_mask[i])
				matches_info.num_inliers++;

		// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
		// using Invariant Features"
		matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

		// Set zero confidence to remove matches between too close images, as they don't provide
		// additional information anyway. The threshold was set experimentally.
		matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

		// Check if we should try to refine motion
		if (matches_info.num_inliers < 6)
			return;

		// Construct point-point correspondences for inliers only
		src_points.create(1, matches_info.num_inliers, CV_32FC2);
		dst_points.create(1, matches_info.num_inliers, CV_32FC2);
		int inlier_idx = 0;
		for (size_t i = 0; i < matches_info.matches.size(); ++i)
		{
			if (!matches_info.inliers_mask[i])
				continue;

			const cv::DMatch& m = matches_info.matches[i];

			cv::Point2f p = features1.keypoints[m.queryIdx].pt;
			src_points.at<cv::Point2f>(0, inlier_idx) = p;

			p = features2.keypoints[m.trainIdx].pt;
			dst_points.at<cv::Point2f>(0, inlier_idx) = p;

			inlier_idx++;
		}

		// Rerun motion estimation on inliers only
		matches_info.H = findHomography(src_points, dst_points, cv::RANSAC);
	}
	std::cout << matches_info.H << std::endl;
	this->AddFeature(features1);
	this->AddFeature(features2);
	this->AddMatcheinfo(matches_info);
}

void LightGlue::AddFeature(const cv::detail::ImageFeatures& features) {
	bool find = false;
	for (int i = 0; i < this->m_features.size(); i++)
	{
		if (features.img_idx == this->m_features[i].img_idx)
		{
			find = true;
		}
	}
	if (find == false)
	{
		this->m_features.push_back(features);
	}
}

void LightGlue::AddMatcheinfo(const cv::detail::MatchesInfo& matches)
{
	bool find = false;
	for (int i = 0; i < this->m_pairwise_matches.size(); i++)
	{
		if (matches.src_img_idx == this->m_pairwise_matches[i].src_img_idx &&
			matches.dst_img_idx == this->m_pairwise_matches[i].dst_img_idx)
		{
			find = true;
		}

		if (matches.src_img_idx == this->m_pairwise_matches[i].dst_img_idx &&
			matches.dst_img_idx == this->m_pairwise_matches[i].src_img_idx)
		{
			find = true;
		}
	}
	if (find == false)
	{
		this->m_pairwise_matches.push_back(cv::detail::MatchesInfo(matches));
	}
}

struct MatchPairsBody : cv::ParallelLoopBody
{
	MatchPairsBody(cv::detail::FeaturesMatcher& matcher, const std::vector<cv::detail::ImageFeatures>& features,
		std::vector<cv::detail::MatchesInfo>& pairwiseMatches, std::vector<std::pair<int, int> >& nearPairs)
		: m_matcher(matcher)
		, m_features(features)
		, m_pairwiseMatches(pairwiseMatches)
		, m_nearPairs(nearPairs) {}

	void operator ()(const cv::Range& r) const CV_OVERRIDE
	{
		// save entry rng state
		cv::RNG rng = cv::theRNG();
		const int num_images = static_cast<int>(m_features.size());
		for (int i = r.start; i < r.end; ++i)
		{
			// force "stable" RNG seed for each processed pair
			cv::theRNG() = cv::RNG(rng.state + i);

			int from = m_nearPairs[i].first;
			int to = m_nearPairs[i].second;
			int pair_idx = from * num_images + to;

			m_matcher(m_features[from], m_features[to], m_pairwiseMatches[pair_idx]);
			m_pairwiseMatches[pair_idx].src_img_idx = from;
			m_pairwiseMatches[pair_idx].dst_img_idx = to;

			size_t dual_pair_idx = to * num_images + from;

			m_pairwiseMatches[dual_pair_idx] = m_pairwiseMatches[pair_idx];
			m_pairwiseMatches[dual_pair_idx].src_img_idx = to;
			m_pairwiseMatches[dual_pair_idx].dst_img_idx = from;

			if (!m_pairwiseMatches[pair_idx].H.empty())
			{
				m_pairwiseMatches[dual_pair_idx].H = m_pairwiseMatches[pair_idx].H.inv();
			}

			for (size_t j = 0; j < m_pairwiseMatches[dual_pair_idx].matches.size(); ++j)
			{
				std::swap(m_pairwiseMatches[dual_pair_idx].matches[j].queryIdx,
					m_pairwiseMatches[dual_pair_idx].matches[j].trainIdx);
			}
		}
	}

	cv::detail::FeaturesMatcher& m_matcher;
	const std::vector<cv::detail::ImageFeatures>& m_features;
	std::vector<cv::detail::MatchesInfo>& m_pairwiseMatches;
	std::vector<std::pair<int, int> >& m_nearPairs;

private:
	void operator =(const MatchPairsBody&);
};

void  LightGlue::operator ()(const std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwiseMatches,
	const cv::UMat& mask)
{
	const int num_images = static_cast<int>(features.size());
	CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.cols == num_images && mask.rows));
	cv::Mat_<uchar> mask_(mask.getMat(cv::ACCESS_READ));
	if (mask_.empty())
		mask_ = cv::Mat::ones(num_images, num_images, CV_8U);

	std::vector<std::pair<int, int> > near_pairs;
	for (int i = 0; i < num_images - 1; ++i)
		for (int j = i + 1; j < num_images; ++j)
			if (features[i].keypoints.size() > 0 && features[j].keypoints.size() > 0 && mask_(i, j))
				near_pairs.push_back(std::make_pair(i, j));
	// clear history values
	pairwiseMatches.clear();
	pairwiseMatches.resize(num_images * num_images);
	// save entry rng state
	cv::RNG rng = cv::theRNG();

	MatchPairsBody body(*this, features, pairwiseMatches, near_pairs);

	if (is_thread_safe_)
	{
		parallel_for_(cv::Range(0, static_cast<int>(near_pairs.size())), body);
	}
	else
	{
		body(cv::Range(0, static_cast<int>(near_pairs.size())));
	}
}
