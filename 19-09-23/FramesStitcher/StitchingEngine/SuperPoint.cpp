#include"SuperPoint.h"

#include <ItvCvUtils/Log.h>

#include "opencv2/stitching.hpp"

#include <iostream>
#include <fstream>

constexpr float match_conf = 0.1f;

SuperPoint::~SuperPoint()
{

}

SuperPoint::SuperPoint(const std::string& modelPath, const std::vector<cv::Mat>& frames):
	m_modelPath(modelPath),
	m_framesBuffer(frames)
{
	m_logger = ItvCv::Utils::CreateStreamLogger(std::cout, 0);

	InferenceWrapper::EngineCreationParams engineParams(
		m_logger.get(),
		itvcvAnalyzerType::itvcvAnalyzerPointDetection,
		itvcvModeType::itvcvModeGPU,
		16,
		0,
		"",
		modelPath,
		"",
		false
	);

	itvcvError error{ itvcvErrorSuccess };
	m_engine = InferenceWrapper::IInferenceEngine<itvcvAnalyzerPointDetection>::Create(error, engineParams);
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
	cv::Mat diff; cv::Mat diff1color;
	cv::compare(mat1, mat2, diff, cv::CMP_NE);
	int nz = cv::countNonZero(diff1color);
	return nz == 0;
}

void SuperPoint::detectAndCompute(cv::InputArray image, cv::InputArray mask,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors,
	bool useProvidedKeypoints)
{
	/*std::ofstream ofs;
	ofs.open("E:/devel/helper/homography/projects/LightGlue/inputTensorCPP.txt");
	for (int h = 0; h < image.rows; h++)
	{
		for (int w = 0; w < image.cols; w++)
		{
			imgData.push_back(floatImage.at<float>(h, w) / 255.0f);
		}
	}
	ofs.close();*/
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
		int imgIndx = 0;
		for (int i = 0; i < 2; i++)
		{
			if (MatIsEqual(image.getMat(), m_framesBuffer[i]))
			{
				imgIndx = i;
			}
		}
		auto kp = m_results.second.keypoints[imgIndx].first.data();
		int keypntcounts = m_results.second.keypoints[imgIndx].second[1];
		keypoints.resize(keypntcounts);
		for (int i = 0; i < keypntcounts; i++)
		{
			cv::KeyPoint p;
			int index = i * 2;
			p.pt.x = ((float)kp[index]);
			p.pt.y = ((float)kp[index + 1]);
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
	if (m_results.first == itvcvErrorSuccess)
	{
		int imgIndx = 0;
		for (int i = 0; i < 2; i++)
		{
			if (MatIsEqual(image.getMat(), m_framesBuffer[i]))
			{
				imgIndx = i;
			}
		}
		auto kp = m_results.second.keypoints[imgIndx].first;
		int keypntcounts = m_results.second.keypoints[imgIndx].second[1];
		keypoints.resize(keypntcounts);
		for (int i = 0; i < keypntcounts; i++)
		{
			cv::KeyPoint p;
			int index = i * 2;
			p.pt.x = ((float)kp[index]);
			p.pt.y = ((float)kp[index + 1]);
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
	if (m_results.first == itvcvErrorSuccess)
	{
		int imgIndx = 0;
		for (int i = 0; i < 2; i++)
		{
			if (MatIsEqual(image.getMat(), m_framesBuffer[i]))
			{
				imgIndx = i;
			}
		}
		auto kp = m_results.second.keypoints[imgIndx].first;
		int keypntcounts = m_results.second.keypoints[imgIndx].second[1];
		keypoints.resize(keypntcounts);
		for (int i = 0; i < keypntcounts; i++)
		{
			cv::KeyPoint p;
			int index = i * 2;
			p.pt.x = ((float)kp[index]);
			p.pt.y = ((float)kp[index + 1]);
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
	int64_t* match1 = (int64_t*)(m_results.second.matches[0].first.data());
	int match1counts = match1shape[1];

	std::vector<int64_t> mscoreshape1 = m_results.second.scores[0].second;
	float* mscore1 = m_results.second.scores[0].first.data();

	std::vector<int64_t> match2shape = m_results.second.matches[1].second;
	int64_t* match2 = (int64_t*)(m_results.second.matches[1].first.data());
	int match2counts = match2shape[1];

	std::vector<int64_t> mscoreshape2 = m_results.second.scores[1].second;
	float* mscore2 = m_results.second.scores[1].first.data();

	matches_info.src_img_idx = features1.img_idx;
	matches_info.dst_img_idx = features2.img_idx;

	std::set<std::pair<int, int> > matches;
	for (int i = 0; i < match1counts; i++)
	{
		if (match1[i] > -1 && mscore1[i] > this->m_matchThresh && match2[match1[i]] == i)
		{
			cv::DMatch mt;
			mt.queryIdx = i;
			mt.trainIdx = match1[i];
			matches_info.matches.push_back(mt);
			matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));
		}
	}

	for (int i = 0; i < match2counts; i++)
	{
		if (match2[i] > -1 && mscore2[i] > this->m_matchThresh && match1[match2[i]] == i)
		{
			cv::DMatch mt;
			mt.queryIdx = match2[i];
			mt.trainIdx = i;

			if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end())
				matches_info.matches.push_back(mt);
		}
	}

	std::cout << "matches count:" << matches_info.matches.size() << std::endl;
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
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			src_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;

			p = features2.keypoints[m.trainIdx].pt;
			p.x -= features2.img_size.width * 0.5f;
			p.y -= features2.img_size.height * 0.5f;
			dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;
		}

		// Find pair-wise motion
		matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, cv::RANSAC);
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
			p.x -= features1.img_size.width * 0.5f;
			p.y -= features1.img_size.height * 0.5f;
			src_points.at<cv::Point2f>(0, inlier_idx) = p;

			p = features2.keypoints[m.trainIdx].pt;
			p.x -= features2.img_size.width * 0.5f;
			p.y -= features2.img_size.height * 0.5f;
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

void LightGlue::AddFeature(cv::detail::ImageFeatures features) {
	bool find = false;
	for (int i = 0; i < this->m_features.size(); i++)
	{
		if (features.img_idx == this->m_features[i].img_idx)
			find = true;
	}
	if (find == false)
		this->m_features.push_back(features);
}

void LightGlue::AddMatcheinfo(const cv::detail::MatchesInfo& matches)
{
	bool find = false;
	for (int i = 0; i < this->m_pairwise_matches.size(); i++)
	{
		if (matches.src_img_idx == this->m_pairwise_matches[i].src_img_idx &&
			matches.dst_img_idx == this->m_pairwise_matches[i].dst_img_idx)
			find = true;
		if (matches.src_img_idx == this->m_pairwise_matches[i].dst_img_idx &&
			matches.dst_img_idx == this->m_pairwise_matches[i].src_img_idx)
			find = true;

	}
	if (find == false)
		this->m_pairwise_matches.push_back(cv::detail::MatchesInfo(matches));
}

struct MatchPairsBody : cv::ParallelLoopBody
{
	MatchPairsBody(cv::detail::FeaturesMatcher& _matcher, const std::vector<cv::detail::ImageFeatures>& _features,
		std::vector<cv::detail::MatchesInfo>& _pairwise_matches, std::vector<std::pair<int, int> >& _near_pairs)
		: matcher(_matcher), features(_features),
		pairwise_matches(_pairwise_matches), near_pairs(_near_pairs) {}

	void operator ()(const cv::Range& r) const CV_OVERRIDE
	{
		cv::RNG rng = cv::theRNG(); // save entry rng state
		const int num_images = static_cast<int>(features.size());
		for (int i = r.start; i < r.end; ++i)
		{
			cv::theRNG() = cv::RNG(rng.state + i); // force "stable" RNG seed for each processed pair

			int from = near_pairs[i].first;
			int to = near_pairs[i].second;
			int pair_idx = from * num_images + to;

			matcher(features[from], features[to], pairwise_matches[pair_idx]);
			pairwise_matches[pair_idx].src_img_idx = from;
			pairwise_matches[pair_idx].dst_img_idx = to;

			size_t dual_pair_idx = to * num_images + from;

			pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
			pairwise_matches[dual_pair_idx].src_img_idx = to;
			pairwise_matches[dual_pair_idx].dst_img_idx = from;

			if (!pairwise_matches[pair_idx].H.empty())
				pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

			for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
				std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
					pairwise_matches[dual_pair_idx].matches[j].trainIdx);
		}
	}

	cv::detail::FeaturesMatcher& matcher;
	const std::vector<cv::detail::ImageFeatures>& features;
	std::vector<cv::detail::MatchesInfo>& pairwise_matches;
	std::vector<std::pair<int, int> >& near_pairs;

private:
	void operator =(const MatchPairsBody&);
};

void  LightGlue::operator ()(const std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches,
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

	pairwise_matches.clear(); // clear history values
	pairwise_matches.resize(num_images * num_images);

	cv::RNG rng = cv::theRNG(); // save entry rng state

	MatchPairsBody body(*this, features, pairwise_matches, near_pairs);

	if (is_thread_safe_)
		parallel_for_(cv::Range(0, static_cast<int>(near_pairs.size())), body);
	else
		body(cv::Range(0, static_cast<int>(near_pairs.size())));
}
