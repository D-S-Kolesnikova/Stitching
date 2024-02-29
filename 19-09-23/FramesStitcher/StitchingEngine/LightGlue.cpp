#include"LightGlue.h"

#include<opencv2/core/utility.hpp>

LightGlue::LightGlue(std::wstring modelPath, cv::Stitcher::Mode mode, float matchThresh)
{
	this->m_matchThresh = matchThresh;
	this->m_mode = mode;
	this->m_modelPath = modelPath;
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

void LightGlue::match(const cv::detail::ImageFeatures& features1, const cv::detail::ImageFeatures& features2,
	cv::detail::MatchesInfo& matches_info)
{
	Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "LightGlue");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	// Load the LightGlue network
	static Ort::Session lightglueSession(env, this->m_modelPath.c_str(), sessionOptions);
	std::vector<float>kp1; kp1.resize(features1.keypoints.size() * 2);
	std::vector<float>kp2; kp2.resize(features2.keypoints.size() * 2);
	float f1wid = features1.img_size.width / 2.0f;
	float f1hei = features1.img_size.height / 2.0f;
	for (int i = 0; i < features1.keypoints.size(); i++)
	{
		kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
		kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
	}
	float f2wid = features2.img_size.width / 2.0f;
	float f2hei = features2.img_size.height / 2.0f;
	for (int i = 0; i < features2.keypoints.size(); i++)
	{
		kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
		kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
	}

	std::vector<float>des1; des1.resize(features1.keypoints.size() * 256);
	cv::Mat des1mat = features1.descriptors.getMat(cv::ACCESS_READ);
	for (int w = 0; w < des1mat.cols; w++)
	{
		for (int h = 0; h < des1mat.rows; h++)
		{
			int index = h * features1.descriptors.cols + w;
			des1[index] = des1mat.at<float>(h, w);
		}
	}
	std::vector<float>des2; des2.resize(features2.keypoints.size() * 256);
	cv::Mat des2mat = features2.descriptors.getMat(cv::ACCESS_READ);
	for (int w = 0; w < des2mat.cols; w++)
	{
		for (int h = 0; h < des2mat.rows; h++)
		{
			int index = h * features2.descriptors.cols + w;
			des2[index] = des2mat.at<float>(h, w);
		}
	}

	const char* input_names[] = { "kpts0", "kpts1", "desc0", "desc1" };
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	std::vector<Ort::Value> inputTensor;
	std::vector<int64_t> kp1Shape{ 1,(int64_t)features1.keypoints.size(), 2 };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp1.data(), kp1.size(), kp1Shape.data(), kp1Shape.size()));
	std::vector<int64_t> kp2Shape{ 1,(int64_t)features2.keypoints.size(), 2 };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp2.data(), kp2.size(), kp2Shape.data(), kp2Shape.size()));
	std::vector<int64_t> des1Shape{ 1,(int64_t)features1.keypoints.size(), features1.descriptors.cols };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des1.data(), des1.size(), des1Shape.data(), des1Shape.size()));
	std::vector<int64_t> des2Shape{ 1,(int64_t)features2.keypoints.size(), features2.descriptors.cols };
	inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des2.data(), des2.size(), des2Shape.data(), des2Shape.size()));
	const char* output_names[] = { "matches0","matches1","mscores0","mscores1" };
	Ort::RunOptions run_options;
	std::vector<Ort::Value> outputs = lightglueSession.Run(run_options, input_names, inputTensor.data(), 4, output_names, 4);
	std::vector<int64_t> match1shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
	int match1counts = match1shape[1];

	std::vector<int64_t> mscoreshape1 = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();

	std::vector<int64_t> match2shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
	int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
	int match2counts = match2shape[1];

	std::vector<int64_t> mscoreshape2 = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
	float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();

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
	for (int i = 0; i < this->features_.size(); i++)
	{
		if (features.img_idx == this->features_[i].img_idx)
			find = true;
	}
	if(find == false)
		this->features_.push_back(features);
}
void LightGlue::AddMatcheinfo(const cv::detail::MatchesInfo& matches)
{
	bool find = false;
	for (int i = 0; i < this->pairwise_matches_.size(); i++)
	{
		if (matches.src_img_idx == this->pairwise_matches_[i].src_img_idx &&
			matches.dst_img_idx == this->pairwise_matches_[i].dst_img_idx)
			find = true;
		if (matches.src_img_idx == this->pairwise_matches_[i].dst_img_idx &&
			matches.dst_img_idx == this->pairwise_matches_[i].src_img_idx)
			find = true;

	}
	if (find == false)
		this->pairwise_matches_.push_back(cv::detail::MatchesInfo(matches));
}

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
