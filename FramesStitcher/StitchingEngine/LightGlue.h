#pragma once
#include <vector>
#include <string>
#include <memory>
#include<iostream>

#include "opencv2/stitching.hpp"
#include "opencv2/calib3d.hpp"

class LightGlue :public cv::detail::FeaturesMatcher
{
protected:
	cv::Stitcher::Mode m_mode;
	std::wstring m_modelPath;
	std::vector<cv::detail::ImageFeatures> features_;
	std::vector<cv::detail::MatchesInfo> pairwise_matches_;
	float m_matchThresh = 0.0;

	void AddFeature(cv::detail::ImageFeatures features);
	void AddMatcheinfo(const cv::detail::MatchesInfo& matches_info);
public:
	LightGlue(std::wstring modelPath, cv::Stitcher::Mode mode, float matchThresh);
	void match(const cv::detail::ImageFeatures& features1, const cv::detail::ImageFeatures& features2,
		cv::detail::MatchesInfo& matches_info);
	CV_WRAP_AS(apply) void operator ()(const cv::detail::ImageFeatures& features1, const cv::detail::ImageFeatures& features2,
		CV_OUT cv::detail::MatchesInfo& matches_info) {
		match(features1, features2, matches_info);
	}
	CV_WRAP_AS(apply2) void operator ()(const std::vector<cv::detail::ImageFeatures>& features, std::vector<cv::detail::MatchesInfo>& pairwise_matches,
		const cv::UMat& mask = cv::UMat());

	std::vector<cv::detail::ImageFeatures> features() { return features_; };
	std::vector<cv::detail::MatchesInfo> matchinfo() { return pairwise_matches_; };

};
