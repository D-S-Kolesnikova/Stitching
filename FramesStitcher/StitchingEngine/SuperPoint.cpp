#include"SuperPoint.h"
#include <iostream>
#include <fstream>

#include <onnxruntime_cxx_api.h>

SuperPoint::SuperPoint(std::wstring modelPath)
{
	this->m_modelPath = modelPath;
}

std::vector<float> SuperPoint::ApplyTransform(const cv::Mat& image, float& mean, float& std)
{
	cv::Mat resized, floatImage;
	image.convertTo(floatImage, CV_32FC1);
	float scale = float(floatImage.cols) / float(floatImage.rows);
	cv::resize(floatImage, resized, cv::Size(1024, int(1024. / scale)));
	std::vector<float> imgData;
	for (int h = 0; h < resized.rows; h++)
	{
		for (int w = 0; w < resized.cols; w++)
		{
			imgData.push_back(resized.at<float>(h, w) / 255.0f);
		}
	}
	return imgData;
}

void SuperPoint::detectAndCompute(cv::InputArray image, cv::InputArray mask,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors,
	bool useProvidedKeypoints)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

	cv::Mat img = image.getMat();
	cv::Mat grayImg;
	cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	float mean, std;

	cv::Mat resized, floatImage;
	grayImg.convertTo(floatImage, CV_32FC1);
	float scale = float(floatImage.cols) / floatImage.rows;
	cv::resize(floatImage, resized, cv::Size(1024, int(1024. / scale)));
	float fx = float(resized.cols) / floatImage.cols;
	float fy = float(resized.rows) / floatImage.rows;

	std::vector<float> imgData;
	for (int h = 0; h < resized.rows; h++)
	{
		for (int w = 0; w < resized.cols; w++)
		{
			imgData.push_back(resized.at<float>(h, w) / 255.0f);
		}
	}

	std::vector<int64_t> inputShape{ 1, 1, resized.rows,  resized.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
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
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	std::vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		cv::KeyPoint p;
		int index = i * 2;
		p.pt.x = ((float)kp[index] + 0.5) / fx - 0.5;
		p.pt.y = ((float)kp[index + 1] + 0.5) / fx - 0.5;
		keypoints[i] = p;
	}

	std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* des = (float*)outputs[2].GetTensorMutableData<void>();

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
void SuperPoint::detect(cv::InputArray image,
	std::vector<cv::KeyPoint>& keypoints,
	cv::InputArray mask)
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

	cv::Mat img = image.getMat();
	cv::Mat grayImg;
	cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	float mean, std;
	std::vector<float> imgData = ApplyTransform(grayImg, mean, std);
	std::vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	std::vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		cv::KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}
}
void SuperPoint::compute(cv::InputArray image,
	std::vector<cv::KeyPoint>& keypoints,
	cv::OutputArray descriptors)
{

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	static Ort::Session extractorSession(env, this->m_modelPath.c_str(), sessionOptions);

	cv::Mat img = image.getMat();
	cv::Mat grayImg;
	cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	float mean, std;

	std::vector<float> imgData = ApplyTransform(grayImg, mean, std);

	std::vector<int64_t> inputShape{ 1, 1, grayImg.rows, grayImg.cols };

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());

	const char* input_names[] = { "image" };
	const char* output_names[] = { "keypoints","scores","descriptors" };
	Ort::RunOptions run_options;
	std::vector<Ort::Value> outputs = extractorSession.Run(run_options, input_names, &inputTensor, 1, output_names, 3);

	std::vector<int64_t> kpshape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int64* kp = (int64*)outputs[0].GetTensorMutableData<void>();
	int keypntcounts = kpshape[1];
	keypoints.resize(keypntcounts);
	for (int i = 0; i < keypntcounts; i++)
	{
		cv::KeyPoint p;
		int index = i * 2;
		p.pt.x = (float)kp[index];
		p.pt.y = (float)kp[index + 1];
		keypoints[i] = p;
	}

	std::vector<int64_t> desshape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	float* des = (float*)outputs[2].GetTensorMutableData<void>();
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
