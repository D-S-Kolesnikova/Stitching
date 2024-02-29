#ifndef INT8_CALIBRATOR_H
#define INT8_CALIBRATOR_H

#include "Buffers.h"

#include <NetworkInformation/NetworkInformationLib.h>

#include <NvInferPlugin.h>

#include <opencv2/opencv.hpp>

#include <vector>

namespace InferenceWrapper
{

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(
        ITV8::ILogger* logger,
        int batchSize,
        const cv::Size& geometry,
        const cv::Scalar& meanValues,
        double scaleFactor,
        ItvCv::PixelFormat pixelFormat,
        const std::string& calibrationDataPath);

    ~Int8EntropyCalibrator2() = default;
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    ITV8::ILogger* m_logger;
    int m_batchSize;
    cv::Size m_geometry;
    cv::Scalar m_meanValues;
    double m_scaleFactor;
    int m_imgIdx;
    std::vector<cv::Mat> m_images;
    std::vector<DeviceBuffer> m_gpuBuffers;
};

}

#endif