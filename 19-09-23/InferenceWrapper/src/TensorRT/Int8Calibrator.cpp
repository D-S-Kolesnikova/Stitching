#include "Int8Calibrator.h"

#include <ItvCvUtils/Log.h>
#include <cryptoWrapper/cryptoWrapperLib.h>

#include <fmt/format.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace InferenceWrapper
{

Int8EntropyCalibrator2::Int8EntropyCalibrator2(
    ITV8::ILogger* logger,
    int batchSize,
    const cv::Size& geometry,
    const cv::Scalar& meanValues,
    double scaleFactor,
    ItvCv::PixelFormat pixelFormat,
    const std::string& calibrationDataPath)
    : m_logger(logger)
    , m_batchSize(batchSize)
    , m_geometry(geometry)
    , m_meanValues(meanValues)
    , m_scaleFactor(scaleFactor)
    , m_imgIdx(0)
{
    const auto inputByteSize = 3 * geometry.width * geometry.height * batchSize * sizeof(float);
    for (auto i = 0; i < m_batchSize; ++i)
    {
        m_gpuBuffers.emplace_back(inputByteSize);
    }

    auto cryptoWrapper = ItvCv::CreateCryptoWrapper(nullptr);
    auto buffers = cryptoWrapper->GetDecryptedContent(calibrationDataPath);

    m_images.reserve(buffers.size());
    for (auto& buf : buffers)
    {
        int w, h, c;
        auto data = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>(
            stbi_load_from_memory(
                reinterpret_cast<const unsigned char*>(buf.second.c_str()),
                buf.second.size(),
                &w, &h, &c, 0),
            &stbi_image_free);

        if (!data)
        {
            throw std::runtime_error("Failed to unwrap sources for int8 optimization.");
        }

        cv::Mat image(h, w, CV_8UC3, data.get(), 0);

        if (pixelFormat == ItvCv::PixelFormat::RGB)
        {
            image = image.clone();
        }
        else
        {
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        }

        m_images.emplace_back(std::move(image));
    }
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return m_batchSize;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* /* names */[], int /* nbBindings */) noexcept
{
    if (m_imgIdx + m_batchSize > static_cast<int>(m_images.size()))
    {
        return false;
    }

    std::vector<cv::Mat> inputImages;
    for (int i = m_imgIdx; i < m_imgIdx + m_batchSize; i++)
    {
        inputImages.emplace_back(m_images[i]);
    }
    m_imgIdx += m_batchSize;

    auto blob = cv::dnn::blobFromImages(inputImages, m_scaleFactor, m_geometry, m_meanValues);

    const auto inputByteSize = 3 * m_geometry.width * m_geometry.height * m_batchSize * sizeof(float);

    for (auto i = 0; i < m_batchSize; ++i)
    {
        auto code = cudaMemcpy(m_gpuBuffers[i].data(), blob.ptr<float>(0), inputByteSize, cudaMemcpyHostToDevice);
        if (code != cudaSuccess)
        {
            ITVCV_LOG(m_logger, ITV8::LOG_ERROR,
                fmt::format("{}; Code: {}; Name: {}; Msg: {}",
                    code,
                    cudaGetErrorName(code),
                    cudaGetErrorString(code)));
            return false;
        }
        bindings[i] = m_gpuBuffers[i].data();
    }

    return true;
}

const void * Int8EntropyCalibrator2::readCalibrationCache(size_t& /* length */) noexcept
{
    return nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* /* cache */, size_t /* length */) noexcept
{
}

}