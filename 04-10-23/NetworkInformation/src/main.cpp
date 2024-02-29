#include <NetworkInformation/NetworkInformationLib.h>
#include <fmt/format.h>

const char* PixelFormatToStr(const ItvCv::PixelFormat& v)
{
    switch (v)
    {
    case ItvCv::PixelFormat::Unspecified:
        return "Unspecified";
    case ItvCv::PixelFormat::RGB:
        return "RGB";
    case ItvCv::PixelFormat::BGR:
        return "BGR";
    case ItvCv::PixelFormat::NV12:
        return "NV12";
    default:
        throw std::runtime_error(fmt::format("In {}; Unknown value: {}", __FUNCTION__, int(v)));
    }
}

const char* ResizePolicyToStr(const ItvCv::ResizePolicy& v)
{
    switch (v)
    {
    case ItvCv::ResizePolicy::Unspecified:
        return "Unspecified";
    case ItvCv::ResizePolicy::AsIs:
        return "AsIs";
    case ItvCv::ResizePolicy::ProportionalWithRepeatedBorders:
        return "ProportionalWithRepeatedBorders";
    case ItvCv::ResizePolicy::ProportionalWithPaddedBorders:
        return "ProportionalWithPaddedBorders";
    default:
        throw std::runtime_error(fmt::format("In {}; Unknown value: {}", __FUNCTION__, int(v)));
    }
}

const char* DataTypeToStr(const ItvCv::DataType& v)
{
    switch (v)
    {
    case ItvCv::DataType::FP16:
        return "FP16";
    case ItvCv::DataType::FP32:
        return "FP32";
    case ItvCv::DataType::INT8:
        return "INT8";
    }
    return "N/A";
}

const char* ArchitectureTypeToStr(const ItvCv::ArchitectureType& v)
{
    switch (v)
    {
    case ItvCv::ArchitectureType::Unknown:
        return "Unknown";
    case ItvCv::ArchitectureType::GoogleNet:
        return "GoogleNet";
    case ItvCv::ArchitectureType::EfficientNet:
        return "EfficientNet";
    case ItvCv::ArchitectureType::SSD_MobileNetv2:
        return "SSD_MobileNetv2";
    case ItvCv::ArchitectureType::Yolo:
        return "Yolo";
    case ItvCv::ArchitectureType::SSD_ResNet34:
        return "SSD_ResNet34";
    case ItvCv::ArchitectureType::SSD_PyTorch:
        return "SSD_PyTorch";
    case ItvCv::ArchitectureType::Openpose18_MobileNet:
        return "Openpose18_MobileNet";
    case ItvCv::ArchitectureType::CustomSegmentation9_v1:
        return "CustomSegmentation9_v1";
    default:
        throw std::runtime_error(fmt::format("In {}; Unknown value: {}", __FUNCTION__, int(v)));
    }
}

const char* ModelRepresentationToStr(const ItvCv::ModelRepresentation& v)
{
    switch (v)
    {
    case ItvCv::ModelRepresentation::openvino:
        return "openvino";
    case ItvCv::ModelRepresentation::caffe:
        return "caffe";
    case ItvCv::ModelRepresentation::onnx:
        return "onnx";
    case ItvCv::ModelRepresentation::ascend:
        return "ascend";
    default:
        throw std::runtime_error(fmt::format("In {}; Unknown value: {}", __FUNCTION__, int(v)));
    }
}

int main(int argc, char* argv[])
{
    return 0;
}
