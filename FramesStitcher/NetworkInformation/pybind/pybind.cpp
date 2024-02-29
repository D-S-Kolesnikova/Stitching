#include <NetworkInformation/NetworkInformationLib.h>
#include <NetworkInformation/Utils.h>
#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>

#include <fmt/format.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <codecvt>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

//PYBIND11_MAKE_OPAQUE(std::vector<ItvCv::NormalizationValue>);
// `boost::variant` as an example -- can be any `std::variant`-like container
namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<boost::variant<Ts...>> : variant_caster<boost::variant<Ts...>> {};

// Specifies the function used to visit the variant -- `apply_visitor` instead of `visit`
template <>
struct visit_helper<boost::variant> {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(boost::apply_visitor(args...)) {
        return boost::apply_visitor(args...);
    }
};

}
} // namespace pybind11::detail
namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};

template <>
struct type_caster<boost::blank> :  void_caster<boost::blank>{};
}
}

namespace py = pybind11;

py::bytes PyConsumeWeightsData(ItvCv::PNetworkInformation const& netInfo)
{
    auto cppStr = ItvCv::ConsumeWeightsData(netInfo);
    return py::bytes(cppStr);
}
bool PyDumpNetworkInformationToAnn(
    const py::bytes& weightsData,
    const ItvCv::NetworkInformation& netInfo,
    const std::string& pathOut,
    std::int64_t byteSize = 1024)
{
    auto cppStr = std::string(weightsData);
    return ItvCv::DumpNetworkInformationToAnn(cppStr, netInfo, pathOut, byteSize);
}

py::tuple DecryptAnn(const std::string& pathToFile)
{
    auto netInfo = ItvCv::GetNetworkInformation(pathToFile.c_str());
    return py::make_tuple(
        PyConsumeWeightsData(netInfo),
        std::move(netInfo));
}

ItvCv::NetworkInformation::NetParams_t CreateEmptyNetParams(const itvcvAnalyzerType analyzerType, const ItvCv::ArchitectureType architectureType)
{
    ItvCv::NetworkInformation::NetParams_t emptyData;
    switch (analyzerType)
    {
    case itvcvAnalyzerType::itvcvAnalyzerClassification:
    case itvcvAnalyzerType::itvcvAnalyzerSSD:
    {
        emptyData = ItvCv::NetworkInformation::Labels_t();
        break;
    }
    case itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator:
    {
        ItvCv::PoseNetworkParams poseParameters;
        if (architectureType == ItvCv::ArchitectureType::Openpose18_MobileNet)
        {
            poseParameters.params = ItvCv::OpenPoseParams();
        }
        else if (architectureType == ItvCv::ArchitectureType::HigherHRNet_AE)
        {
            poseParameters.params = ItvCv::AEPoseParams();
        }
        else
        {
            throw std::runtime_error(fmt::format("Unsupported pose architectureType:{}", architectureType));
        }
        emptyData = std::move(poseParameters);
        break;
    }
    case itvcvAnalyzerType::itvcvAnalyzerSiamese:
    {
        emptyData = ItvCv::ReidParams();
        break;
    }
    case itvcvAnalyzerType::itvcvAnalyzerMaskSegments:
    {
        emptyData = ItvCv::SemanticSegmentationParameters();
        break;
    }
    default:
        throw std::runtime_error(fmt::format("Unsupported analyzerType: {}", analyzerType));
    }

    return emptyData;
}

PYBIND11_MODULE(NetworkInformationPyBind, m) {
    m.doc() = R"pbdoc(NetworkInformation binding)pbdoc";

    py::enum_<ItvCv::DataType>(m, "DataType")
        .value("Unknown", ItvCv::DataType::Unknown)
        .value("FP16", ItvCv::DataType::FP16)
        .value("FP32", ItvCv::DataType::FP32)
        .value("INT8", ItvCv::DataType::INT8)
        .export_values();
    py::enum_<ItvCv::ModelRepresentation>(m, "ModelRepresentation")
        .value("unknown", ItvCv::ModelRepresentation::unknown)
        .value("caffe", ItvCv::ModelRepresentation::caffe)
        .value("onnx", ItvCv::ModelRepresentation::onnx)
        .value("ascend", ItvCv::ModelRepresentation::ascend)
        .value("openvino", ItvCv::ModelRepresentation::openvino)
        .export_values();
    py::enum_<ItvCv::PixelFormat>(m, "PixelFormat")
        .value("Unspecified", ItvCv::PixelFormat::Unspecified)
        .value("RGB", ItvCv::PixelFormat::RGB)
        .value("BGR", ItvCv::PixelFormat::BGR)
        .value("NV12", ItvCv::PixelFormat::NV12)
        .export_values();
    py::enum_<ItvCv::SubTypes>(m, "SubTypes")
        .value("None", ItvCv::SubTypes::None)
        .value("BodySegments", ItvCv::SubTypes::BodySegments)
        .export_values();

    py::enum_<ItvCv::ArchitectureType>(m, "ArchitectureType")
        .value("Unknown", ItvCv::ArchitectureType::Unknown)
        .value("GoogleNet", ItvCv::ArchitectureType::GoogleNet)
        .value("EfficientNet", ItvCv::ArchitectureType::EfficientNet)
        .value("SSD_MobileNetv2", ItvCv::ArchitectureType::SSD_MobileNetv2)
        .value("Yolo", ItvCv::ArchitectureType::Yolo)
        .value("SSD_ResNet34", ItvCv::ArchitectureType::SSD_ResNet34)
        .value("SSD_PyTorch", ItvCv::ArchitectureType::SSD_PyTorch)
        .value("Openpose18_MobileNet", ItvCv::ArchitectureType::Openpose18_MobileNet)
        .value("HigherHRNet_AE", ItvCv::ArchitectureType::HigherHRNet_AE)
        .value("CustomSegmentation9_v1", ItvCv::ArchitectureType::CustomSegmentation9_v1)
        .value("DDRNet", ItvCv::ArchitectureType::DDRNet)
        .value("Osnetfpn", ItvCv::ArchitectureType::Osnetfpn)
        .export_values();

    py::enum_<ItvCv::ResizePolicy>(m, "ResizePolicy")
        .value("Unspecified", ItvCv::ResizePolicy::Unspecified)
        .value("AsIs", ItvCv::ResizePolicy::AsIs)
        .value("ProportionalWithRepeatedBorders", ItvCv::ResizePolicy::ProportionalWithRepeatedBorders)
        .value("ProportionalWithPaddedBorders", ItvCv::ResizePolicy::ProportionalWithPaddedBorders)
        .export_values();

    py::class_<ItvCv::ComplexLabelType>(m, "ComplexLabelType")
        .def(py::init<>())
        .def(py::init<ObjectType, ItvCv::ComplexLabelType::SubTypes_t>())
        .def(py::init<ObjectType>())
        .def("GetSubtype", &ItvCv::ComplexLabelType::GetSubtype<ItvCv::SubTypes::BodySegments>)
        .def("GetSubtype", &ItvCv::ComplexLabelType::GetSubtype<ItvCv::SubTypes::None>)
        .def("WichSubtype", &ItvCv::ComplexLabelType::WichSubtype)
        .def_readwrite("subType", &ItvCv::ComplexLabelType::subType)
        .def_readwrite("objectType", &ItvCv::ComplexLabelType::objectType);

    py::class_<ItvCv::ReidParams>(m, "ReidParams")
        .def(py::init<>())
        .def_readwrite("type", &ItvCv::ReidParams::type)
        .def_readwrite("version", &ItvCv::ReidParams::version)
        .def_readwrite("datasetGroup", &ItvCv::ReidParams::datasetGroup)
        .def_readwrite("vectorSize", &ItvCv::ReidParams::vectorSize);

    py::class_<ItvCv::Label>(m, "Label")
        .def(py::init<>())
        .def_readwrite("name", &ItvCv::Label::name)
        .def_readwrite("position", &ItvCv::Label::position)
        .def_readwrite("type", &ItvCv::Label::type)
        .def_readwrite("reportType", &ItvCv::Label::reportType);

    py::class_<ItvCv::SemanticSegmentationParameters>(m, "SemanticSegmentationParameters")
        .def(py::init<>())
        .def_readwrite("labels", &ItvCv::SemanticSegmentationParameters::labels)
        .def_readwrite("isSingleChannel", &ItvCv::SemanticSegmentationParameters::isSingleChannel);

    py::class_<ItvCv::OpenPoseParams>(m, "OpenPoseParams")
        .def(py::init<>())
        .def_readwrite("boxSize", &ItvCv::OpenPoseParams::boxSize)
        .def_readwrite("stride", &ItvCv::OpenPoseParams::stride)
        .def_readwrite("minPeaksDistance", &ItvCv::OpenPoseParams::minPeaksDistance)
        .def_readwrite("midPointsScoreThreshold", &ItvCv::OpenPoseParams::midPointsScoreThreshold)
        .def_readwrite("midPointsRatioThreshold", &ItvCv::OpenPoseParams::midPointsRatioThreshold)
        .def_readwrite("upSampleRatio", &ItvCv::OpenPoseParams::upSampleRatio);

    py::class_<ItvCv::AEPoseParams>(m, "AEPoseParams")
        .def(py::init<>())
        .def_readwrite("delta", &ItvCv::AEPoseParams::delta)
        .def_readwrite("peakThreshold", &ItvCv::AEPoseParams::peakThreshold)
        .def_readwrite("tagsThreshold", &ItvCv::AEPoseParams::tagsThreshold)
        .def_readwrite("confidenceThreshold", &ItvCv::AEPoseParams::confidenceThreshold)
        .def_readwrite("maxObjects", &ItvCv::AEPoseParams::maxObjects);

    py::class_<ItvCv::PoseLinker::SPafElement>(m, "SPafElement")
        .def(py::init<>())
        .def_readwrite("idChannel", &ItvCv::PoseLinker::SPafElement::idChannel)
        .def_readwrite("idPointFrom", &ItvCv::PoseLinker::SPafElement::idPointFrom)
        .def_readwrite("idPointTo", &ItvCv::PoseLinker::SPafElement::idPointTo);

    py::class_<ItvCv::PoseLinker>(m, "PoseLinker")
        .def(py::init<>())
        .def_readwrite("paf", &ItvCv::PoseLinker::paf)
        .def_readwrite("heatmap", &ItvCv::PoseLinker::heatmap);

    py::class_<ItvCv::NormalizationValue>(m, "NormalizationValue")
        .def(py::init<>())
        .def_readwrite("mean", &ItvCv::NormalizationValue::mean)
        .def_readwrite("scale", &ItvCv::NormalizationValue::scale);

    py::class_<ItvCv::PoseNetworkParams>(m, "PoseNetworkParams")
        .def(py::init<>())
        .def_readwrite("linker", &ItvCv::PoseNetworkParams::linker)
        .def_readwrite("params", &ItvCv::PoseNetworkParams::params)
        .def_readwrite("minSubsetScore", &ItvCv::PoseNetworkParams::minSubsetScore)
        .def_readwrite("type", &ItvCv::PoseNetworkParams::type);

    py::class_<ItvCv::CommonParams>(m, "CommonParams")
        .def(py::init<>())
        .def_readwrite("architecture", &ItvCv::CommonParams::architecture)
        .def_readwrite("analyzerType", &ItvCv::CommonParams::analyzerType)
        .def_readwrite("modelRepresentation", &ItvCv::CommonParams::modelRepresentation)
        .def_readwrite("weightType", &ItvCv::CommonParams::weightType);

    py::class_<ItvCv::InputParams>(m, "InputParams")
        .def(py::init<>())
        .def_readwrite("numChannels", &ItvCv::InputParams::numChannels)
        .def_readwrite("inputWidth", &ItvCv::InputParams::inputWidth)
        .def_readwrite("inputHeight", &ItvCv::InputParams::inputHeight)
        .def_readwrite("supportedDynamicBatch", &ItvCv::InputParams::supportedDynamicBatch)
        .def_readwrite("resizePolicy", &ItvCv::InputParams::resizePolicy)
        .def_readwrite("pixelFormat", &ItvCv::InputParams::pixelFormat)
        .def_readwrite("normalizationValues", &ItvCv::InputParams::normalizationValues);

    py::class_<ItvCv::ModelDescription>(m, "ModelDescription")
        .def(py::init<>())
        .def_readwrite("author", &ItvCv::ModelDescription::author)
        .def_readwrite("task", &ItvCv::ModelDescription::task)
        .def_readwrite("info", &ItvCv::ModelDescription::info)
        .def_readwrite("metadataVersion", &ItvCv::ModelDescription::metadataVersion);

    py::class_<ItvCv::Version>(m, "Version")
        .def(py::init<>())
        .def_readwrite("major", &ItvCv::Version::major)
        .def_readwrite("minor", &ItvCv::Version::minor)
        .def_readwrite("patch", &ItvCv::Version::patch);

    py::class_<ItvCv::NetworkInformation, ItvCv::PNetworkInformation>(m, "NetworkInformation")
        .def(py::init<>())
        .def_readwrite("modelData", &ItvCv::NetworkInformation::modelData)
        .def_readwrite("inputParams", &ItvCv::NetworkInformation::inputParams)
        .def_readwrite("commonParams", &ItvCv::NetworkInformation::commonParams)
        .def_readwrite("networkParams", &ItvCv::NetworkInformation::networkParams)
        .def_readwrite("description", &ItvCv::NetworkInformation::description);

    py::enum_<NetworkInfoUtils::ValidationError>(m, "ValidationError")
        .value("NoError", NetworkInfoUtils::ValidationError::NoError)
        .value("Other", NetworkInfoUtils::ValidationError::Other)
        .value("BadNetworkInformation", NetworkInfoUtils::ValidationError::BadNetworkInformation)
        .value("BadInputParams", NetworkInfoUtils::ValidationError::BadInputParams)
        .value("BadCommonParams", NetworkInfoUtils::ValidationError::BadCommonParams)
        .value("BadLabelsParams", NetworkInfoUtils::ValidationError::BadLabelsParams)
        .value("BadReidParams", NetworkInfoUtils::ValidationError::BadReidParams)
        .value("BadSemanticSegmentationParameters", NetworkInfoUtils::ValidationError::BadSemanticSegmentationParameters)
        .value("BadPoseNetworkParams", NetworkInfoUtils::ValidationError::BadPoseNetworkParams)
        .value("BadOpenPoseParams", NetworkInfoUtils::ValidationError::BadOpenPoseParams)
        .value("BadAEPoseParams", NetworkInfoUtils::ValidationError::BadAEPoseParams)
        .export_values();

    m.def("ValidationParameters", py::overload_cast<const ItvCv::NetworkInformation&>(&NetworkInfoUtils::ValidationParameters<ItvCv::NetworkInformation>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::InputParams&>(&NetworkInfoUtils::ValidationParameters<ItvCv::InputParams>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::CommonParams&>(&NetworkInfoUtils::ValidationParameters<ItvCv::CommonParams>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::NetworkInformation::Labels_t&>(&NetworkInfoUtils::ValidationParameters<ItvCv::NetworkInformation::Labels_t>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::PoseNetworkParams&>(&NetworkInfoUtils::ValidationParameters<ItvCv::PoseNetworkParams>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::AEPoseParams&>(&NetworkInfoUtils::ValidationParameters<ItvCv::AEPoseParams>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::OpenPoseParams&>(&NetworkInfoUtils::ValidationParameters<ItvCv::OpenPoseParams>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::ReidParams&>(&NetworkInfoUtils::ValidationParameters<ItvCv::ReidParams>));
    m.def("ValidationParameters", py::overload_cast<const ItvCv::SemanticSegmentationParameters&>(&NetworkInfoUtils::ValidationParameters<ItvCv::SemanticSegmentationParameters>));

    m.def("CreateEmptyNetParams", CreateEmptyNetParams);

    m.def("GenerateMetadata", &ItvCv::GenerateMetadata);
    m.def("DecryptAnn", &DecryptAnn);
    m.def("ConsumeWeightsData", &PyConsumeWeightsData);
    m.def("GetNetworkInformation", &ItvCv::GetNetworkInformation);
    m.def("DumpNetworkInformationToAnn", &PyDumpNetworkInformationToAnn, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg("byteSize") = 1024);
    m.def("GetMetadataParserVersion", &ItvCv::GetMetadataParserVersion);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
