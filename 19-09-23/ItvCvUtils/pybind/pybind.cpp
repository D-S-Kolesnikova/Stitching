#include "pybind.h"
#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/NeuroAnalyticsApi.h>
#include <ItvCvUtils/Pose.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

//

namespace
{
    std::vector<TextIOWrapper> streamsOfCreateStreamLogger;

    enum LOG_level
    {
        LOG_DEBUG = ITV8::LOG_DEBUG,
        LOG_INFO = ITV8::LOG_INFO,
        LOG_WARNING = ITV8::LOG_WARNING,
        LOG_ERROR = ITV8::LOG_ERROR
    };
}

//

std::shared_ptr<ITV8::ILogger> get_simple_stdout_logger()
{
    class SimpleStdoutLogger : public ITV8::ILogger
    {
        ITV8_BEGIN_CONTRACT_MAP()
            ITV8_CONTRACT_ENTRY(ITV8::IContract)
            ITV8_CONTRACT_ENTRY(ITV8::ILogger)
        ITV8_END_CONTRACT_MAP()

    public:
        uint32_t GetLogLevel() const override
        {
            return 0;
        }

        void Log(uint32_t level, const char* message) override
        {
            std::cout << "SimpleStdoutLogger: " << message << std::endl;
        }

        void Log(uint32_t level, const char* file, uint32_t line, const char* function, const char* message) override
        {
            Log(level, message);
        }
    };

    return std::make_shared<SimpleStdoutLogger>();
}

template <typename T>
class NodeleteSharedPtr
{
private:
    struct PtrProxy
    {
        explicit PtrProxy(T* p)
            : p(p) {}
        T* p;
    };

public:
    explicit NodeleteSharedPtr(T* p)
        : ptrProxy(std::make_shared<PtrProxy>(p)) {}

    T* get() const
    {
        return this->ptrProxy;
    }

private:
    std::shared_ptr<PtrProxy> ptrProxy;
};
PYBIND11_DECLARE_HOLDER_TYPE(T, NodeleteSharedPtr<T>);

//

#define GetConstKeypointsVectorFunc(f) (const std::vector<ITV8::PointF>& (ItvCvUtils::Pose::*) () const)(&f)
#define GetConstKeypointFunc(f) (const ITV8::PointF& (ItvCvUtils::Pose::*) () const)(&f)

PYBIND11_MODULE(ItvCvUtilsPyBind, m) {
    m.doc() = R"pbdoc(ItvCvUtils binding)pbdoc";

    // ItvCvUtilsPyBind attributes:

    py::enum_<itvcvModeType>(m, "itvcvModeType")
        .value("GPU", itvcvModeType::itvcvModeGPU)
        .value("CPU", itvcvModeType::itvcvModeCPU)
        .value("IntelGPU", itvcvModeType::itvcvModeIntelGPU);
    py::enum_<itvcvAnalyzerType>(m, "itvcvAnalyzerType")
        .value("HumanPoseEstimator", itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator)
        .value("Classification", itvcvAnalyzerType::itvcvAnalyzerClassification)
        .value("ReID", itvcvAnalyzerType::itvcvAnalyzerSiamese)
        .value("SemanticSegmentation", itvcvAnalyzerType::itvcvAnalyzerMaskSegments)
        .value("Detection", itvcvAnalyzerType::itvcvAnalyzerSSD)
        .export_values();

    py::enum_<ClassReportType>(m, "ClassReportType")
        .value("Report", ClassReportType::itvcvNeedReport)
        .value("NotReport", ClassReportType::itvcvNeedNotReport)
        .value("Skip", ClassReportType::itvcvNeedSkip)
        .export_values();

    py::enum_<itvcvError>(m, "itvcvError")
        .value("Success", itvcvError::itvcvErrorSuccess);

    py::enum_<Pose::HumanPosePointType>(m, "HumanPosePointType")
        .value("humanPointUnknown", Pose::HumanPosePointType::humanPointUnknown)
        .value("humanPointNose", Pose::HumanPosePointType::humanPointNose)
        .value("humanPointNeck", Pose::HumanPosePointType::humanPointNeck)
        .value("humanPointRightShoulder", Pose::HumanPosePointType::humanPointRightShoulder)
        .value("humanPointRightElbow", Pose::HumanPosePointType::humanPointRightElbow)
        .value("humanPointRightWrist", Pose::HumanPosePointType::humanPointRightWrist)
        .value("humanPointLeftShoulder", Pose::HumanPosePointType::humanPointLeftShoulder)
        .value("humanPointLeftElbow", Pose::HumanPosePointType::humanPointLeftElbow)
        .value("humanPointLeftWrist", Pose::HumanPosePointType::humanPointLeftWrist)
        .value("humanPointRightHip", Pose::HumanPosePointType::humanPointRightHip)
        .value("humanPointRightKnee", Pose::HumanPosePointType::humanPointRightKnee)
        .value("humanPointRightAnkle", Pose::HumanPosePointType::humanPointRightAnkle)
        .value("humanPointLeftHip", Pose::HumanPosePointType::humanPointLeftHip)
        .value("humanPointLeftKnee", Pose::HumanPosePointType::humanPointLeftKnee)
        .value("humanPointLeftAnkle", Pose::HumanPosePointType::humanPointLeftAnkle)
        .value("humanPointRightEye", Pose::HumanPosePointType::humanPointRightEye)
        .value("humanPointLeftEye", Pose::HumanPosePointType::humanPointLeftEye)
        .value("humanPointRightEar", Pose::HumanPosePointType::humanPointRightEar)
        .value("humanPointLeftEar", Pose::HumanPosePointType::humanPointLeftEar)
        .export_values();

    py::enum_<ObjectType>(m, "ObjectType")
        .value("Noise", ObjectType::itvcvObjectNoise)
        .value("Human", ObjectType::itvcvObjectHuman)
        .value("GroupOfHumans", ObjectType::itvcvObjectGroupOfHumans)
        .value("Vehicle", ObjectType::itvcvObjectVehicle)
        .value("Reserved", ObjectType::itvcvObjectReserved)
        .value("Face", ObjectType::itvcvObjectFace)
        .value("Animal", ObjectType::itvcvObjectAnimal)
        .value("RobotDog", ObjectType::itvcvObjectRobotDog)
        .value("Smoke", ObjectType::itvcvObjectSmoke)
        .value("Fire", ObjectType::itvcvObjectFire)
        .value("Car", ObjectType::itvcvObjectCar)
        .value("Motorcycle", ObjectType::itvcvObjectMotorcycle)
        .value("Bus", ObjectType::itvcvObjectBus)
        .value("Bicyclist", ObjectType::itvcvObjectBicyclist)
        .value("Sack", ObjectType::itvcvObjectSack)
        .value("Box", ObjectType::itvcvObjectBox)
        .value("GasBottle", ObjectType::itvcvObjectGasBottle)
        .value("Child", ObjectType::itvcvObjectChild)
        .value("Cat", ObjectType::itvcvObjectCat)
        .value("Other", ObjectType::itvcvObjectOther)
        .value("Undefined", ObjectType::itvcvObjectUndefined)
        .export_values();

    py::enum_<Segments::BodySegment>(m, "BodySegment")
        .value("Unknown", Segments::BodySegment::Unknown)
        .value("Body", Segments::BodySegment::Body)
        .value("Hips", Segments::BodySegment::Hips)
        .value("Shin", Segments::BodySegment::Shin)
        .value("Foot", Segments::BodySegment::Foot)
        .value("Face", Segments::BodySegment::Face)
        .value("Hand", Segments::BodySegment::Hand)
        .value("Shoulder", Segments::BodySegment::Shoulder)
        .value("Forearm", Segments::BodySegment::Forearm)
        .value("All", Segments::BodySegment::All)
        .value("Head", Segments::BodySegment::Head)
        .export_values();

    // ItvCvUtilsPyBind.ITV8 attributes:

    auto m_itv8 = m.def_submodule("ITV8");
    py::enum_<LOG_level>(m_itv8, "LOG_level")
        .value("LOG_DEBUG", LOG_level::LOG_DEBUG)
        .value("LOG_INFO", LOG_level::LOG_INFO)
        .value("LOG_WARNING", LOG_level::LOG_WARNING)
        .value("LOG_ERROR", LOG_level::LOG_ERROR)
        .export_values();

    py::class_<ITV8::PointF>(m_itv8, "PointF")
        .def(py::init<double_t, double_t>())
        .def_readwrite("x", &ITV8::PointF::x)
        .def_readwrite("y", &ITV8::PointF::y);

    py::class_<ITV8::ILogger, NodeleteSharedPtr<ITV8::ILogger>>(m_itv8, "ILogger")
        .def("Log", [](ITV8::ILogger& logger, uint32_t level, const char* message) { logger.Log(level, message); });

    // ItvCvUtilsPyBind.ItvCvUtils attributes:

    auto m_itvCvUtils = m.def_submodule("ItvCvUtils");
    py::class_<ItvCvUtils::Pose>(m_itvCvUtils, "Pose")
        .def(py::init<>())
        .def_property_readonly("keypoints", GetConstKeypointsVectorFunc(ItvCvUtils::Pose::keypoints))
        .def_property_readonly("nose", GetConstKeypointFunc(ItvCvUtils::Pose::nose))
        .def_property_readonly("neck", GetConstKeypointFunc(ItvCvUtils::Pose::neck))
        .def_property_readonly("rightShoulder", GetConstKeypointFunc(ItvCvUtils::Pose::rightShoulder))
        .def_property_readonly("rightElbow", GetConstKeypointFunc(ItvCvUtils::Pose::rightElbow))
        .def_property_readonly("rightWrist", GetConstKeypointFunc(ItvCvUtils::Pose::rightWrist))
        .def_property_readonly("leftShoulder", GetConstKeypointFunc(ItvCvUtils::Pose::leftShoulder))
        .def_property_readonly("leftElbow", GetConstKeypointFunc(ItvCvUtils::Pose::leftElbow))
        .def_property_readonly("leftWrist", GetConstKeypointFunc(ItvCvUtils::Pose::leftWrist))
        .def_property_readonly("rightHip", GetConstKeypointFunc(ItvCvUtils::Pose::rightHip))
        .def_property_readonly("rightKnee", GetConstKeypointFunc(ItvCvUtils::Pose::rightKnee))
        .def_property_readonly("rightAnkle", GetConstKeypointFunc(ItvCvUtils::Pose::rightAnkle))
        .def_property_readonly("leftHip", GetConstKeypointFunc(ItvCvUtils::Pose::leftHip))
        .def_property_readonly("leftKnee", GetConstKeypointFunc(ItvCvUtils::Pose::leftKnee))
        .def_property_readonly("leftAnkle", GetConstKeypointFunc(ItvCvUtils::Pose::leftAnkle))
        .def_property_readonly("leftEye", GetConstKeypointFunc(ItvCvUtils::Pose::leftEye))
        .def_property_readonly("rightEye", GetConstKeypointFunc(ItvCvUtils::Pose::rightEye))
        .def_property_readonly("rightEar", GetConstKeypointFunc(ItvCvUtils::Pose::rightEar))
        .def_property_readonly("leftEar", GetConstKeypointFunc(ItvCvUtils::Pose::leftEar));

    // ItvCvUtilsPyBind.ItvCv.Utils attributes:

    auto m_itvCv = m.def_submodule("ItvCv");
    auto m_itvCv_Utils = m_itvCv.def_submodule("Utils");
    m_itvCv_Utils.def("CreateStreamLogger",
        [](TextIOWrapper& stream, uint32_t logLevel)
        {
            streamsOfCreateStreamLogger.emplace_back(stream);
            auto& storedStream = streamsOfCreateStreamLogger.back();
            return ItvCv::Utils::CreateStreamLogger(storedStream, logLevel);
        },
        R"pbdoc(
            :param TextIOWrapper& arg0: `stream`: Any oject inherits :class:`io.TextIOWrapper`.
            :param uint32_t arg1: `logLevel`: Loglevel from set of `ItvCvUtilsPyBind.ITV8.LOG_*`.
        )pbdoc"
    );

    py::module_::import("atexit").attr("register")(py::cpp_function([]() {
        streamsOfCreateStreamLogger.clear();
    }));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
