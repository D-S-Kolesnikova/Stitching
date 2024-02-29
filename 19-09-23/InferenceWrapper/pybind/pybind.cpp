#include <InferenceWrapper/InferenceEngine.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/imgcodecs.hpp>

#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
typedef py::detail::npy_api::constants numpy_constants;

// ---

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <>
struct type_caster<cv::Mat>
{
public:
    // Python to C++
    bool load(py::handle src, bool convert)
    {
        if (!convert && !py::array::check_(src))
            return false;

        auto const arr = py::array::ensure(src);
        if (!arr)
            return false;

        auto const dtype = arr.dtype().num();
        auto const shape = arr.shape();
        auto& rows = shape[0];
        auto& cols = shape[1];
        auto& channels = shape[2];

        auto const dtype2cv =
            (dtype == numpy_constants::NPY_UBYTE_) ? CV_8U :
            (dtype == numpy_constants::NPY_BYTE_) ? CV_8S :
            (dtype == numpy_constants::NPY_USHORT_) ? CV_16U :
            (dtype == numpy_constants::NPY_SHORT_) ? CV_16S :
            (dtype == numpy_constants::NPY_INT_) ? CV_32S :
            (dtype == numpy_constants::NPY_INT32_) ? CV_32S :
            (dtype == numpy_constants::NPY_FLOAT_) ? CV_32F :
            (dtype == numpy_constants::NPY_DOUBLE_) ? CV_64F :
            -1;
        if (dtype2cv == -1)
            return false;
        auto const cv_arr_type = CV_MAKETYPE(dtype2cv, (int)channels);

        // auto buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
        // TODO: py::array::c_style ??? If numpy array stores elements diffused???

        value = cv::Mat(
            cv::Size2l(cols, rows),
            cv_arr_type,
            const_cast<void*>(arr.data()));  // TODO???: size_t step = AUTO_STEP

        return true;
    }

    // C++ to Python 
    static py::handle
    cast(const cv::Mat& src, py::return_value_policy policy, py::handle parent)
    {
        // TODO !!!!!
        //py::array a(std::move(src.shape()), std::move(src.strides(true)), src.data());
        //return a.release();
        return handle();
    }

    PYBIND11_TYPE_CASTER(cv::Mat, const_name("cv::Mat"));
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

// ---

PYBIND11_MODULE(InferenceWrapperPyBind, m) {
    m.doc() = R"pbdoc(InferenceWrapper binding)pbdoc";

    // InferenceWrapperPyBind.InferenceWrapper attributes:

    auto m_inferenceWrapper = m.def_submodule("InferenceWrapper");

    py::class_<ItvCv::Frame>(m_inferenceWrapper, "Frame", py::buffer_protocol())
        .def(py::init<int, int, int, const unsigned char*>())
        .def(py::init([](cv::Mat img) {
            return ItvCv::Frame(img.cols, img.rows, static_cast<int>(img.step), img.data); }));

    py::class_<InferenceWrapper::EngineCreationParams>(m_inferenceWrapper, "EngineCreationParams")
        .def(py::init<
            ITV8::ILogger*,
            itvcvAnalyzerType,
            itvcvModeType,
            int,
            int,
            std::string,
            std::string>());

    py::class_<InferenceWrapper::InferenceChannelParams>(m_inferenceWrapper, "InferenceChannelParams")
        .def(py::init<>());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
