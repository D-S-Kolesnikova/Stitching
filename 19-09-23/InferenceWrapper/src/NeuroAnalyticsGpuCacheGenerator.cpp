#include <InferenceWrapper/InferenceEngine.h>
#include <ItvCvUtils/Log.h>
#include <ItvCvUtils/Envar.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/detail/utf8_codecvt_facet.hpp>
#include <boost/program_options.hpp>

#include <fmt/core.h>

#include <iostream>
#include <string>

template <itvcvAnalyzerType T>
void CreateEngine(const InferenceWrapper::EngineCreationParams& engineParams)
{
    itvcvError error{};
    auto engine = InferenceWrapper::IInferenceEngine<T>::Create(error, engineParams);
    if (!(engine && error == itvcvErrorSuccess))
    {
        throw std::runtime_error("Failed to create and cache engine.");
    }
}

int main(int argc, char* argv[])
{
    namespace bpo = boost::program_options;
    namespace bfs = boost::filesystem;

#ifdef _WIN32
    setlocale(LC_ALL, ".1251");
#endif

    bpo::options_description options("Options");
    options.add_options()
        ("gpu-ordinal,g", bpo::value<uint32_t>()->required(), "Gpu ordinal according to nvidia-smi.")
        ("ann-path,p", bpo::value<std::string>()->required(), "Path to *.ann file.")
        ("int8", bpo::value<bool>()->default_value(true), "Allow int8 callibration.")
        ("verbose,v", "Verbose logging.");

    bpo::options_description desc("NeuroAnalyticsGpuCacheGenerator");
    desc.add_options()("help,h", "Print help");
    desc.add(options);

    bpo::variables_map vm;
    try
    {
        if (ItvCvUtils::CEnvar::GpuCacheDir().empty())
        {
            fmt::print("Please add system enviromental variable GPU_CACHE_DIR that points to an accessible directory and try again.\n");
            return -1;
        }

        bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        bpo::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }

        boost::system::error_code ec;
        auto inputFilePath = boost::filesystem::path(vm["ann-path"].as<std::string>());
        if (!(bfs::is_regular_file(inputFilePath, ec) && inputFilePath.extension() == ".ann"))
        {
            fmt::print("Invalid *.ann file, error:{}\n", ec.message());
            return -1;
        }

        itvcvError error{};
        auto logger = ItvCv::Utils::CreateStreamLogger(std::cout, vm.count("verbose") > 0 ? ITV8::LOG_DEBUG : ITV8::LOG_ERROR);
        auto netInfo = ItvCv::GetNetworkInformation(inputFilePath.string(bfs::detail::utf8_codecvt_facet()).c_str());
        auto analyzerType = netInfo->commonParams.analyzerType;
        InferenceWrapper::EngineCreationParams engineParams(
            logger.get(),
            analyzerType,
            itvcvModeType::itvcvModeGPU,
            16,
            static_cast<int>(vm["gpu-ordinal"].as<uint32_t>()),
            "",
            inputFilePath.string(bfs::detail::utf8_codecvt_facet()).c_str(),
            "",
            vm["int8"].as<bool>()
        );

        switch (analyzerType)
        {
        case itvcvAnalyzerClassification:
            CreateEngine<itvcvAnalyzerClassification>(engineParams);
            break;
        case itvcvAnalyzerSSD:
            CreateEngine<itvcvAnalyzerSSD>(engineParams);
            break;
        case itvcvAnalyzerSiamese:
            CreateEngine<itvcvAnalyzerSiamese>(engineParams);
            break;
        case itvcvAnalyzerHumanPoseEstimator:
            CreateEngine<itvcvAnalyzerHumanPoseEstimator>(engineParams);
            break;
        case itvcvAnalyzerMaskSegments:
            CreateEngine<itvcvAnalyzerMaskSegments>(engineParams);
            break;
        default:
            fmt::print("Unsupported ann file yet.\n");
            return -1;
        }
    }
    catch (const std::exception& e)
    {
        fmt::print("{}\n\n", e.what());
        std::cout << desc << std::endl;
        return -1;
    }

    return 0;
}