#include "InferenceWrapper/InferenceWrapperLib.h"
#include <InferenceWrapper/InferenceEngine.h>

#include <ItvCvUtils/Log.h>

#include <fmt/format.h>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <chrono>
#include <string>
#include <iostream>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <future>
#include <stack>

#include <stb_image.h>
#include <stb_image_write.h>

std::shared_ptr<ITV8::ILogger> g_logger = ItvCv::Utils::CreateStreamLogger(std::cout, ITV8::LOG_DEBUG);

struct IFrameProvider
{
    virtual bool GetFrame(cv::Mat& frame) = 0;
    virtual int GetFps() { return 40; };
    virtual bool IsValid() = 0;
    virtual void ResetState() = 0;

    ~IFrameProvider()
    {
    };
};

class VideoFrameProvider : public IFrameProvider
{
public:
    VideoFrameProvider(const std::string& filename)
        :
#ifdef ENABLE_OPENCV_GUI
        cap(filename),
#endif
        m_filename(filename)
    {
#ifdef ENABLE_OPENCV_GUI
        if (!cap.isOpened())
        {
            std::cout << "Could not open the video!" << std::endl;
            throw std::logic_error("Could not open the video!");
        }
#else
        throw std::logic_error("Couldn't open video. OpenCV highgui module is required");
#endif
    }

    void ResetState() override
    {
#ifdef ENABLE_OPENCV_GUI
        cap = cv::VideoCapture(m_filename);
#endif
    }

    bool IsValid() override
    {
#ifdef ENABLE_OPENCV_GUI
        return cap.isOpened();
#else
        return false;
#endif
    }

    bool GetFrame(cv::Mat& frame) override
    {
#ifdef ENABLE_OPENCV_GUI
        return cap.read(frame);
#else
        return false;
#endif
    }

    int GetFps() override
    {
#ifdef ENABLE_OPENCV_GUI
        return cap.get(cv::CAP_PROP_FPS);
#else
        return 0;
#endif
    }

private:
#ifdef ENABLE_OPENCV_GUI
    cv::VideoCapture cap;
#endif
    std::string m_filename;
};

class ImageFrameProvider : public IFrameProvider
{
public:
    ImageFrameProvider(const std::string& pathToDir)
        : it(pathToDir),
          m_pathToDir(pathToDir)
    {
    }

    bool IsValid() override
    {
        return it != end;
    }

    void ResetState() override
    {
        it = boost::filesystem::directory_iterator(m_pathToDir);
    }

    bool GetFrame(cv::Mat& frame) override
    {
        /* while (it != end && boost::filesystem::extension(it->path()) != ".png")
        {
            ++it;
        }*/
        if (it == end)
        {
            return false;
        }

        printf("Reading image %s\n", it->path().c_str());

#ifdef ENABLE_OPENCV_GUI
        frame = cv::imread(it->path().string());
#else
        int x,y,n;
        unsigned char *data_b = stbi_load(it->path().string().c_str(), &x, &y, &n, 0);
        cv::Mat rgb(y, x, CV_8UC3, data_b, 0);
        cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);
        stbi_image_free(data_b);
#endif

        ++it;
        return true;
    }

private:
    boost::filesystem::directory_iterator it;
    boost::filesystem::directory_iterator end;
    std::string m_pathToDir;
};

class StaticImageProvider : public IFrameProvider
{
public:
    StaticImageProvider(const std::string& imageFile)
    {
#ifdef ENABLE_OPENCV_GUI
        m_frame = cv::imread(imageFile);
#else
        int x,y,n;
        unsigned char *data_b = stbi_load(imageFile.c_str(), &x, &y, &n, 0);
        if (!data_b)
            throw std::runtime_error("Failed to read image file " + imageFile);
        cv::Mat rgb(y, x, CV_8UC3, data_b, 0);
        cv::cvtColor(rgb, m_frame, cv::COLOR_RGB2BGR);
        stbi_image_free(data_b);
#endif
    }
    StaticImageProvider(const ITV8::Size& size)
    {
        m_frame = cv::Mat::zeros(size.height, size.width, CV_8UC3) + 127;
    }

    bool IsValid() override
    {
        return true;
    }

    void ResetState() override
    {
    }

    bool GetFrame(cv::Mat& frame) override
    {
        frame = m_frame.clone();
        return true;
    }

private:
    cv::Mat m_frame;
};

std::istream& operator>>(std::istream& in, itvcvAnalyzerType& type)
{
    std::string token;
    in >> token;
    if (token == "Classification") type = itvcvAnalyzerType::itvcvAnalyzerClassification;
    else if (token == "SSD") type = itvcvAnalyzerType::itvcvAnalyzerSSD;
    else if (token == "Siamese") type = itvcvAnalyzerType::itvcvAnalyzerSiamese;
    else if (token == "HPE") type = itvcvAnalyzerType::itvcvAnalyzerHumanPoseEstimator;
    else if (token == "MaskSegments") type = itvcvAnalyzerType::itvcvAnalyzerMaskSegments;

    return in;
}

std::istream& operator>>(std::istream& in, itvcvModeType& type)
{
    std::string token;
    in >> token;

    if (token == "GPU") type = itvcvModeType::itvcvModeGPU;
    else if (token == "CPU") type = itvcvModeType::itvcvModeCPU;
    else if (token == "FPGAMovidius") type = itvcvModeType::itvcvModeFPGAMovidius;
    else if (token == "IntelGPU") type = itvcvModeType::itvcvModeIntelGPU;
    else if (token == "Hetero") type = itvcvModeType::itvcvModeHetero;
    else if (token == "HDDL") type = itvcvModeType::itvcvModeHDDL;
    else if (token == "MULTI") type = itvcvModeType::itvcvModeMULTI;
    else if (token == "Balanced") type = itvcvModeType::itvcvModeBalanced;
    else if (token == "HuaweiNPU") type = itvcvModeType::itvcvModeHuaweiNPU;

    return in;
}

namespace
{

std::stack<std::function<bool(bool eof)>> g_enterPressedHanders;
std::mutex g_enterPressedHandersMutex;
bool g_eofSeen = false;

enum class OnEnterPressedResult { Handled, Unhandled };

OnEnterPressedResult onEnterPressed(bool eof)
{
    for(;;)
    {
        std::unique_lock<std::mutex> lock(g_enterPressedHandersMutex);
        if (eof)
            g_eofSeen = true;
        if (g_enterPressedHanders.empty())
            return OnEnterPressedResult::Unhandled;
        auto handler = std::move(g_enterPressedHanders.top());
        g_enterPressedHanders.pop();
        lock.unlock();
        if (handler(eof))
            return OnEnterPressedResult::Handled;
    }
}

bool waitForEnterPressed(std::chrono::seconds timeout)
{
    auto promise = std::make_shared<std::promise<bool>>();
    auto future = promise->get_future();
    {
        std::lock_guard<std::mutex> lock(g_enterPressedHandersMutex);
        if (g_eofSeen)
            return false;
        g_enterPressedHanders.push([wp = std::weak_ptr<std::promise<bool>>(promise)](bool eof){
            if (auto promise = wp.lock())
            {
                promise->set_value(eof);
                return true;
            }
            return false;
        });
    }
    return std::future_status::ready == future.wait_for(timeout) ? !future.get() : false;
}

template <itvcvAnalyzerType T>
void PostProcess(
    const bool displayGUI,
    const cv::Mat &frame,
    const typename InferenceWrapper::AnalyzerTraits<T>::ResultType &rslts)
{
    throw std::logic_error("this should not be called with this implementation");
}



void DisplayFrame(const cv::Mat& frame)
{
    const std::chrono::seconds timeout{15};
#ifdef ENABLE_OPENCV_GUI
    cv::imshow("window", frame);
    cv::waitKey(timeout.count());
#else
    static std::atomic<int32_t> frameIndex { 0 };
    auto frameNum = frameIndex.fetch_add(1);
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    auto fname = fmt::format(FMT_STRING("frame_{}.png"), frameNum);
    stbi_write_png(fname.c_str(), frame.cols, frame.rows, 3, rgb.data, frame.cols*3);
    std::cout << fmt::format(FMT_STRING("Frame #{} inference results have been written into {}. Waiting for {}s. Press ENTER to continue immediately...\n"), frameNum, fname, timeout.count());
    if (waitForEnterPressed(timeout))
        std::cout << "Continuing...\n";
#endif
}

template <>
void PostProcess<itvcvAnalyzerClassification> (
    const bool displayGUI,
    const cv::Mat &frame,
    const InferenceWrapper::AnalyzerTraits<itvcvAnalyzerClassification>::ResultType &rslts)
{
    if (displayGUI)
    {
       cv::rectangle(
           frame,
           cv::Point(0, frame.rows - 150),
           cv::Point(15, frame.rows),
           cv::Scalar(255, 0, 0),
           -1);
       cv::rectangle(
           frame,
           cv::Point(0, frame.rows - rslts[1] * 150),
           cv::Point(15, frame.rows),
           cv::Scalar(0, 0, 255),
           -1);
       DisplayFrame(frame);
    }
}

template <>
void PostProcess<itvcvAnalyzerSSD>(
    const bool displayGUI,
    const cv::Mat &frame,
    const InferenceWrapper::AnalyzerTraits<itvcvAnalyzerSSD>::ResultType &rslts)
{
    if (displayGUI)
    {
       for (auto &rslt : rslts)
       {
           cv::rectangle(
               frame,
               cv::Point(rslt[3] * frame.cols, rslt[4] * frame.rows),
               cv::Point(rslt[5] * frame.cols, rslt[6] * frame.rows),
               cv::Scalar(255, 0, 0));
           cv::putText(
               frame,
               fmt::format("{:.2f}", rslt[2]),
               cv::Point(rslt[3] * frame.cols, rslt[4] * frame.rows),
               cv::FONT_HERSHEY_TRIPLEX,
               0.4,
               cv::Scalar(0, 255, 255));
       }
        DisplayFrame(frame);
    }
}
}

class IWTester
{
public:
    struct IWTesterOpts
    {
        std::shared_ptr<ITV8::ILogger> logger;
        size_t threads;
        std::chrono::milliseconds feedFramePeriod;
        std::chrono::seconds statPeriod;
        bool displayGui;
        std::string imageFile;
        std::string videoFile;
    };

public:
    IWTester(
        const IWTesterOpts &TesterOpts,
        const InferenceWrapper::EngineCreationParams &engineParams)
        : m_opts(TesterOpts)
        , m_engineParams(engineParams)
    {
    }

    bool LaunchWorkers()
    {
        if (m_runSwitch)
            return true;

        switch(m_engineParams.netType)
        {
        case itvcvAnalyzerClassification:
            m_func = &LaunchTest<itvcvAnalyzerClassification>;
            break;
        case itvcvAnalyzerSSD:
            m_func = &LaunchTest<itvcvAnalyzerSSD>;
            break;
        case itvcvAnalyzerSiamese:
            m_func = &LaunchTest<itvcvAnalyzerSiamese>;
            break;
        case itvcvAnalyzerHumanPoseEstimator:
            m_func = &LaunchTest<itvcvAnalyzerHumanPoseEstimator>;
            break;
        case itvcvAnalyzerMaskSegments:
            m_func = &LaunchTest<itvcvAnalyzerMaskSegments>;
            break;
        }

        for (auto i = 0; i < m_opts.threads; ++i)
        {
            m_futures.emplace_back(std::async(std::launch::async, m_func, this));
        }

        m_runSwitch.store(true);
        return true;
    }

    void KillWorkers()
    {
        m_runSwitch.store(false);
    }

private:
    template <itvcvAnalyzerType T>
    static void LaunchTest(IWTester *self)
    {
        using namespace std::chrono_literals;
        // engine
        itvcvError error{ itvcvErrorSuccess };
        auto engine = InferenceWrapper::IInferenceEngine<T>::Create(error, self->m_engineParams);
        if (error != itvcvErrorSuccess)
        {
            throw std::logic_error("unable to create engine");
        }
        // frame provider
        auto inputShape = engine->GetInputGeometry();
        std::unique_ptr<IFrameProvider> frameProvider;
        if (!self->m_opts.videoFile.empty())
        {
            frameProvider.reset(new VideoFrameProvider(self->m_opts.videoFile));
        }
        else if (!self->m_opts.imageFile.empty())
        {
            boost::system::error_code ec;
            if (boost::filesystem::is_directory(self->m_opts.imageFile, ec))
                frameProvider = std::make_unique<ImageFrameProvider>(self->m_opts.imageFile);
            else
                frameProvider.reset(new StaticImageProvider(self->m_opts.imageFile));
        }
        else
        {
            frameProvider.reset(new StaticImageProvider(inputShape));
        }
        // wait for all channels
        while (!self->m_runSwitch)
        {
            std::this_thread::sleep_for(10ms);
        }
        // working loop
        cv::Mat frame;
        auto lastTs = std::chrono::system_clock::now();
        while (self->m_runSwitch)
        {
            // get frame
            if (!frameProvider->GetFrame(frame))
        {
                frameProvider->ResetState();
                continue;
        }
            // inference
            auto rsltsFuture = engine->AsyncProcessFrame(InferenceWrapper::Frame { frame.cols, frame.rows, (int)frame.step, frame.data });
            // print stats
            auto nowTs = std::chrono::system_clock::now();
            if (nowTs - lastTs > self->m_opts.statPeriod)
            {
                engine->TakeStats(std::chrono::duration_cast<std::chrono::milliseconds>(nowTs - lastTs));
                lastTs = nowTs;
            }
            // display frame with results
            auto rslts = rsltsFuture.get();
            if (rslts.first == itvcvErrorSuccess)
            {
                PostProcess<T>(self->m_opts.displayGui, frame, rslts.second);
            }
        }
    }

private:
    const IWTesterOpts m_opts;
    const InferenceWrapper::EngineCreationParams m_engineParams;

    void (*m_func)(IWTester*);
    std::atomic<bool> m_runSwitch{false};
    std::vector<std::future<void>> m_futures;
};

int main(int argc, char** argv)
{
    using namespace std::chrono_literals;
    namespace po = boost::program_options;

    po::options_description modelDesc("Model options");
    modelDesc.add_options()
        ("net-type", po::value<itvcvAnalyzerType>()->required(), "network type {Classification, SSD, Siamese, HPE, MaskSegments}.")
        ("input-files", po::value<std::vector<std::string>>()->required()->multitoken(),
            "path to input files\n"
            "  caffe: weights, model\n"
            "  onnx: onnxFile\n"
            "  ann: annFile");

    po::options_description sysDesc("System options");
    sysDesc.add_options()
        ("device", po::value<itvcvModeType>()->required(), "device type {GPU, CPU, FPGAMovidius, IntelGPU, Hetero, HDDL, MULTI, Balanced, HuaweiNPU}.")
        ("device-ordinal", po::value<size_t>()->default_value(0), "device ordinal to be used.")
        ("threads", po::value<size_t>()->default_value(1), "threads amount to be used for calling inference concurrently.")
        ("period", po::value<size_t>()->default_value(40), "frames feeding period in milliseconds  (not implemanted yet).")
        ("stat-period", po::value<size_t>()->default_value(5), "statistics' logging period in seconds. (5 sec by default)")
        ("image-file", po::value<std::string>()->default_value(""), "path to image file as an input if needed.")
        ("video-file", po::value<std::string>()->default_value(""), "path to video file as an input if needed.")
        ("display-gui", po::value<bool>()->default_value(false), "NOTE use only if one thread is created.")
        ("log-level", po::value<uint32_t>()->default_value(0), "logging level (int) {0: debug(by default), 1: info, 2: warning, 3: error}.")
        ("log-to-file", po::value<std::string>()->default_value(""), "path to log file as an output if needed.")
        ("int8", po::value<bool>()->default_value(false), "use int8 calibration");

    po::options_description desc("InferenceWrapper");
    desc.add_options()("help", "print help");
    desc.add(sysDesc).add(modelDesc);

    po::variables_map vm;
    bool printHelpAndExit{ false };
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help") || argc == 1)
        {
            printHelpAndExit = true;
        }
    }
    catch (const std::exception& e)
    {
        printHelpAndExit = true;
        std::cout << "exception: " << e.what() << std::endl;
    }
    if (printHelpAndExit)
    {
        std::cout
            << desc
            << "\n"
            << "Usage templates:\n"
            << "  --device GPU --device-ordinal 0 --threads 1 --display-gui true --log-level 0 --net-type Classification --input-files \"E:\\axxon-networks\\smfr\\smoke_original.ann\" --video-file \"E:\\axxon-networks\\smfr\\1.avi\"\n"
            << "  --device GPU --device-ordinal 0 --threads 1 --display-gui true --log-level 0 --net-type SSD --input-files \"E:\\axxon-networks\\neuro-tracker\\neurotracker_55.ann\" --video-file \"E:\\axxon-networks\\smfr\\1_fhd_25fps_264.avi\""
            << std::endl;
        return 0;
    }

    std::shared_ptr<ITV8::ILogger> logger;
    if (!vm["log-to-file"].as<std::string>().empty())
    {
        logger = ItvCv::Utils::CreateFileLogger(vm["log-to-file"].as<std::string>(), vm["log-level"].as<uint32_t>());
    }
    else
    {
        logger = ItvCv::Utils::CreateStreamLogger(std::cout, vm["log-level"].as<uint32_t>());
    }

    auto inputFiles = vm["input-files"].as<std::vector<std::string>>();
    
    InferenceWrapper::EngineCreationParams engineParams(
        logger.get(),
        vm["net-type"].as<itvcvAnalyzerType>(),
        vm["device"].as<itvcvModeType>(),
        16,
        vm["device-ordinal"].as<size_t>(),
        inputFiles.size() > 1 ? inputFiles[1] : "",
        inputFiles.front(),
        "",
        vm["int8"].as<bool>()
        );

    IWTester::IWTesterOpts opts
    {
        logger,
        vm["threads"].as<size_t>(),
        std::chrono::milliseconds(vm["period"].as<size_t>()),
        std::chrono::seconds(vm["stat-period"].as<size_t>()),
        vm["display-gui"].as<bool>(),
        vm["image-file"].as<std::string>(),
        vm["video-file"].as<std::string>(),
    };

    IWTester tester(opts, engineParams);
    tester.LaunchWorkers();

    do
    {
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    while(OnEnterPressedResult::Handled == onEnterPressed(!std::cin || std::cin.eof()));
    std::cout << "Finalizing...\n";
    tester.KillWorkers();

    return 0;
}
