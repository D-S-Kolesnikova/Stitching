#include "StitchingEngineImpl.h"
#include "SuperPoint.h"

#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::detail;

void writePanoramaToFile(const int nFrames, const vector<CameraParams>& cameras, const std::vector<Mat>& input_img);

class CStitchingWrapper : public IStitchingWrapper
{
private:
    const int m_direction;
    const int* m_widths;
    const int* m_heights;
    const int* m_strides;
    const int* m_colorSchemes;

    // Main stitching settings
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    double work_megapix = 0.6;
    double seam_megapix = 0.1;
    float match_conf = 0.3f;
    float conf_thresh = 0.3f;
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    double compose_work_aspect = 1;
    float warped_image_scale;
    float wrapperRatio;
    double seam_work_aspect = 1;
    double compose_megapix = -1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
    bool try_gpu = false;
    bool do_wave_correct = true;
    WaveCorrectKind wave_correct;
    string features_type = "SuperPoint";
    string matcher_type = "BestOf2NearestMatcher";
    string ba_cost_func = "ray";
    string ba_refine_mask = "xxxxx";
    int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
    string seam_find_type = "gc_color";
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    Ptr<WarperCreator> warper_creator;
    Ptr<RotationWarper> warper;
    bool isInit = false;
    stWarpType prevType = stWarpPlane;
public:
    CStitchingWrapper(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                      const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, const char* netPath, stError* outputError)
        : m_direction(direction)
        , m_widths(widths)
        , m_heights(heights)
        , m_strides(strides)
        , m_colorSchemes(colorSchemes)
    {
        *outputError = ESTNoError;
        // ���������� �������� ������������ ����������
        for (int i = 0; i < nFrames; ++i)
        {
            if (0 == frameDataBGR[i] || widths[i] <= 0 || heights[i] <= 0 ||
                strides[i] <= 0 || colorSchemes[i] <= 0 ||
                strides[i] < widths[i] || colorSchemes[i] > 16)
            {
                *outputError = ESTBadInputParams;
                return;
            }
        }

        std::vector<Mat> input_img(nFrames, Mat());
        int totalShift = 0;
        for (int i = 0; i < nFrames; ++i)
        {

            int type = colorSchemes[i] == 0 ? CV_8UC1 : CV_8UC3;
            int shift = i > 0 ? strides[i - 1] * heights[i - 1] : 0;
            totalShift += shift;
            Mat(heights[i], widths[i], type,
                (void*)(frameDataBGR + totalShift), strides[i]).copyTo(input_img[i]);
            //cv::imwrite(std::to_string(i) + "___1_.jpg", input_img[i]);
        }

        // Define direction
        //////////////////////////////////////////////////////////////////////////////////////////////////////////
        wave_correct = direction == 0 ? detail::WAVE_CORRECT_HORIZ : detail::WAVE_CORRECT_VERT;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////

        Ptr<Feature2D> finder;
        if (features_type == "orb")
        {
            finder = ORB::create();
        }
        else if (features_type == "surf")
        {
            finder = xfeatures2d::SURF::create();
        }
        else if (features_type == "sift")
        {
            finder = SIFT::create();
        }
        else if (features_type == "SuperPoint")
        {
            matcher_type = "LightGlue";
            finder = makePtr<SuperPoint>(netPath, input_img);
            is_work_scale_set = true;
            do_wave_correct = false;
            ba_cost_func = "no";
        }
        else
        {
            cout << "Unknown 2D features type: '" << features_type << "'.\n";
            *outputError = ESTCalibrationError;
            return;
        }

        Mat full_img, img;
        vector<ImageFeatures> features(nFrames);
        vector<Mat> images(nFrames);
        vector<Size> full_img_sizes(nFrames);

        for (int i = 0; i < nFrames; ++i)
        {
            full_img = input_img[i];
            full_img_sizes[i] = full_img.size();

            if (full_img.empty())
            {
                *outputError = ESTCalibrationError;
                return;
            }
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
            if (!is_seam_scale_set)
            {
                seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
                seam_work_aspect = seam_scale / work_scale;
                is_seam_scale_set = true;
            }

            computeImageFeatures(finder, img, features[i]);
            features[i].img_idx = i;

            resize(full_img, img, Size(), seam_scale, seam_scale);
            images[i] = img.clone();
        }

        full_img.release();
        img.release();

        cv::Ptr<FeaturesMatcher> matcher;
        if (matcher_type == "LightGlue")
        {
            Ptr<SuperPoint> superPointFinder = finder.dynamicCast<SuperPoint>();
            if (!superPointFinder.empty())
            {
                matcher = superPointFinder->getMatcher();
            }
            else
            {
                *outputError = ESTGetPointsError;
                return;
            }
        }
        else
        {
            matcher = makePtr<BestOf2NearestMatcher>(try_gpu, match_conf);
        }

        vector<MatchesInfo> pairwise_matches;

        (*matcher)(features, pairwise_matches);
        (*matcher).collectGarbage();
        // Leave only images we are sure are from the same panorama
        vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
        if (indices.size() != nFrames)
        {
            *outputError = ESTCalibrationError;
            return;
        }

        estimator(features, pairwise_matches, cameras);
        //LightGlue gives a result that is not normalized relative to the center of the frame
        if (matcher_type == "LightGlue")
        {
            for (int i = 0; i < nFrames; ++i)
            {
                cameras[i].ppx -= 0.5 * features[i].img_size.width;
                cameras[i].ppy -= 0.5 * features[i].img_size.height;
            }
        }

        for (size_t i = 0; i < cameras.size(); ++i)
        {
            Mat R;
            cameras[i].R.convertTo(R, CV_32F);
            cameras[i].R = R;
        }

        Ptr<detail::BundleAdjusterBase> adjuster;
        if (ba_cost_func == "reproj")
        {
            adjuster = new detail::BundleAdjusterReproj();
        }
        else if (ba_cost_func == "ray")
        {
            adjuster = new detail::BundleAdjusterRay();
        }
        else
        {
            adjuster = new detail::NoBundleAdjuster();
        }

        adjuster->setConfThresh(conf_thresh);
        Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
        if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
        if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
        if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
        if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
        if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
        adjuster->setRefinementMask(refine_mask);
        (*adjuster)(features, pairwise_matches, cameras);

        // Find median focal length
        vector<double> focals;
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            if (isnan(cameras[i].focal))
            {
                *outputError = ESTCamReconstructionError;
                return;
            }
            focals.push_back(cameras[i].focal);
        }

        sort(focals.begin(), focals.end());
        if (focals.size() % 2 == 1)
            warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
        else
            warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

        if (do_wave_correct)
        {
            vector<Mat> rmats;
            for (size_t i = 0; i < cameras.size(); ++i)
                rmats.push_back(cameras[i].R.clone());
            waveCorrect(rmats, wave_correct);
            for (size_t i = 0; i < cameras.size(); ++i)
                cameras[i].R = rmats[i];
        }

        for (int img_idx = 0; img_idx < nFrames; ++img_idx)
        {
            // Read image and resize it if necessary
            full_img = input_img[img_idx];
            if (!is_compose_scale_set)
            {
                if (compose_megapix > 0)
                    compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
                is_compose_scale_set = true;

                // Compute relative scales
                //compose_seam_aspect = compose_scale / seam_scale;
                compose_work_aspect = compose_scale / work_scale;

                // Update corners and sizes
                for (int i = 0; i < nFrames; ++i)
                {
                    // Update intrinsics
                    cameras[i].focal *= compose_work_aspect;
                    cameras[i].ppx *= compose_work_aspect;
                    cameras[i].ppy *= compose_work_aspect;
                }
            }
        }

        int globalIndex = 0;
        for (int img_idx = 0; img_idx < nFrames; ++img_idx)
        {
            cv::Mat K, R;
            cameras[img_idx].K().convertTo(K, CV_32F);
            cameras[img_idx].R.convertTo(R, CV_32F);

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    outputIntinsicCameraMatrix[globalIndex] = K.at<float>(i, j);
                    outputRotationMatrix[globalIndex] = R.at<float>(i, j);
                    ++globalIndex;
                }
        }

        wrapperRatio = warped_image_scale * compose_work_aspect;
        *ratio = wrapperRatio;
    }

    void getPoints(const stWarpType warpType, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                   const int nPoints, const float* inputPoints, float* outputPoints, stError* outputError, const int nFrame = -1)
    {
        // Warp images and their masks
        if (!isInit || prevType != warpType)
        {
            if (warpType == stWarpPlane) warper_creator = new cv::PlaneWarper();
            else if (warpType == stWarpCylindrical) warper_creator = new cv::CylindricalWarper();
            else if (warpType == stWarpSpherical) warper_creator = new cv::SphericalWarper();
            else if (warpType == stWarpFisheye) warper_creator = new cv::FisheyeWarper();
            else if (warpType == stWarpStereographic) warper_creator = new cv::StereographicWarper();
            else if (warpType == srWarpCompressedPlaneA2B1) warper_creator = new cv::CompressedRectilinearWarper(2, 1);
            else if (warpType == srWarpCompressedPlaneA15B1) warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
            else if (warpType == srWarpCompressedPlanePortraitA2B1) warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
            else if (warpType == srWarpCompressedPlanePortraitA15B1) warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
            else if (warpType == srWarpPaniniA2B1) warper_creator = new cv::PaniniWarper(2, 1);
            else if (warpType == srWarpPaniniA15B1) warper_creator = new cv::PaniniWarper(1.5, 1);
            else if (warpType == srWarpPaniniPortraitA2B1) warper_creator = new cv::PaniniPortraitWarper(2, 1);
            else if (warpType == srWarpPaniniPortraitA15B1) warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
            else if (warpType == srWarpMercator) warper_creator = new cv::MercatorWarper();
            else if (warpType == srWarpTransverseMercator) warper_creator = new cv::TransverseMercatorWarper();

            if (warper_creator.empty())
            {
                *outputError = ESTGetPointsError;
                return;
            }

            warper = warper_creator->create(wrapperRatio);
            prevType = warpType;
            isInit = true;
        }

        cv::Mat K, R;
        if (nFrame == -1)
        {
            K = cv::Mat(3, 3, CV_32F);
            R = cv::Mat(3, 3, CV_32F);
            int globalIndex = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    K.at<float>(i, j) = inputIntinsicCameraMatrix[globalIndex];
                    R.at<float>(i, j) = inputRotationMatrix[globalIndex];
                    ++globalIndex;
                }
        }
        else
        {
            cameras[nFrame].K().convertTo(K, CV_32F);
            cameras[nFrame].R.convertTo(R, CV_32F);
        }

        for (int i = 0; i < 2*nPoints; i+=2)
        {
            cv::Point2f src(inputPoints[i], inputPoints[i+1]);
            cv::Point2f tmp = warper->warpPoint(src, K, R);
            outputPoints[i] = tmp.x;
            outputPoints[i + 1] = tmp.y;
        }
    }

    ~CStitchingWrapper(){}
};


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void getPointsStatic(const stWarpType type, const float* inputIntinsicCameraMatrix, const float* inputRotationMatrix,
                     const int nPoints, const float* inputPoints, float* outputPoints, const float wrapperRatio, stError* outputError)
{

    Ptr<WarperCreator> warper_creator;
    if (type == stWarpPlane) warper_creator = new cv::PlaneWarper();
    else if (type == stWarpCylindrical) warper_creator = new cv::CylindricalWarper();
    else if (type == stWarpSpherical) warper_creator = new cv::SphericalWarper();
    else if (type == stWarpFisheye) warper_creator = new cv::FisheyeWarper();
    else if (type == stWarpStereographic) warper_creator = new cv::StereographicWarper();
    else if (type == srWarpCompressedPlaneA2B1) warper_creator = new cv::CompressedRectilinearWarper(2, 1);
    else if (type == srWarpCompressedPlaneA15B1) warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
    else if (type == srWarpCompressedPlanePortraitA2B1) warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
    else if (type == srWarpCompressedPlanePortraitA15B1) warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
    else if (type == srWarpPaniniA2B1) warper_creator = new cv::PaniniWarper(2, 1);
    else if (type == srWarpPaniniA15B1) warper_creator = new cv::PaniniWarper(1.5, 1);
    else if (type == srWarpPaniniPortraitA2B1) warper_creator = new cv::PaniniPortraitWarper(2, 1);
    else if (type == srWarpPaniniPortraitA15B1) warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
    else if (type == srWarpMercator) warper_creator = new cv::MercatorWarper();
    else if (type == srWarpTransverseMercator) warper_creator = new cv::TransverseMercatorWarper();

    if (warper_creator.empty())
    {
        *outputError = ESTGetPointsError;
        return;
    }

    Ptr<RotationWarper> warper = warper_creator->create(wrapperRatio);

    cv::Mat K, R;

    K = cv::Mat(3, 3, CV_32F);
    R = cv::Mat(3, 3, CV_32F);
    int globalIndex = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            K.at<float>(i, j) = inputIntinsicCameraMatrix[globalIndex];
            R.at<float>(i, j) = inputRotationMatrix[globalIndex];
            ++globalIndex;
        }


    for (int i = 0; i < 2 * nPoints; i += 2)
    {
        cv::Point2f src(inputPoints[i], inputPoints[i + 1]);
        cv::Point2f tmp = warper->warpPoint(src, K, R);
        outputPoints[i] = tmp.x;
        outputPoints[i + 1] = tmp.y;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IStitchingWrapper* CreateStitchingEngine(const int nFrames, const int direction, const int* widths, const int* heights, const int* strides, const int* colorSchemes,
                                         const unsigned char* frameDataBGR, float* outputIntinsicCameraMatrix, float* outputRotationMatrix, float* ratio, const char* netPath, stError* outputError)
{
    return new CStitchingWrapper(nFrames, direction, widths, heights, strides, colorSchemes, frameDataBGR, outputIntinsicCameraMatrix, outputRotationMatrix, ratio, netPath, outputError);
}

void writePanoramaToFile(const int nFrames, const vector<CameraParams>& cameras, const std::vector<Mat>& input_img)
{
    vector<Mat> H(nFrames);
    cv::Mat K, R;
    for (int i = 0; i < nFrames; ++i)
    {
        cameras[i].K().convertTo(K, CV_32F);
        cameras[i].R.convertTo(R, CV_32F);
        H[i] = cv::Mat(3, 3, CV_32F, 0.0);
        H[i] = K * R * K.inv();
    }

    std::vector<cv::Mat> img_stitch(nFrames);
    for (int i = 0; i < nFrames; ++i)
    {
        warpPerspective(input_img[i], img_stitch[i], H[i], cv::Size(input_img[i].cols * 2, input_img[i].rows * 2));
        cv::imwrite("E:/devel/helper/homography/stitch_" + std::to_string(i) + ".jpg", img_stitch[i]);
    }
    cv::Mat res_stitch = img_stitch[1];
    for (int i = 0; i < input_img[0].rows; ++i)
        for (int j = 0; j < input_img[0].cols; ++j)
        {
            cv::Vec3b pxVal = input_img[0].at<cv::Vec3b>(i, j);
            if (pxVal[0] != 0 && pxVal[1] != 0 && pxVal[2] != 0)
            {
                res_stitch.at<cv::Vec3b>(i, j) = input_img[0].at<cv::Vec3b>(i, j);
            }
        }
    cv::imwrite("E:/devel/helper/homography/stitch_res.jpg", res_stitch);
}
