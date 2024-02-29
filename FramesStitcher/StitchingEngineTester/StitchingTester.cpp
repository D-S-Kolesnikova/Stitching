// StitchingTester.cpp : Defines the entry point for the console application.
//

//#define DEBUG 1

#ifndef DEBUG

#include "StitchingEngine.h"
#include "SuperPoint.h"
#include "LightGlue.h"

#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

int main()
{

    Mat img1, img2, img3, img4;
    img1 = imread("E:/devel/helper/homography/special/7/src7-1.jpg");
    img2 = imread("E:/devel/helper/homography/special/7/src7-2.jpg");

    vector<Mat> imgs = {  img1, img2 };
    Mat pano;
    const int numImages = 2; //3
    //img1 = imread("11.png");
    //img2 = imread("12.png");
    //img3 = imread("13.png");
    //img4 = imread("14.png");

    /*img1 = imread("left22.jpg");
    img2 = imread("center22.jpg");
    img3 = imread("right22.jpg");*/

    //int widths[] = { img1.cols, img2.cols, img3.cols, img4.cols };
    //int heights[] = { img1.rows, img2.rows, img3.rows, img4.rows };
    //int strides[] = { img1.step, img2.step, img3.step, img4.step };
    //int schemes[] = { img1.type(), img2.type(), img3.type(), img4.type() };
    ////unsigned char* datas[] = { (unsigned char*)img1.data, (unsigned char*)img2.data };
    //unsigned char* datas = new unsigned char[img1.rows*img1.step + img2.rows*img2.step + img3.rows*img3.step + img4.rows*img4.step];
    //memcpy(datas, img1.data, img1.rows*img1.step);
    //memcpy(datas + img1.rows*img1.step, img2.data, img2.rows*img2.step);
    //memcpy(datas + img1.rows*img1.step + img2.rows*img2.step, img3.data, img3.rows*img3.step);
    //memcpy(datas + img1.rows*img1.step + img2.rows*img2.step + img3.rows*img3.step, img4.data, img4.rows*img4.step);

    //int widths[] = { img1.cols, img2.cols, img3.cols };
    //int heights[] = { img1.rows, img2.rows, img3.rows };
    //int strides[] = { img1.step1(), img2.step1(), img3.step1() };
    //int schemes[] = { img1.type(), img2.type(), img3.type() };
    ////unsigned char* datas[] = { (unsigned char*)img1.data, (unsigned char*)img2.data };
    //unsigned char* datas = new unsigned char[img1.rows*img1.step + img2.rows*img2.step + img3.rows*img3.step];
    //memcpy(datas, img1.data, img1.rows*img1.step);
    //memcpy(datas + img1.rows*img1.step, img2.data, img2.rows*img2.step);
    //memcpy(datas + img1.rows*img1.step + img2.rows*img2.step, img3.data, img3.rows*img3.step);

    int widths[] = { img1.cols, img2.cols };
    int heights[] = { img1.rows, img2.rows };
    int strides[] = { img1.step, img2.step };
    int schemes[] = { img1.type(), img2.type() };
    unsigned char* datas = new unsigned char[img1.rows*img1.step + img2.rows*img2.step];
    memcpy(datas, img1.data, img1.rows*img1.step);
    memcpy(datas + img1.rows*img1.step, img2.data, img2.rows*img2.step);


    /*std::ifstream ifs("framedata.dat", std::ios::binary | std::ios::ate);
    std::ifstream::pos_type pos = ifs.tellg();
    int length = pos;
    int widths[] = { 640, 1920 };
    int heights[] = { 360, 1080};
    int strides[] = { 1920, 5760 };
    int schemes[] = { 16,16 };
    char* datas = new char[length];
    ifs.seekg(0, std::ios::beg);
    ifs.read(datas, length);*/

    stError err;
    float camMat[9 * numImages], rotMat[9 * numImages];
    float ratio;
    void* engine = createStitching(numImages, 0, widths, heights, strides, schemes, (unsigned char*)datas, camMat, rotMat, &ratio, &err);
    (err == ESTNoError) ? std::cout << "good" : std::cout << "bad";
    std::cout << std::endl;
    std::vector<cv::Mat> src_imgs(numImages);
    src_imgs[0] = img1;
    src_imgs[1] = img2;
    //src_imgs[2] = img3;
    //src_imgs[3] = img4;
    for (int i = 0; i < numImages; ++i)
    {
        float srcPoints[4] = { 0, 0, src_imgs[i].cols, src_imgs[i].rows };
        float dstPoints[4];
        getStitchingPoints(engine, stWarpSpherical, &camMat[9 * i], &rotMat[9 * i], 2, srcPoints, dstPoints, &err, -1, -1);
        std::cout << srcPoints[0] << "," << srcPoints[1] << " to" << dstPoints[0] << "," << dstPoints[1] << std::endl;
        std::cout << srcPoints[2] << "," << srcPoints[3] << " to" << dstPoints[2] << "," << dstPoints[3] << std::endl;
    }    

    //cv::vector<cv::Mat> img_stitch(numImages);
    //cv::vector<cv::Mat> src_imgs(numImages);
    //cv::vector<cv::Mat> Homos(numImages);
    //src_imgs[0] = img1;
    //src_imgs[1] = img2;
    ////src_imgs[2] = img3;
    ////src_imgs[3] = img4;
    //for (int i = 0; i < numImages; ++i)
    //{
    //    Homos[i] = cv::Mat(3, 3, CV_32F, 0.0);
    //    for (int p = 0; p < 3; ++p)
    //    {
    //        for (int q = 0; q < 3; ++q)
    //        {
    //            Homos[i].at<float>(p, q) = homoMatrices[9 * i + 3 * p + q];
    //        }
    //    }
    //
    //    //Homos[i] = Homos[i].inv();
    //    warpPerspective(src_imgs[i], img_stitch[i], Homos[i], cv::Size(src_imgs[i].cols * 5, src_imgs[i].rows * 5));
    //    cv::imwrite("stitch"+std::to_string(i)+".jpg", img_stitch[i]);
    //}


    //cv::vector<cv::Mat> img_stitch2(numImages);
    //for (int p = 0; p < numImages; ++p)
    //{
    //    img_stitch2[p] = cv::Mat(cv::Size(src_imgs[p].cols * 5, src_imgs[p].rows * 5), src_imgs[p].type(), cv::Scalar(0, 0, 0));

    //    for (int i = 0; i < img_stitch2[p].rows; ++i)
    //        for (int j = 0; j < img_stitch2[p].cols; ++j)
    //        {
    //            float oldx = (homoMatrices[9 * p] * j + homoMatrices[9 * p + 1] * i + homoMatrices[9 * p + 2]) /
    //                         (homoMatrices[9 * p + 3 * 2] * j + homoMatrices[9 * p + 3 * 2 + 1] * i + homoMatrices[9 * p + 3 * 2 + 2]);
    //            float oldy = (homoMatrices[9 * p + 3 * 1] * j + homoMatrices[9 * p + +3 * 1 + 1] * i + homoMatrices[9 * p + +3 * 1 + 2]) /
    //                         (homoMatrices[9 * p + 3 * 2] * j + homoMatrices[9 * p + 3 * 2 + 1] * i + homoMatrices[9 * p + 3 * 2 + 2]);

    //            if (oldx > 0 && oldy > 0 && oldx < src_imgs[p].cols && oldy < src_imgs[p].rows)
    //                img_stitch2[p].at<cv::Vec3b>(i, j) = src_imgs[p].at<cv::Vec3b>(int(oldy), int(oldx));
    //        }
    //    cv::imwrite("stitch" + std::to_string(p) + "_manual.jpg", img_stitch2[p]);
    //}


    //cv::Mat res_stitch = img_stitch[0];
    //for (int i = 0; i < res_stitch.rows; ++i)
    //    for (int j = 0; j < res_stitch.cols; ++j)
    //    {
    //        cv::Vec3b pxVal = res_stitch.at<cv::Vec3b>(i, j);
    //        if (pxVal[0] == 0 && pxVal[1] == 0 && pxVal[2] == 0)
    //        {
    //            int a = 1;
    //            res_stitch.at<cv::Vec3b>(i, j) = img_stitch[1].at<cv::Vec3b>(i, j);
    //        }
    //    }
    //cv::imwrite("stitch_res.jpg", res_stitch);

    //cv::Mat res_stitch2 = img_stitch2[0];
    //for (int i = 0; i < res_stitch2.rows; ++i)
    //    for (int j = 0; j < res_stitch2.cols; ++j)
    //    {
    //        cv::Vec3b pxVal = res_stitch2.at<cv::Vec3b>(i, j);
    //        if (pxVal[0] == 0 && pxVal[1] == 0 && pxVal[2] == 0)
    //        {
    //            int a = 1;
    //            res_stitch2.at<cv::Vec3b>(i, j) = img_stitch2[1].at<cv::Vec3b>(i, j);
    //        }
    //    }
    //cv::imwrite("stitch_res2.jpg", res_stitch2);

    return 0;
}




///*M///////////////////////////////////////////////////////////////////////////////////////
////
////  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
////
////  By downloading, copying, installing or using the software you agree to this license.
////  If you do not agree to this license, do not download, install,
////  copy or use the software.
////
////
////                          License Agreement
////                For Open Source Computer Vision Library
////
//// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
//// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
//// Third party copyrights are property of their respective owners.
////
//// Redistribution and use in source and binary forms, with or without modification,
//// are permitted provided that the following conditions are met:
////
////   * Redistribution's of source code must retain the above copyright notice,
////     this list of conditions and the following disclaimer.
////
////   * Redistribution's in binary form must reproduce the above copyright notice,
////     this list of conditions and the following disclaimer in the documentation
////     and/or other materials provided with the distribution.
////
////   * The name of the copyright holders may not be used to endorse or promote products
////     derived from this software without specific prior written permission.
////
//// This software is provided by the copyright holders and contributors "as is" and
//// any express or implied warranties, including, but not limited to, the implied
//// warranties of merchantability and fitness for a particular purpose are disclaimed.
//// In no event shall the Intel Corporation or contributors be liable for any direct,
//// indirect, incidental, special, exemplary, or consequential damages
//// (including, but not limited to, procurement of substitute goods or services;
//// loss of use, data, or profits; or business interruption) however caused
//// and on any theory of liability, whether in contract, strict liability,
//// or tort (including negligence or otherwise) arising in any way out of
//// the use of this software, even if advised of the possibility of such damage.
////
////
////M*/

#else

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

static void printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching_detailed img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb)\n"
        "      Type of features used for images matching. The default is surf.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (reproj|ray)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}


// Default command line args
vector<string> img_names;
bool preview = false;
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.3f;
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_gpu = true;
            else
            {
                cout << "Bad --try_gpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (features_type == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}


int main(int argc, char* argv[])
{
#if ENABLE_LOG
    int64 app_start_time = getTickCount();
#endif

    cv::setBreakOnError(true);

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<Feature2D> finder;
    if (features_type == "surf")
    {
        finder = xfeatures2d::SURF::create();
    }
    else if (features_type == "orb")
    {
        finder = ORB::create();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Check if we should save matches graph
    if (save_graph)
    {
        LOGLN("Saving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<string> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial intrinsics #" << indices[i] + 1 << ":\n" << cameras[i].K());
    }

    //for (size_t i = 0; i < cameras.size(); ++i)
    //{
    //    std::cout << "##################### Step 0 ##########################" <<std::endl;
    //    std::cout << "K = " << std::endl << cameras[i].K() << std::endl;
    //    std::cout << "R = " << std::endl << cameras[i].R << std::endl;
    //}

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
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
            int p = 1;
            int q = 2;
        }
        LOGLN("Camera #" << indices[i] + 1 << ":\n" << cameras[i].K());
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
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

    //for (size_t i = 0; i < cameras.size(); ++i)
    //{
    //    std::cout << "##################### Step 1 ##########################" << std::endl;
    //    std::cout << "K = " << std::endl << cameras[i].K() << std::endl;
    //    std::cout << "R = " << std::endl << cameras[i].R << std::endl;
    //}

    //std::cout << "cameraParams-K: " << cameras[0].K();
    //std::cout << "focal = " << cameras[0].focal << "; cameras[0].ppx = " << cameras[0].ppx << "; cameras[0].ppy = " << cameras[0].ppy << std::endl;
    //
    //vector<Mat> img_stitch(2);
    //for (int nFrame = 0; nFrame < num_images; ++nFrame)
    //{
    //    const cv::Mat_<float> K = cameras[nFrame].K();
    //    const cv::Mat_<float> R = cameras[nFrame].R;
    //    cv::Mat homomorphy = K * R * K.inv();
    //    homomorphy /= homomorphy.at<float>(2, 2);
    //    //homomorphy = homomorphy.inv();

    //    warpPerspective(images[nFrame], img_stitch[nFrame], homomorphy, cv::Size(images[nFrame].cols * 2, images[nFrame].rows));
    //}

        //cv::Mat res_stitch = img_stitch[0];
        //for (int i = 0; i < res_stitch.rows; ++i)
        //    for (int j = 0; j < res_stitch.cols; ++j)
        //    {
        //        cv::Vec3b pxVal = res_stitch.at<cv::Vec3b>(i, j);
        //        if (pxVal[0] == 0 && pxVal[1] == 0 && pxVal[2] == 0)
        //        {
        //            int a = 1;
        //            res_stitch.at<cv::Vec3b>(i, j) = img_stitch[1].at<cv::Vec3b>(i, j);
        //        }
        //    }
        //cv::imwrite("stitch_res.jpg", res_stitch);


    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_gpu && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarperGpu();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarperGpu();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarperGpu();
    }
    else
#endif
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
        else if (warp_type == "fisheye") warper_creator = new cv::FisheyeWarper();
        else if (warp_type == "stereographic") warper_creator = new cv::StereographicWarper();
        else if (warp_type == "compressedPlaneA2B1") warper_creator = new cv::CompressedRectilinearWarper(2, 1);
        else if (warp_type == "compressedPlaneA1.5B1") warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
        else if (warp_type == "compressedPlanePortraitA2B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
        else if (warp_type == "compressedPlanePortraitA1.5B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
        else if (warp_type == "paniniA2B1") warper_creator = new cv::PaniniWarper(2, 1);
        else if (warp_type == "paniniA1.5B1") warper_creator = new cv::PaniniWarper(1.5, 1);
        else if (warp_type == "paniniPortraitA2B1") warper_creator = new cv::PaniniPortraitWarper(2, 1);
        else if (warp_type == "paniniPortraitA1.5B1") warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
        else if (warp_type == "mercator") warper_creator = new cv::MercatorWarper();
        else if (warp_type == "transverseMercator") warper_creator = new cv::TransverseMercatorWarper();
    }

    if (warper_creator.empty())
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }

    std::cout << "warped_image_scale = " << warped_image_scale << "; seam_work_aspect << " << seam_work_aspect << std::endl;
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa; K(0, 2) *= swa;
        K(1, 1) *= swa; K(1, 2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    //for (size_t i = 0; i < cameras.size(); ++i)
    //{
    //    std::cout << "##################### Step 2 ##########################" << std::endl;
    //    std::cout << "K = " << std::endl << cameras[i].K() << std::endl;
    //    std::cout << "R = " << std::endl << cameras[i].R << std::endl;
    //}

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "voronoi")
        seam_finder = new detail::VoronoiSeamFinder();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    if (seam_finder.empty())
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx] + 1);

        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            std::cout << "warped_image_scale = " << warped_image_scale << std::endl;
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        //std::cout << "##################### Step 3 ##########################" << std::endl;
        //std::cout << "K = " << std::endl << cameras[img_idx].K() << std::endl;
        //std::cout << "R = " << std::endl << cameras[img_idx].R << std::endl;


    // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        //std::cout << "##################### Step 3 ##########################" << std::endl;
        cv::Mat  R;
        //cameras[img_idx].K().convertTo(K, CV_32F);
        cameras[img_idx].R.convertTo(R, CV_32F);
        std::cout << "K = " << std::endl << cameras[img_idx].K() << std::endl;
        std::cout << "R = " << std::endl << cameras[img_idx].R << std::endl;
        std::cout << 0 << "," << 0 << " to" << warper->warpPoint(cv::Point2f(0, 0), K, R) << std::endl;
        std::cout << img.cols << "," << img.rows << " to" << warper->warpPoint(cv::Point2f(img.cols, img.rows), K, R) << std::endl;
        std::cout << "Writes with R-K mats: " << "step3_img_" + std::to_string(img_idx) + ".jpg" << std::endl;
        std::cout << "K = " << std::endl << cameras[img_idx].K() << std::endl;
        std::cout << "R = " << std::endl << cameras[img_idx].R << std::endl;
        //cv::imwrite("step3_img_" + std::to_string(img_idx) + ".jpg", img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        //cv::imwrite("step3_mask_" + std::to_string(img_idx) + ".jpg", mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
        //cv::imwrite("step3_img_compensate_" + std::to_string(img_idx) + ".jpg", img_warped);
        //cv::imwrite("step3_mask_compensate_" + std::to_string(img_idx) + ".jpg", mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_gpu);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_gpu);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
                fb->setSharpness(1.f / blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);

    //for (size_t i = 0; i < cameras.size(); ++i)
    //{
    //    std::cout << "##################### Step 5 ##########################" << std::endl;
    //    std::cout << "K = " << std::endl << cameras[i].K() << std::endl;
    //    std::cout << "R = " << std::endl << cameras[i].R << std::endl;
    //}

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    imwrite(result_name, result);

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return 0;
}

#endif
