#include "StitchingEngine.h"
#include "SuperPoint.h"

#include <boost/test/unit_test.hpp>
#include <boost/optional.hpp>

BOOST_AUTO_TEST_SUITE(CreateEngine)

BOOST_AUTO_TEST_CASE(SimpleTest)
{
    img1 = imread("E:/devel/helper/homography/special/9/src9-1.jpg");
    img2 = imread("E:/devel/helper/homography/special/9/src9-2.jpg");
    std::vector<cv::Mat> matBuf = {img1, img2};
    const auto modelPath = "";

    // Проверка на невалидный путь к сети
    itvcvError error = itvcvErrorSuccess;
    const auto engine = SuperPoint(modelPath, matBuf, &error);
    BOOST_CHECK_MESSAGE(error == itvcvErrorLoadANN, error);
}

BOOST_AUTO_TEST_SUITE_END();