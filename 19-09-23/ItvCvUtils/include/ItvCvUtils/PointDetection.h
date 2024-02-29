#ifndef COMPUTERVISION_POINTDETECTIONDATA_H
#define COMPUTERVISION_POINTDETECTIONDATA_H

#include "ItvCvUtils.h"

#include <ItvSdk/include/baseTypes.h>
#include <ItvCvUtils/Math.h>

#include <vector>

namespace PointDetection
{

    struct ITVCV_UTILS_API PointDetectionData
    {
        PointDetectionData();

        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> keypoints;
        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> descriptors;
        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> matches;
        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> scores;
    };
}

#endif // COMPUTERVISION_POINTDETECTIONDATA_H
