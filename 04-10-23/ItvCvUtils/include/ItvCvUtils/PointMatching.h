#ifndef COMPUTERVISION_POINTMATCHINGDATA_H
#define COMPUTERVISION_POINTMATCHINGDATA_H

#include "ItvCvUtils.h"

#include <ItvSdk/include/baseTypes.h>
#include <ItvCvUtils/Math.h>

#include <vector>

namespace PointMatching
{

    struct ITVCV_UTILS_API PointMatchingData
    {
        PointMatchingData();

        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> keypoints;
        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> descriptors;
        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> matches;
        std::array<std::pair<std::vector<float>, std::vector<int64_t>>, 2> scores;
    };
}

#endif // COMPUTERVISION_POINTMATCHINGDATA_H
