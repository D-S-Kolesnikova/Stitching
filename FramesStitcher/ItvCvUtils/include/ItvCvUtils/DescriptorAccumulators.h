#ifndef DESCRIPTORACCUMULATOR_H
#define DESCRIPTORACCUMULATOR_H

#include "ItvCvUtils.h"

#include <boost/optional.hpp>

#include <vector>
#include <memory>

namespace DescriptorAccumulation
{

struct ReIdOutput
{
    std::vector<float> features;
    float qualityScore;
};

struct IDescriptorAccumulator
{
public:
    using ReIdResult_t = ReIdOutput;

public:
    virtual void Accumulate(const ReIdResult_t& descriptor) = 0;
    virtual boost::optional<std::vector<float>> Reduce() = 0;

    virtual ~IDescriptorAccumulator() = default;
    };

enum class DescriptorAccumulationMethod
{
    Average = 0,
    AverageNormed = 1,
    QualityScore = 2,
    TopK = 3
};

ITVCV_UTILS_API std::shared_ptr<IDescriptorAccumulator> CreateDescriptorAccumulator(DescriptorAccumulationMethod accumulationMethod);

}

#endif
