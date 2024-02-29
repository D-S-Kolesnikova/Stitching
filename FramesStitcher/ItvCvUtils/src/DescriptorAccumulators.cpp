#include <ItvCvUtils//DescriptorAccumulators.h>

#include <boost/circular_buffer.hpp>

#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <iostream>
#include <numeric>

namespace DescriptorAccumulation
{

constexpr auto QUALITY_THRESHOLD = 0.6;
constexpr auto COS_SCORE_THRESHOLD = 0.8;
constexpr auto QUALITY_SCORE_MEASUREMENT_COUNT = 24;
constexpr auto TOPK_TTL = 20;
constexpr auto TOPK_COUNT_K = 20;

float compareVectors(
    const std::vector<float>& vec1,
    const std::vector<float>& vec2
)
{
    double mul = 0;
    double denomA = 0;
    double denomB = 0;

    for (auto idx = 0; idx < vec1.size(); ++idx)
    {
        mul += vec1[idx] * vec2[idx];
        denomA += vec1[idx] * vec1[idx];
        denomB += vec2[idx] * vec2[idx];
    }

    return mul / sqrt(denomA * denomB);
}

float GeometryVectorLength(const std::vector<float>& vec)
{
    auto sumSqr = 0.0f;
    for (const auto &elem : vec)
    {
        sumSqr += elem * elem;
    }
    return std::sqrt(sumSqr);
}

void GeometryNormalizeVector(std::vector<float>& vec)
{
    const auto length = GeometryVectorLength(vec);
    std::transform(
        vec.begin(),
        vec.end(),
        vec.begin(),
        [&length](const auto& elem){return elem / length;});
}

class DescriptorAccumulatorByAverage : public IDescriptorAccumulator
{
public:
    void Accumulate(const ReIdResult_t& descriptor) override
    {
        const auto& feature = descriptor.features;

        if (m_accumulatedDescriptor.empty())
        {
            m_accumulatedDescriptor.resize(feature.size(), 0);
        }

        if (m_accumulatedDescriptor.size() != feature.size())
        {
            throw std::logic_error("DescriptorAccumulatorByAverage: Mismatching descriptor size");
        }

        for (size_t i = 0; i < feature.size(); ++i)
        {
            m_accumulatedDescriptor[i] += feature[i];
        }
        ++m_count;
    }

    boost::optional<std::vector<float>> Reduce() override
    {
        std::vector<float> reduced(m_accumulatedDescriptor.size());
        for (size_t i = 0; i < m_accumulatedDescriptor.size(); ++i)
        {
            reduced[i] = m_accumulatedDescriptor[i] / m_count;
        }

        return { reduced };
    }

private:
    std::vector<float> m_accumulatedDescriptor;
    size_t m_count = 0;
};

class DescriptorAccumulatorByAverageNormed : public IDescriptorAccumulator
{
private:
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Функция удаления из трека всех векторов, которые плохо сравнимы внутри данного трека ( с порогом, меньше переданного)
    // Input:
    // vec - массив из "векторов" похожести, полученных для трека
    // thresholdSimil [0-1] - минимальный порог сходства, при которых вектора считаются принадлежащими одной и той же персоне
    // bestPercent [0-1]- минимальный процент хорошо сравнимых векторов от общего числа векторов на трек, при котором вектор
    //                    считается хорошим и не исключается из формирования общего вектора
    // Output:
    // vec - модифицированный массив векторов, после удаления "шумовых" из них
    void CleanVector(std::vector <std::vector<float> >& vec, const float thresholdSimil = 0.5, const float bestPercent = 0.2)
    {
        std::vector <std::vector<float> > scores; // похожесть всех со всеми

        // Строим матрицу похожестей всех со всеми
        for (auto i = size_t(0); i < vec.size(); ++i)
        {
            std::vector<float> compares;
            compares.reserve(vec.size());
            for (auto j = size_t(0); j < vec.size(); ++j)
            {
                const auto distance = compareVectors(vec[i], vec[j]);
                compares.push_back(distance);
            }

            scores.emplace_back(compares);
        }

        const int thresholdGoodWithSimilarity = std::max<int>(1, vec.size() * bestPercent); // минимальное количество "хороших" соответсвий для, чтого не считать вектор шумовым
                                                                                                      // 1 всегда будет - сам с собой.
        std::vector<bool> goodVec(scores.size(), true); // количество сходств выше, чем пороговое

        auto nCurrentVec = size_t(0);

        while (nCurrentVec < scores.size())
        {
            // Если вектор не был еще ранее отбракован, то только тогда анализируем
            if (goodVec[nCurrentVec])
            {

                // подсчитаем число хороших соответсвий, анализируемый вектор не пропускаем - в алгоритме учтено, что минимальное число равно 1
                int countGood = std::count_if(scores[nCurrentVec].begin(), scores[nCurrentVec].end(), [thresholdSimil](float elem) {return (elem > thresholdSimil); });

                // если у вектора нет достаточного числа хороших совпадений и он не занулен ранее, то удалим его упоминание в таблице сходств
                // countGood != 0 не трогаем уже ранее "зануленные" вектора
                if (countGood <= thresholdGoodWithSimilarity && countGood != 0)
                {
                    goodVec[nCurrentVec] = false;
                    // занулим строку и столбец в корреляционной матрицы сравнений всех со всеми
                    std::fill(scores[nCurrentVec].begin(), scores[nCurrentVec].end(), 0.f);
                    for (auto j = size_t(0); j < scores.size(); ++j)
                    {
                        scores[j][nCurrentVec] = 0.f;
                    }

                    // и начинаем все с начала
                    nCurrentVec = 0;
                    continue;
                }
            }
            ++nCurrentVec;
        }

        // удаляем плохие вектора в обратном порядке. При этом как минимум 1 вектор должны оставить, чтобы трек не остался без reid вектора
        for (int i = static_cast<int>(goodVec.size()) - 1; i >= 0; --i)
        {
            if (!goodVec[i] && vec.size() > 1)
            {
                vec.erase(vec.begin() + i);
            }
        }

    }

    // вычисление длины вектора
    // Input:
    // vec - вектора, для которых необходимо посчитать длину
    // Output:
    // lengths - вектор, соответсвующих длин
    std::vector <float> GetVecLength(const std::vector <std::vector<float> >& vec)
    {
        std::vector<float> lengths(vec.size(), .0f);
        for (auto i = size_t(0); i < vec.size(); ++i)
        {
            for (const auto &elem : vec[i])
            {
                lengths[i] += elem * elem;
            }
            lengths[i] = std::sqrt(lengths[i]);
        }

        return lengths;
    }

    // Функция нормированного среднего среди векторов
    // Input:
    // vec - вектора, среди которых надо найти усредненный
    // Output:
    // accum - итоговый "средений вектор"
    std::vector<float> AvgVectorsNormed(const std::vector <std::vector<float> > &vec)
    {

        std::vector<float> accum(vec.front().size(), 0);

        // отнормируем все вектора на их длину и произведем сложение
        std::vector<float> lengths = GetVecLength(vec);

        const auto sizeVec = vec.size();
        for (auto i = size_t(0); i < sizeVec; ++i)
        {
            for (auto j = size_t(0); j < vec[i].size(); ++j)
            {
                accum[j] += vec[i][j] / lengths[i];
            }
        }

        // отнормируем получившийся вектора так же на его длину
        std::vector <std::vector<float> > accumVec(1, accum);
        std::vector<float> accumLengths = GetVecLength(accumVec);

        for (auto j = size_t(0); j < accum.size(); ++j)
        {
            accum[j] /= accumLengths[0];
        }

        return accum;
    }

public:
    DescriptorAccumulatorByAverageNormed(float qualityThreshold) : m_qualityThreshold(qualityThreshold) {}

    void Accumulate(const ReIdResult_t& descriptor) override
    {
        if (!std::isnan(descriptor.qualityScore) && descriptor.qualityScore < m_qualityThreshold)
        {
            return;
        }

        if (!m_accumulatedDescriptors.empty() && m_accumulatedDescriptors.back().size() != descriptor.features.size())
        {
            throw std::logic_error("DescriptorAccumulatorByAverageNormed: Mismatching descriptor size");
        }

        m_accumulatedDescriptors.emplace_back(descriptor.features);
    }

    boost::optional<std::vector<float>> Reduce() override
    {
        if (m_accumulatedDescriptors.empty())
        {
            return {};
        }

        // удалим шумовые вектора
        CleanVector(m_accumulatedDescriptors);
        m_accumulatedDescriptors.shrink_to_fit();

        // посчитаем теперь нормированную сумму
        std::vector<float> reduced = AvgVectorsNormed(m_accumulatedDescriptors);

        return { reduced };
    }

private:
    std::vector<std::vector<float> > m_accumulatedDescriptors;
    const float m_qualityThreshold;
};

class DescriptorAccumulatorByQualityScore : public IDescriptorAccumulator
{
public:
    DescriptorAccumulatorByQualityScore(
        const int measurementCount,
        float cosScore,
        float qualityScoreThreshold)
        : m_qualityThreshold(qualityScoreThreshold)
        , m_cosScoreThreshold(cosScore)
        , m_actualFeatures(measurementCount) {}

    void Accumulate(const ReIdResult_t& descriptor) override
    {
        m_actualFeatures.push_back(descriptor);
        GeometryNormalizeVector(m_actualFeatures.back().features);
    }

    boost::optional<std::vector<float>> Reduce() override
    {
        CleanWorst();
        return GetBestFeatureFromBuffer();
    }

private:
    void CleanWorst()
    {
        boost::circular_buffer<ReIdResult_t> actual(m_actualFeatures.capacity());

        const int countFeatures = static_cast<int>(m_actualFeatures.size());

        if (countFeatures <= 1)
        {
            return;
        }

        std::vector<float> cosScores;
        const int diagMatrixSize = (std::pow(countFeatures, 2) - countFeatures) / 2;
        cosScores.reserve(diagMatrixSize);

        for (int i = 0; i < (countFeatures - 1); ++i)
        {
            const auto& firstFeature = m_actualFeatures[i].features;

            for (int j = i + 1; j < countFeatures; ++j)
            {
                const auto& secondFeature = m_actualFeatures[j].features;
                const auto distance = compareVectors(firstFeature, secondFeature);
                cosScores.emplace_back(distance);
            }
        }

        int offset = 0;
        float maxCosScore = 0;

        ReIdResult_t maxVector;

        for (int i = 0; i < countFeatures; ++i)
        {
            const auto& startIter = cosScores.begin() + offset;
            const auto& endIter = startIter + (countFeatures - 1) - i;

            float score = i < (countFeatures - 1) ? std::accumulate(startIter, endIter, 0) : 0;

            int offsetJ = 0;
            for (int j = 0; j < i; ++j)
            {
                score += *(cosScores.begin() + (i - 1 - j) + offsetJ);
                offsetJ += countFeatures - (j + 1);
            }

            offset += countFeatures - (i + 1);

            if (maxCosScore < score)
            {
                maxCosScore = score;
                maxVector = m_actualFeatures[i];
            }

            if (score >= m_cosScoreThreshold)
            {
                actual.push_back(m_actualFeatures[i]);
            }
        }

        if (actual.empty())
        {
            m_actualFeatures.clear();
            m_actualFeatures.push_back(maxVector);
        }
        else
        {
            m_actualFeatures.swap(actual);
        }
    }

    //added clean function

    boost::optional<std::vector<float>> GetBestFeatureFromBuffer() const
    {
        std::vector<ReIdResult_t> data(m_actualFeatures.begin(), m_actualFeatures.end());
        std::sort(
            data.begin(),
            data.end(),
            [](const ReIdResult_t& first, const ReIdResult_t& second)
        {
            const auto firstIsValid = !std::isnan(first.qualityScore);
            const auto secondIsValid = !std::isnan(second.qualityScore);
            if (firstIsValid && secondIsValid)
            {
                return first.qualityScore > second.qualityScore;
            }

            return firstIsValid;
        });

        if (!std::isnan(data.front().qualityScore) && data.front().qualityScore >= m_qualityThreshold)
        {
            auto& feature = data.front().features;

            return  { feature };
        }

        return {};
    }

private:
    const float m_qualityThreshold;
    const float m_cosScoreThreshold;
    boost::circular_buffer<ReIdResult_t> m_actualFeatures;
};

class DescriptorAccumulatorByTopK : public IDescriptorAccumulator
{
private:
    struct KFeature
    {
        int id;
        std::vector<float> feature;
        float qualityScore;
        int ttl;
    };

public:
    DescriptorAccumulatorByTopK(const int countK, const int ttl, const float cosThreshold, float qualityScoreThreshold)
    : m_qualityThreshold(qualityScoreThreshold)
    , m_cosThreshold(cosThreshold)
    , m_distanceScore(10)
    , m_idBest(-1)
    , m_countK(countK)
    , m_maxTtl(ttl)
    {
        m_actualFeatures.reserve(m_countK);
    }

    void Accumulate(const ReIdResult_t& descriptor) override
    {
        m_idBest = -1;
        const auto qualityScore = std::isnan(descriptor.qualityScore) ? -1 : descriptor.qualityScore;
        auto feature = descriptor.features;
        GeometryNormalizeVector(feature);

        if (m_actualFeatures.empty())
        {
            m_actualFeatures.emplace_back(
                KFeature{
                    0,
                    feature,
                    qualityScore,
                    m_maxTtl });
            m_idBest = 0;
            return;
        }

        float maxCosScore = 0;
        int bestId = -1;

        for (int i = 0; i < static_cast<int>(m_actualFeatures.size()); ++i)
        {
            const auto& kFeature = m_actualFeatures[i];
            const auto cosScore = compareVectors(kFeature.feature, feature);

            if (cosScore > maxCosScore)
            {
                maxCosScore = cosScore;
                bestId = i;
            }
        }

        if (maxCosScore >= m_cosThreshold && bestId >= 0)
        {
            m_idBest = bestId;
            auto& bestFeature = m_actualFeatures[bestId];
            const auto diffLess = (1 - maxCosScore) < m_distanceScore;

            if (diffLess && (bestFeature.qualityScore < qualityScore || bestFeature.qualityScore < 0))
            {
                ++bestFeature.ttl;
                bestFeature.qualityScore = qualityScore;
                bestFeature.feature = feature;
                return;
            }

            if (!diffLess && static_cast<int>(m_actualFeatures.size()) < m_countK)
            {
                m_actualFeatures.emplace_back(
                    KFeature{
                            static_cast<int>(m_actualFeatures.size()),
                            feature,
                            qualityScore,
                            m_maxTtl });
                    m_idBest = static_cast<int>(m_actualFeatures.size()) - 1;
                }
        }
    }

    boost::optional<std::vector<float>> Reduce() override
    {
        FindBestId();
        std::vector<float> result;

        if (m_actualFeatures[m_idBest].qualityScore < 0)
        {
            result = m_actualFeatures[m_idBest].feature;
        }
        else if (m_actualFeatures[m_idBest].qualityScore >= m_qualityThreshold)
        {
            result = m_actualFeatures[m_idBest].feature;
        }

        CleanDead();
        DieFeature();

        if (result.empty())
        {
            return { };
        }
        return { result };
    };

private:
    void DieFeature()
    {
        for (auto& kFeature : m_actualFeatures)
        {
            --kFeature.ttl;
        }
    }

    void CleanDead()
    {
        for (auto iter = m_actualFeatures.begin(); iter != m_actualFeatures.end();)
        {
            if (iter->ttl < 0)
            {
                iter = m_actualFeatures.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    void FindBestId()
    {
        if (m_idBest < 0)
        {
            const auto& bestIter = std::max_element(
                m_actualFeatures.begin(),
                m_actualFeatures.end(),
                [](const KFeature& left, const KFeature& right)
            {
                return left.qualityScore < right.qualityScore;
            });
            m_idBest = std::distance(m_actualFeatures.begin(), bestIter);
        }
    };

private:
    const float m_qualityThreshold;
    const float m_cosThreshold;
    float m_distanceScore;
    int m_idBest;
    const int m_countK;
    const int m_maxTtl;
    std::vector<KFeature> m_actualFeatures;
};

ITVCV_UTILS_API std::shared_ptr<IDescriptorAccumulator> CreateDescriptorAccumulator(DescriptorAccumulationMethod accumulationMethod)
{
    switch (accumulationMethod)
    {
        case DescriptorAccumulationMethod::Average:
            return std::make_shared<DescriptorAccumulatorByAverage>();
        case DescriptorAccumulationMethod::AverageNormed:
            return std::make_shared<DescriptorAccumulatorByAverageNormed>(QUALITY_THRESHOLD);
        case DescriptorAccumulationMethod::QualityScore:
            return  std::make_shared<DescriptorAccumulatorByQualityScore>(QUALITY_SCORE_MEASUREMENT_COUNT, COS_SCORE_THRESHOLD, QUALITY_THRESHOLD);
        case DescriptorAccumulationMethod::TopK:
            return  std::make_shared<DescriptorAccumulatorByTopK>(TOPK_COUNT_K, TOPK_TTL, COS_SCORE_THRESHOLD, QUALITY_THRESHOLD);
        default:
            throw std::logic_error("Unknown descriptor accumulation method");
    }
}
}
