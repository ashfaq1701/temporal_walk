#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../utils/utils.h"
#include "../core/structs.h"
#include "../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class WeightBasedRandomPicker final : public RandomPicker
{
public:
    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
