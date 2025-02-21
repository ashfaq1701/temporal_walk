#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../core/structs.h"
#include "../cuda_common/types.cuh"
#include "../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class WeightBasedRandomPicker final : public RandomPicker, public PolicyProvider<GPUUsage>
{
public:
    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
