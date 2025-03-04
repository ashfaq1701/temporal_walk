#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../data/enums.h"
#include "../cuda_common/types.cuh"
#include "../cuda_common/macros.cuh"

template<GPUUsageMode GPUUsage>
class WeightBasedRandomPicker final : public RandomPicker<GPUUsage>
{
public:
    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end)
    {
        #ifdef HAS_CUDA
        if (GPUUsage == GPUUsageMode::ON_GPU)
        {
            return this->pick_random_device(cumulative_weights, group_start, group_end);
        }
        else
        #endif
        {
            return this->pick_random_host(cumulative_weights, group_start, group_end);
        }
    };

    [[nodiscard]] HOST int pick_random_host(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);

    #ifdef HAS_CUDA
    [[nodiscard]] DEVICE int pick_random_device(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);
    #endif

    int get_picker_type() override
    {
        return WEIGHT_BASED_PICKER_TYPE;
    }
};

#endif //WEIGHTBASEDRANDOMPICKER_H
