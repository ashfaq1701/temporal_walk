#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../cuda/types.cuh"

template<bool UseGPU>
class WeightBasedRandomPicker final : public RandomPicker<UseGPU>
{
public:
    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, UseGPU>::type& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
