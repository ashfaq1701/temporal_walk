#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../utils/utils.h"
#include "../cuda_common/types.cuh"

class WeightBasedRandomPicker final : public RandomPicker
{
public:
    bool use_gpu;

    explicit WeightBasedRandomPicker(bool use_gpu);

    [[nodiscard]] int pick_random(
        const VectorTypes<double>::Vector& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
