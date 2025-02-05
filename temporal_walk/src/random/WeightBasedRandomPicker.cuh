#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include <cuda/dual_vector.cuh>

#include "RandomPicker.h"


#ifdef HAS_CUDA
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>
#endif

class WeightBasedRandomPicker final : public RandomPicker
{
public:
    template<typename T>
    [[nodiscard]] int pick_random(
        const DualVector<T>& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
