#include "WeightBasedRandomPicker.cuh"
#include "../cuda/cuda_random_functions.cuh"

template<typename T>
int WeightBasedRandomPicker::pick_random(
    const DualVector<T>& cumulative_weights,
    const int group_start,
    const int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size()))
    {
        return -1;
    }

    // Call the platform-specific implementation
    auto [rand_val, index] = cuda_random_functions::pick_random_with_weights(
        cumulative_weights,
        static_cast<size_t>(group_start),
        static_cast<size_t>(group_end),
        cumulative_weights.is_gpu()
    );

    return static_cast<int>(index);
}

template int WeightBasedRandomPicker::pick_random<double>(
    const DualVector<double>&, const int, const int);
