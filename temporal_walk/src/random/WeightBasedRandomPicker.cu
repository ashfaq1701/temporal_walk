#include "WeightBasedRandomPicker.cuh"

template<bool UseGPU>
int WeightBasedRandomPicker<UseGPU>::pick_random(
    const typename SelectVectorType<double, UseGPU>::type& cumulative_weights,
    const int group_start,
    const int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size())) {
        return -1;
    }

    // Get start and end sums
    double start_sum = 0.0;
    if (group_start > 0 && cumulative_weights[group_start] >= cumulative_weights[group_start - 1]) {
        start_sum = cumulative_weights[group_start - 1];
    }
    const double end_sum = cumulative_weights[group_end - 1];

    if (end_sum < start_sum) {
        return -1;
    }

    // Generate random value between [start_sum, end_sum]
    const double random_val = generate_random_value(start_sum, end_sum);

    // Find the group where random_val falls
    return static_cast<int>(std::lower_bound(
        cumulative_weights.begin() + group_start,
        cumulative_weights.begin() + group_end,
        random_val) - cumulative_weights.begin());
}

template class WeightBasedRandomPicker<false>;
#ifdef USE_CUDA
template class WeightBasedRandomPicker<true>;
#endif
