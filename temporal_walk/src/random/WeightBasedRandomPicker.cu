#include "WeightBasedRandomPicker.cuh"
#include "../cuda/cuda_functions.cuh"
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

    const bool use_gpu = cumulative_weights.is_gpu();

    // Get start and end sums
    const T start_sum = (group_start > 0) ?
        cuda_functions::get_value_at(cumulative_weights, group_start - 1, use_gpu) : T{0};
    const double end_sum = cuda_functions::get_value_at(cumulative_weights, group_end - 1, use_gpu);

    if (end_sum <= start_sum) {
        return -1;
    }

    // Generate random value between [start_sum, end_sum]
    const T random_val = start_sum +
        cuda_random_functions::generate_uniform_random(T{0}, end_sum - start_sum, use_gpu);

    // Find the index where random_val falls using the appropriate search
    if (use_gpu) {
        #ifdef HAS_CUDA
        auto it = thrust::lower_bound(
            thrust::device,
            cumulative_weights.device_begin() + group_start,
            cumulative_weights.device_begin() + group_end,
            random_val
        );
        return static_cast<int>(thrust::distance(cumulative_weights.device_begin(), it));
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    } else {
        auto it = std::lower_bound(
            cumulative_weights.host_begin() + group_start,
            cumulative_weights.host_begin() + group_end,
            random_val
        );
        return static_cast<int>(std::distance(cumulative_weights.host_begin(), it));
    }
}

template int WeightBasedRandomPicker::pick_random<double>(
    const DualVector<double>&, const int, const int);
