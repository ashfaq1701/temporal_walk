#include "WeightBasedRandomPicker.cuh"

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

    if (cumulative_weights.is_gpu()) {
        #ifdef HAS_CUDA
        // Use CUDA random generator
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<T> dist(0.0, cumulative_weights.device_at(group_end - 1));
        const T rand_val = dist(rng);

        auto it = thrust::upper_bound(
            thrust::device,
            cumulative_weights.device_begin() + group_start,
            cumulative_weights.device_begin() + group_end,
            rand_val
        );

        return group_start + thrust::distance(
            cumulative_weights.device_begin() + group_start,
            it
        );
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    } else {
        // Use CPU random generator
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<T> dist(0.0, cumulative_weights.host_at(group_end - 1));
        const T rand_val = dist(rng);

        auto it = std::upper_bound(
            cumulative_weights.host_begin() + group_start,
            cumulative_weights.host_begin() + group_end,
            rand_val
        );

        return group_start + std::distance(
            cumulative_weights.host_begin() + group_start,
            it
        );
    }
}