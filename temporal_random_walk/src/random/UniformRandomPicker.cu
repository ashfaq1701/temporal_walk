#include "UniformRandomPicker.cuh"

#include <stdexcept>
#include "../utils/rand_utils.cuh"

template<GPUUsageMode GPUUsage>
int UniformRandomPicker<GPUUsage>::pick_random_host(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    return generate_random_int_host(start, end - 1);
}

#ifdef HAS_CUDA
template<GPUUsageMode GPUUsage>
int UniformRandomPicker<GPUUsage>::pick_random_device(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        return -1;
    }

    return generate_random_int_device(start, end - 1);
}
#endif

template class UniformRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class UniformRandomPicker<GPUUsageMode::ON_GPU>;
#endif
