#include <stdexcept>
#include "UniformRandomPicker.cuh"

template<GPUUsageMode GPUUsage>
int UniformRandomPicker<GPUUsage>::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    return generate_random_int(start, end - 1);
}

template class UniformRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class UniformRandomPicker<GPUUsageMode::DATA_ON_GPU>;
template class UniformRandomPicker<GPUUsageMode::DATA_ON_HOST>;
#endif
