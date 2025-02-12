#include <random>
#include <stdexcept>
#include "UniformRandomPicker.cuh"

template<GPUUsageMode GPUUsage>
int UniformRandomPicker<GPUUsage>::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::uniform_int_distribution<> dist(start, end - 1);

    return dist(thread_local_gen);
}

template class UniformRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class UniformRandomPicker<GPUUsageMode::DATA_ON_GPU>;
template class UniformRandomPicker<GPUUsageMode::DATA_ON_HOST>;
#endif
