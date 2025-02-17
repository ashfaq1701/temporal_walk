#include <stdexcept>
#include "../cuda_common/cuda_random_functions.cuh"
#include "UniformRandomPicker.cuh"

template<GPUUsageMode GPUUsage>
int UniformRandomPicker<GPUUsage>::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    if (GPUUsage == GPUUsageMode::ON_CPU) {
        return generate_random_int(start, end - 1);
    } else {
        #ifdef HAS_CUDA
        return generate_random_int_cuda(start, end - 1);
        #else
        throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        #endif
    }
}

template class UniformRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class UniformRandomPicker<GPUUsageMode::DATA_ON_GPU>;
template class UniformRandomPicker<GPUUsageMode::DATA_ON_HOST>;
#endif
