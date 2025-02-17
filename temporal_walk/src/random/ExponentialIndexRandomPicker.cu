#include "ExponentialIndexRandomPicker.cuh"
#include <cmath>
#include <stdexcept>
#include "../cuda_common/cuda_random_functions.cuh"

// Derivation available in derivations folder
template<GPUUsageMode GPUUsage>
int ExponentialIndexRandomPicker<GPUUsage>::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // Generate uniform random number between 0 and 1
    double u;
    if (GPUUsage == GPUUsageMode::ON_CPU) {
        u = generate_random_value(0.0, 1.0);
    } else {
        #ifdef HAS_CUDA
        u = generate_random_value_cuda(0.0, 1.0);
        #else
        throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        #endif
    }

    double k;
    if (len_seq < 710) {
        // Inverse CDF formula,
        // k = ln(1 + u * (e^len seq − 1)) − 1
        k = log1p(u * expm1(len_seq)) - 1;
    } else {
        // Inverse CDF approximation for large len_seq,
        // k = len_seq + ln(u) − 1
        k = len_seq + std::log(u) - 1;
    }

    // Due to rounding, the trailing "-1" in the inverse CDF formula causes error.
    // To compensate for this we add 1 with k.
    // And bound the results within limits.
    const int rounded_index = std::max(0, std::min(static_cast<int>(k + 1), len_seq - 1));

    if (prioritize_end) {
        return start + rounded_index;
    } else {
        return start + (len_seq - 1 - rounded_index);
    }
}

template class ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class ExponentialIndexRandomPicker<GPUUsageMode::DATA_ON_GPU>;
template class ExponentialIndexRandomPicker<GPUUsageMode::DATA_ON_HOST>;
#endif
