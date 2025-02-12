#include <random>
#include <cmath>
#include <stdexcept>
#include "LinearRandomPicker.cuh"

// Derivation available in derivations folder
template<GPUUsageMode GPUUsage>
int LinearRandomPicker<GPUUsage>::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // For a sequence of length n, weights form an arithmetic sequence
    // When prioritizing end: weights are 1, 2, 3, ..., n
    // When prioritizing start: weights are n, n-1, n-2, ..., 1
    // Sum of arithmetic sequence = n(a1 + an)/2 = n(n+1)/2
    const long double total_weight = static_cast<long double>(len_seq) *
                                   (static_cast<long double>(len_seq) + 1.0L) / 2.0L;

    // Generate random value in [0, total_weight)
    const auto random_value = generate_random_value(0.0L, total_weight);

    // For both cases, we solve quadratic equation i² + i - 2r = 0
    // where r is our random value (or transformed random value)
    // Using quadratic formula: (-1 ± √(1 + 8r))/2
    const long double discriminant = 1.0L + 8.0L * random_value;
    const long double root = (-1.0L + std::sqrt(discriminant)) / 2.0L;
    const int index = static_cast<int>(std::floor(root));

    if (prioritize_end) {
        // For prioritize_end=true, larger indices should have higher probability
        return start + std::min(index, len_seq - 1);
    } else {
        // For prioritize_end=false, we reverse the index to give
        // higher probability to smaller indices
        const int revered_index = len_seq - 1 - index;
        return start + std::max(0, revered_index);
    }
}

template class LinearRandomPicker<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class LinearRandomPicker<GPUUsageMode::DATA_ON_GPU>;
template class LinearRandomPicker<GPUUsageMode::DATA_ON_HOST>;
#endif
