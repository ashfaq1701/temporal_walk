#include <stdexcept>
#include "UniformRandomPicker.cuh"

#include <cuda/cuda_functions.cuh>

int UniformRandomPicker::pick_random(const int start, const int end, const bool prioritize_end, const bool use_gpu) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    return cuda_functions::generate_uniform_random_int(start, end - 1, use_gpu);
}
