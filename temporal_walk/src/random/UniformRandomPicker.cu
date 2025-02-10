#include <random>
#include <stdexcept>
#include "UniformRandomPicker.cuh"

UniformRandomPicker::UniformRandomPicker(bool use_gpu): use_gpu(use_gpu) {}

int UniformRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::uniform_int_distribution<> dist(start, end - 1);

    return dist(thread_local_gen);
}
