#include "ExponentialRandomPicker.h"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

int ExponentialRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    const double total_weight = std::expm1(len_seq);
    std::uniform_real_distribution<double> dist(0.0, total_weight);
    const double random_value = dist(thread_local_gen);

    const int index = static_cast<int>(std::log1p(random_value));


    if (prioritize_end) {
        // For prioritizing end: P(i) ∝ exp(i)
        return start + std::min(index, len_seq - 1);
    } else {
        // For prioritizing start: P(i) ∝ exp(len_seq - 1 - i)
        // So index i gets weight e^(len_seq - 1 - i)
        // This gives highest weight e^(len_seq-1) to index 0
        const int revered_index = len_seq - 1 - index;
        return start + std::max(0, revered_index);
    }
}
