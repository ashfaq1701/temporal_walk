#include "ExponentialIndexRandomPicker.h"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Derivation available in derivations folder
int ExponentialIndexRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // Generate uniform random number between 0 and 1
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double u = dist(thread_local_gen);

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