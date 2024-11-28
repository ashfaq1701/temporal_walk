#include "ExponentialRandomPicker.h"
#include <random>
#include <cmath>
#include <stdexcept>

int ExponentialRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // Instead of generating U ~ Uniform(0, e^n - 1) and taking log(U + 1),
    // we can generate u ~ Uniform(0, 1) and transform it to get the same distribution
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double u = dist(thread_local_gen);

    if (prioritize_end) {
        // For prioritize_end=true, we want P(i) ∝ exp(i)
        // The total weight is W = exp(len_seq) - 1.
        // We generate a random value u in [0, 1) and scale it to the range [0, W).
        // Using inverse transform sampling, we solve for index:
        //   F(i) = (exp(i) - 1) / W  =>  F^(-1)(u) = log(1 + u * W)
        // Here, W is computed using expm1(len_seq) to avoid overflow for large len_seq.
        const int index = static_cast<int>(std::log1p(u * std::expm1(len_seq)));
        return start + std::min(index, len_seq - 1);
    } else {
        // For prioritize_end=false, we want P(i) ∝ exp(n-1-i)
        // This is equivalent to len_seq - 1 - index from above
        const int index = len_seq - 1 - static_cast<int>(std::log1p(u * std::expm1(len_seq)));
        return start + std::max(0, index);
    }
}