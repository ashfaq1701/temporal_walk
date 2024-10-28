#include "ExponentialRandomPicker.h"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

int ExponentialRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    const int len_seq = end - start;

    if (prioritize_end) {
        // For prioritizing end: P(i) ∝ exp(i)
        const double total_weight = std::expm1(len_seq);

        std::uniform_real_distribution<double> dist(0.0, total_weight);
        const double random_value = dist(gen);

        const int index = static_cast<int>(std::log1p(random_value));
        return start + std::min(index, len_seq - 1);
    } else {
        // For prioritizing start: P(i) ∝ exp(-i)
        // First, compute total weight for normalization (sum of exp(-i) from 0 to len_seq-1)
        const double total_weight = (1 - std::exp(-len_seq)) / (1 - std::exp(-1));

        std::uniform_real_distribution<double> dist(0.0, total_weight);
        const double random_value = dist(gen);

        // Using inverse CDF to directly compute the index
        // If F(x) = (1-e^(-x))/(1-e^(-1)) is our CDF
        // Then F^(-1)(y) = -ln(1 - y*(1-e^(-1)))
        const double inverse_cdf = -std::log(1 - random_value * (1 - std::exp(-1)));

        const int index = static_cast<int>(inverse_cdf);
        return start + std::min(index, len_seq - 1);
    }
}
