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

    const double total_weight = std::expm1(len_seq);

    std::uniform_real_distribution<double> dist(0.0, total_weight);
    const double random_value = dist(gen);

    int index = static_cast<int>(std::log1p(random_value));

    if (!prioritize_end) {
        index = len_seq - 1 - index;
    }

    return std::max(start, std::min(start + index, end - 1));
}
