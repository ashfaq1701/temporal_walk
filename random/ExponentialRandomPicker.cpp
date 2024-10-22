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

    const double totalWeight = std::exp(len_seq) - 1;

    std::uniform_real_distribution<double> dist(0.0, totalWeight);
    const double randomValue = dist(gen);

    int index = static_cast<int>(std::log(randomValue + 1));
    if (!prioritize_end) {
        index = len_seq - index;
    }

    return std::min(start + index, end - 1);
}
