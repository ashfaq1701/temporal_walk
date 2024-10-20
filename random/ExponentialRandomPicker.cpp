#include "ExponentialRandomPicker.h"
#include <random>
#include <cmath>
#include <iostream>
#include <stdexcept>

int ExponentialRandomPicker::pick_random(const int start, const int end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    const double totalWeight = std::exp(end - start) - 1;

    std::uniform_real_distribution<double> dist(0.0, totalWeight);
    const double randomValue = dist(gen);

    const int index = static_cast<int>(std::log(randomValue + 1));

    return std::min(start + index, end - 1);
}
