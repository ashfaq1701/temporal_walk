#include <random>
#include <stdexcept>

#include "LinearRandomPicker.h"

int LinearRandomPicker::pick_random(const int start, const int end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    const double totalWeight = (end - start) * (end - start + 1) / 2.0;

    std::uniform_real_distribution<double> dist(0.0, totalWeight);
    const double randomValue = dist(gen);

    const int index = static_cast<int>(floor((-1 + sqrt(1 + 8 * randomValue)) / 2));

    return std::min(start + index, end - 1);
}
