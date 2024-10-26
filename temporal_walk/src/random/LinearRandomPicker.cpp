#include <random>
#include <stdexcept>

#include "LinearRandomPicker.h"

int LinearRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    const int len_seq = end - start;

    const double total_weight = len_seq * (len_seq + 1) / 2.0;

    std::uniform_real_distribution<double> dist(0.0, total_weight);
    const double randomValue = dist(gen);

    int index = static_cast<int>(floor((-1 + sqrt(1 + 8 * randomValue)) / 2));
    if (!prioritize_end) {
        index = len_seq - 1 - index;
    }

    return std::max(start, std::min(start + index, end - 1));
}
