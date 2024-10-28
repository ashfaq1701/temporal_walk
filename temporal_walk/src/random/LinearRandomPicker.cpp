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

    if (prioritize_end) {
        // weight(i) = i + 1
        int index = static_cast<int>(floor((-1 + sqrt(1 + 8 * randomValue)) / 2));
        return start + std::min(index, len_seq - 1);
    } else {
        // weight(i) = len_seq - i
        // Solve: (len_seq)(len_seq + 1)/2 - (len_seq - index)(len_seq - index + 1)/2 = randomValue
        // This gives us the index where the cumulative weight exceeds randomValue
        int index = len_seq - static_cast<int>(floor((-1 + sqrt(1 + 8 * (total_weight - randomValue))) / 2));
        return start + std::min(index, len_seq - 1);
    }
}
