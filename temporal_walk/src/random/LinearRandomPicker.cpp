#include <random>
#include <stdexcept>
#include "LinearRandomPicker.h"

int LinearRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;
    const double total_weight = len_seq * (len_seq + 1) / 2.0;

    std::uniform_real_distribution<double> dist(0.0, total_weight);
    const double randomValue = dist(thread_local_gen);

    if (prioritize_end) {
        // weight(i) = i + 1
        const int index = static_cast<int>(floor((-1 + sqrt(1 + 8 * randomValue)) / 2));
        return start + std::min(index, len_seq - 1);
    } else {
        const double shifted_random = total_weight - randomValue;
        const int index = len_seq - 1 - static_cast<int>(floor((-1 + sqrt(1 + 8 * shifted_random)) / 2));
        return start + std::max(0, std::min(index, len_seq - 1));
    }
}
