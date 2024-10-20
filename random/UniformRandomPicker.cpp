#include <random>
#include <stdexcept>
#include "UniformRandomPicker.h"

int UniformRandomPicker::pick_random(const int start, const int end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(start, end - 1);

    return dist(gen);
}
