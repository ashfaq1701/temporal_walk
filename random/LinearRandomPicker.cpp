#include <random>
#include <stdexcept>

#include "LinearRandomPicker.h"

int LinearRandomPicker::pick_random(const int start, const int end) {
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<double> weights(end - start);
    double sumWeights = 0.0;

    for (int i = 0; i < end - start; ++i) {
        weights[i] = i;
        sumWeights += weights[i];
    }

    for (auto& weight : weights) {
        weight /= sumWeights;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double randomValue = dist(gen);

    double cumulativeProbability = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulativeProbability += weights[i];
        if (randomValue <= cumulativeProbability) {
            return start + (static_cast<int>(i) - 1);
        }
    }

    return end - 1;
}
