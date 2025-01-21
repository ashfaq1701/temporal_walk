#ifndef EXPONENTIALWEIGHTRANDOMPICKER_H
#define EXPONENTIALWEIGHTRANDOMPICKER_H

#include "WeightBasedRandomPicker.h"
#include <vector>

class ExponentialWeightRandomPicker : public WeightBasedRandomPicker {
public:
    [[nodiscard]] int pick_random(
        const std::vector<double>& prob,
        const std::vector<int>& alias,
        int start,
        int length) override;
};

#endif //EXPONENTIALWEIGHTRANDOMPICKER_H