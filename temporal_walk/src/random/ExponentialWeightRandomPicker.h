#ifndef EXPONENTIALWEIGHTRANDOMPICKER_H
#define EXPONENTIALWEIGHTRANDOMPICKER_H

#include "WeightBasedRandomPicker.h"
#include <vector>

class ExponentialWeightRandomPicker : public WeightBasedRandomPicker
{
public:
    [[nodiscard]] int pick_random(
        const std::vector<double>& group_probs,
        const std::vector<int>& group_aliases,
        int group_start,
        int group_length) override;
};

#endif //EXPONENTIALWEIGHTRANDOMPICKER_H
