#ifndef EXPONENTIALWEIGHTRANDOMPICKER_H
#define EXPONENTIALWEIGHTRANDOMPICKER_H

#include "WeightBasedRandomPicker.h"
#include <vector>

class ExponentialWeightRandomPicker : public WeightBasedRandomPicker
{
public:
     int pick_random(
        const std::vector<double>& weights,
        int group_start,
        int group_end) override;
};

#endif //EXPONENTIALWEIGHTRANDOMPICKER_H
