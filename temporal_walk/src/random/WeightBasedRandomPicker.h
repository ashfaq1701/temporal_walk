#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"

class WeightBasedRandomPicker : public RandomPicker
{
public:
    [[nodiscard]] int pick_random(
        const std::vector<double>& weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
