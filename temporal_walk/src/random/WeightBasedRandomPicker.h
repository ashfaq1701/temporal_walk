#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"

class WeightBasedRandomPicker final : public RandomPicker
{
public:
    template<typename T>
    [[nodiscard]] int pick_random(
        const std::vector<T>& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKER_H
