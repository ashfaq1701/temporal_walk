#ifndef WEIGHTBASEDRANDOMPICKER_H
#define WEIGHTBASEDRANDOMPICKER_H

#include "RandomPicker.h"

class WeightBasedRandomPicker : public RandomPicker {

public:
    virtual int pick_random(
        const std::vector<double>& prob,
        const std::vector<int>& alias,
        int start,
        int length) = 0;
};

#endif //WEIGHTBASEDRANDOMPICKER_H
