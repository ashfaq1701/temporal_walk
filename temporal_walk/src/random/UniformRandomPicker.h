#ifndef UNIFORMRANDOMPICKER_H
#define UNIFORMRANDOMPICKER_H

#include "RandomPicker.h"

class UniformRandomPicker : public RandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //UNIFORMRANDOMPICKER_H
