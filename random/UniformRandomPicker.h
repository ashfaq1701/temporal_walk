#ifndef UNIFORMRANDOMPICKER_H
#define UNIFORMRANDOMPICKER_H

#include "RandomPicker.h"

class UniformRandomPicker : public RandomPicker {
public:
    int pick_random(int start, int end) override;
};

#endif //UNIFORMRANDOMPICKER_H
