#ifndef EXPONENTIALRANDOMPICKER_H
#define EXPONENTIALRANDOMPICKER_H

#include "RandomPicker.h"

class ExponentialRandomPicker : public RandomPicker {
public:
    int pick_random(int start, int end) override;
};

#endif //EXPONENTIALRANDOMPICKER_H
