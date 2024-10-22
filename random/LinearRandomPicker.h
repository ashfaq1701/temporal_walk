#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "RandomPicker.h"

class LinearRandomPicker : public RandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //LINEARRANDOMPICKER_H
