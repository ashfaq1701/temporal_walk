#ifndef INDEXBASEDRANDOMPICKER_H
#define INDEXBASEDRANDOMPICKER_H

#include "RandomPicker.h"

class IndexBasedRandomPicker : public RandomPicker {

public:
    virtual int pick_random(int start, int end, bool prioritize_end) = 0;
};

#endif //INDEXBASEDRANDOMPICKER_H