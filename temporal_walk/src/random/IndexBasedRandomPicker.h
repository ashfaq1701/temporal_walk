#ifndef INDEXBASEDRANDOMPICKER_H
#define INDEXBASEDRANDOMPICKER_H

#include "RandomPicker.h"

template<bool UseGPU>
class IndexBasedRandomPicker : public RandomPicker<UseGPU> {

public:
    virtual int pick_random(int start, int end, bool prioritize_end) = 0;
};

#endif //INDEXBASEDRANDOMPICKER_H
