#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"

template<bool UseGPU>
class LinearRandomPicker final : public IndexBasedRandomPicker<UseGPU> {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //LINEARRANDOMPICKER_H
