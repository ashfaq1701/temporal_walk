#ifndef EXPONENTIALINDEXRANDOMPICKER_H
#define EXPONENTIALINDEXRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"

template<bool UseGPU>
class ExponentialIndexRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //EXPONENTIALINDEXRANDOMPICKER_H
