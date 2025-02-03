#ifndef EXPONENTIALINDEXRANDOMPICKER_H
#define EXPONENTIALINDEXRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"

class ExponentialIndexRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end, bool use_gpu) override;
};

#endif //EXPONENTIALINDEXRANDOMPICKER_H
