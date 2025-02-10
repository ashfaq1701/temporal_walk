#ifndef EXPONENTIALINDEXRANDOMPICKER_H
#define EXPONENTIALINDEXRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"
#include "../utils/utils.h"

class ExponentialIndexRandomPicker final : public IndexBasedRandomPicker {
public:
    bool use_gpu;

    explicit ExponentialIndexRandomPicker(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //EXPONENTIALINDEXRANDOMPICKER_H
