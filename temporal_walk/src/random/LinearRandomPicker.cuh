#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"
#include "../utils/utils.h"

class LinearRandomPicker final : public IndexBasedRandomPicker {
public:
    bool use_gpu;

    explicit LinearRandomPicker(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //LINEARRANDOMPICKER_H
