#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "../data/enums.h"
#include "IndexBasedRandomPicker.h"

template<GPUUsageMode GPUUsage>
class LinearRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //LINEARRANDOMPICKER_H
