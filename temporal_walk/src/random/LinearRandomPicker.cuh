#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "../core/structs.h"

#include "IndexBasedRandomPicker.h"
#include "../utils/utils.h"

template<GPUUsageMode GPUUsage>
class LinearRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //LINEARRANDOMPICKER_H
