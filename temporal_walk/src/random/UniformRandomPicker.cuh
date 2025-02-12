#ifndef UNIFORMRANDOMPICKER_H
#define UNIFORMRANDOMPICKER_H

#include "../core/structs.h"

#include "IndexBasedRandomPicker.h"
#include "../utils/utils.h"

template<GPUUsageMode GPUUsage>
class UniformRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //UNIFORMRANDOMPICKER_H
