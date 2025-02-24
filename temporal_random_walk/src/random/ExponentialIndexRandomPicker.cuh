#ifndef EXPONENTIALINDEXRANDOMPICKER_H
#define EXPONENTIALINDEXRANDOMPICKER_H

#include "../structs/structs.cuh"
#include "../structs/enums.h"

#include "IndexBasedRandomPicker.h"
#include "../utils/utils.h"

template<GPUUsageMode GPUUsage>
class ExponentialIndexRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //EXPONENTIALINDEXRANDOMPICKER_H
