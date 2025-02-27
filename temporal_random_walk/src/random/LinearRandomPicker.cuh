#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "../data/enums.h"
#include "../cuda_common/macros.cuh"
#include "IndexBasedRandomPicker.h"

template<GPUUsageMode GPUUsage>
class LinearRandomPicker final : public IndexBasedRandomPicker<GPUUsage> {
public:
    HOST int pick_random_host(int start, int end, bool prioritize_end) override;

    #ifdef HAS_CUDA
    DEVICE int pick_random_device(int start, int end, bool prioritize_end) override;
    #endif
};

#endif //LINEARRANDOMPICKER_H
