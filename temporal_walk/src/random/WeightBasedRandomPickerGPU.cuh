#ifndef WEIGHTBASEDRANDOMPICKERGPU_H
#define WEIGHTBASEDRANDOMPICKERGPU_H

#include <cuda_common/PolicyProvider.cuh>

#include "RandomPicker.h"
#include "../utils/utils.h"
#include "../core/structs.h"
#include "../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class WeightBasedRandomPickerGPU  final : public RandomPicker, public PolicyProvider<GPUUsage> {
public:
    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);
};



#endif //WEIGHTBASEDRANDOMPICKERGPU_H
