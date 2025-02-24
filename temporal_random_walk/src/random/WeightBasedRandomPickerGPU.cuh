// WeightBasedRandomPickerGPU.cuh
#ifndef WEIGHTBASEDRANDOMPICKERGPU_CUH
#define WEIGHTBASEDRANDOMPICKERGPU_CUH

#include "RandomPicker.h"
#include "../structs/enums.h"
#include "../common/types.cuh"


template<GPUUsageMode GPUUsage>
class WeightBasedRandomPickerGPU final : public RandomPicker {
private:
    double* d_random_val{};  // Persistent device memory

public:
    WeightBasedRandomPickerGPU();

    ~WeightBasedRandomPickerGPU() override;

    [[nodiscard]] int pick_random(
        const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
        int group_start,
        int group_end);
};

#endif //WEIGHTBASEDRANDOMPICKERGPU_CUH
