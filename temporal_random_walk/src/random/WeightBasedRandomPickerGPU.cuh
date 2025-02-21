// WeightBasedRandomPickerGPU.cuh
#ifndef WEIGHTBASEDRANDOMPICKERGPU_CUH
#define WEIGHTBASEDRANDOMPICKERGPU_CUH

#include <cuda_common/PolicyProvider.cuh>

#include "RandomPicker.h"
#include "../core/structs.h"
#include "../cuda_common/types.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

template<GPUUsageMode GPUUsage>
class WeightBasedRandomPickerGPU final : public RandomPicker, public PolicyProvider<GPUUsage> {
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
