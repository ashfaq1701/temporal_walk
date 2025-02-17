#include "WeightBasedRandomPickerGPU.cuh"

template<GPUUsageMode GPUUsage>
int WeightBasedRandomPickerGPU<GPUUsage>::pick_random(
    const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
    const int group_start,
    const int group_end) {
    return group_start;
}