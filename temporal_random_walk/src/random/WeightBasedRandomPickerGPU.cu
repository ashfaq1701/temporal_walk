#include "WeightBasedRandomPickerGPU.cuh"

#ifdef HAS_CUDA

#include "../cuda_common/CudaRandomStates.cuh"

__global__ void generate_random_kernel(
    double* random_val,
    const double* cumulative_weights,
    const int group_start,
    const int group_end,
    curandState* states)
{
    // Calculate sums in the kernel
    double start_sum = 0.0;
    if (group_start > 0 && cumulative_weights[group_start] >= cumulative_weights[group_start - 1]) {
        start_sum = cumulative_weights[group_start - 1];
    }
    const double end_sum = cumulative_weights[group_end - 1];

    if (end_sum < start_sum) {
        // Signal error with a special value
        *random_val = -1.0;
        return;
    }

    // Generate random value
    *random_val = curand_uniform(&states[0]) * (end_sum - start_sum) + start_sum;
}


template<GPUUsageMode GPUUsage>
WeightBasedRandomPickerGPU<GPUUsage>::WeightBasedRandomPickerGPU() {
    cudaMalloc(&d_random_val, sizeof(double));
}

template<GPUUsageMode GPUUsage>
WeightBasedRandomPickerGPU<GPUUsage>::~WeightBasedRandomPickerGPU() {
    if (d_random_val) {
        cudaFree(d_random_val);
    }
}


template<GPUUsageMode GPUUsage>
int WeightBasedRandomPickerGPU<GPUUsage>::pick_random(
    const typename SelectVectorType<double, GPUUsage>::type& cumulative_weights,
    const int group_start,
    const int group_end)
{
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size())) {
        return -1;
        }

    // Generate random value using kernel
    generate_random_kernel<<<1, 1>>>(
        d_random_val,
        cumulative_weights.data,
        group_start,
        group_end,
        CUDARandomStates::get_states()
    );

    double random_val;
    cudaMemcpy(&random_val, d_random_val, sizeof(double), cudaMemcpyDeviceToHost);

    // Check for error condition
    if (random_val < 0) {
        return -1;
    }

    return static_cast<int>(std::lower_bound(
            cumulative_weights.begin() + group_start,
            cumulative_weights.begin() + group_end,
            random_val) - cumulative_weights.begin());
}


template class WeightBasedRandomPickerGPU<GPUUsageMode::ON_GPU>;
#endif
