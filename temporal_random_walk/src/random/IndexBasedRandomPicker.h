#ifndef INDEXBASEDRANDOMPICKER_H
#define INDEXBASEDRANDOMPICKER_H

#include "RandomPicker.h"
#include "../cuda_common/functions.cuh"

#ifdef HAS_CUDA
#include <curand_kernel.h>
#endif

template<GPUUsageMode GPUUsage>
class IndexBasedRandomPicker : public RandomPicker<GPUUsage> {

public:
    ~IndexBasedRandomPicker() override = default;

    virtual int pick_random(const int start, const int end, const bool prioritize_end)
    {
        int* picked_value;

        #ifdef HAS_CUDA
        if (GPUUsage == GPUUsageMode::ON_GPU)
        {
            int* d_picked_value;
            int h_picked_value = -1;
            curandState* d_rand_states;

            cudaMalloc(&d_picked_value, sizeof(int));
            cudaMalloc(&d_rand_states, sizeof(curandState));

            setup_curand_states<<<1, 1>>>(d_rand_states, time(nullptr));
            pick_random_kernel<<<1, 1>>>(this, start, end, prioritize_end, d_picked_value, d_rand_states);

            cudaMemcpy(&h_picked_value, d_picked_value, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_picked_value);
            cudaFree(d_rand_states);

            return h_picked_value;
        }
        else
        #endif
        {
            return this->pick_random_host(start, end, prioritize_end);
        }
    }

    virtual HOST int pick_random_host(int start, int end, bool prioritize_end) { return -1; }

    #ifdef HAS_CUDA
    virtual DEVICE int pick_random_device(int start, int end, bool prioritize_end, curandState* rand_state) { return -1; };
    #endif

    int get_picker_type() override
    {
        return INDEX_BASED_PICKER_TYPE;
    }
};

template<GPUUsageMode GPUUsage>
__global__ void pick_random_kernel(
    IndexBasedRandomPicker<GPUUsage>* random_picker,
    int start,
    int end,
    bool prioritize_end,
    int* picked_value, curandState* rand_states) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = rand_states[tid];
    *picked_value = random_picker->pick_random_device(start, end, prioritize_end, &localState);
    rand_states[tid] = localState;  // Store back the updated state
}

#endif //INDEXBASEDRANDOMPICKER_H
