#ifndef RANDOM_PICKER_PROXIES_H
#define RANDOM_PICKER_PROXIES_H

#include "../src/core/structs.h"
#include "../src/random/ExponentialIndexRandomPicker.cuh"
#include "../src/random/LinearRandomPicker.cuh"
#include "../src/random/UniformRandomPicker.cuh"
#include "../src/random/WeightBasedRandomPicker.cuh"

class ExponentialIndexRandomPickerProxy
{
private:
    GPUUsageMode gpu_usage;
    #ifdef HAS_CUDA
    std::unique_ptr<ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    std::unique_ptr<ExponentialIndexRandomPicker<GPUUsageMode::DATA_ON_GPU>> gpu_impl;
    std::unique_ptr<ExponentialIndexRandomPicker<GPUUsageMode::DATA_ON_HOST>> host_impl;
    #else
    std::unique_ptr<ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif


public:
    explicit ExponentialIndexRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            gpu_impl = std::make_unique<ExponentialIndexRandomPicker<GPUUsageMode::DATA_ON_GPU>>();
            break;
        case GPUUsageMode::DATA_ON_HOST:
            host_impl = std::make_unique<ExponentialIndexRandomPicker<GPUUsageMode::DATA_ON_HOST>>();
            break;
        default:  // ON_CPU
            cpu_impl = std::make_unique<ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>>();
        }
        #else
        cpu_impl = std::make_unique<ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>>();
        #endif
    }

    int pick_random(int start, int end, bool prioritize_end)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            return gpu_impl->pick_random(start, end, prioritize_end);
        case GPUUsageMode::DATA_ON_HOST:
            return host_impl->pick_random(start, end, prioritize_end);
        default:  // ON_CPU
            return cpu_impl->pick_random(start, end, prioritize_end);
        }
        #else
        return cpu_impl->pick_random(start, end, prioritize_end);
        #endif
    }
};


class LinearRandomPickerProxy
{
private:
    GPUUsageMode gpu_usage;
    #ifdef HAS_CUDA
    std::unique_ptr<LinearRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    std::unique_ptr<LinearRandomPicker<GPUUsageMode::DATA_ON_GPU>> gpu_impl;
    std::unique_ptr<LinearRandomPicker<GPUUsageMode::DATA_ON_HOST>> host_impl;
    #else
    std::unique_ptr<LinearRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif


public:
    explicit LinearRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            gpu_impl = std::make_unique<LinearRandomPicker<GPUUsageMode::DATA_ON_GPU>>();
            break;
        case GPUUsageMode::DATA_ON_HOST:
            host_impl = std::make_unique<LinearRandomPicker<GPUUsageMode::DATA_ON_HOST>>();
            break;
        default:  // ON_CPU
            cpu_impl = std::make_unique<LinearRandomPicker<GPUUsageMode::ON_CPU>>();
        }
        #else
        cpu_impl = std::make_unique<LinearRandomPicker<GPUUsageMode::ON_CPU>>();
        #endif
    }

    int pick_random(int start, int end, bool prioritize_end)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            return gpu_impl->pick_random(start, end, prioritize_end);
        case GPUUsageMode::DATA_ON_HOST:
            return host_impl->pick_random(start, end, prioritize_end);
        default:  // ON_CPU
            return cpu_impl->pick_random(start, end, prioritize_end);
        }
        #else
        return cpu_impl->pick_random(start, end, prioritize_end);
        #endif
    }
};


class UniformRandomPickerProxy
{
private:
    GPUUsageMode gpu_usage;
    #ifdef HAS_CUDA
    std::unique_ptr<UniformRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    std::unique_ptr<UniformRandomPicker<GPUUsageMode::DATA_ON_GPU>> gpu_impl;
    std::unique_ptr<UniformRandomPicker<GPUUsageMode::DATA_ON_HOST>> host_impl;
    #else
    std::unique_ptr<UniformRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif


public:
    explicit UniformRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            gpu_impl = std::make_unique<UniformRandomPicker<GPUUsageMode::DATA_ON_GPU>>();
            break;
        case GPUUsageMode::DATA_ON_HOST:
            host_impl = std::make_unique<UniformRandomPicker<GPUUsageMode::DATA_ON_HOST>>();
            break;
        default:  // ON_CPU
            cpu_impl = std::make_unique<UniformRandomPicker<GPUUsageMode::ON_CPU>>();
        }
        #else
        cpu_impl = std::make_unique<UniformRandomPicker<GPUUsageMode::ON_CPU>>();
        #endif
    }

    int pick_random(int start, int end, bool prioritize_end)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            return gpu_impl->pick_random(start, end, prioritize_end);
        case GPUUsageMode::DATA_ON_HOST:
            return host_impl->pick_random(start, end, prioritize_end);
        default:  // ON_CPU
            return cpu_impl->pick_random(start, end, prioritize_end);
        }
        #else
        return cpu_impl->pick_random(start, end, prioritize_end);
        #endif
    }
};

class WeightBasedRandomPickerProxy {
private:
    GPUUsageMode gpu_usage;
    #ifdef HAS_CUDA
    std::unique_ptr<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    std::unique_ptr<WeightBasedRandomPicker<GPUUsageMode::DATA_ON_GPU>> gpu_impl;
    std::unique_ptr<WeightBasedRandomPicker<GPUUsageMode::DATA_ON_HOST>> host_impl;
    #else
    std::unique_ptr<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif

public:
    explicit WeightBasedRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::DATA_ON_GPU:
            gpu_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::DATA_ON_GPU>>();
            break;
        case GPUUsageMode::DATA_ON_HOST:
            host_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::DATA_ON_HOST>>();
            break;
        default:  // ON_CPU
            cpu_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>>();
        }
        #else
        cpu_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>>();
        #endif
    }

    #ifdef HAS_CUDA
    int pick_random(const thrust::host_vector<double>& cumulative_weights, int group_start, int group_end) {
        return host_impl->pick_random(cumulative_weights, group_start, group_end);
    }

    int pick_random(const thrust::device_vector<double>& cumulative_weights, int group_start, int group_end) {
        return gpu_impl->pick_random(cumulative_weights, group_start, group_end);
    }
    #endif

    int pick_random(const std::vector<double>& cumulative_weights, int group_start, int group_end) {
        return cpu_impl->pick_random(cumulative_weights, group_start, group_end);
    }
};

#endif