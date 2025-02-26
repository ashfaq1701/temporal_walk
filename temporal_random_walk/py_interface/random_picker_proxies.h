#ifndef RANDOM_PICKER_PROXIES_H
#define RANDOM_PICKER_PROXIES_H

#include "../src/data/structs.cuh"
#include "../src/data/enums.h"
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
    std::unique_ptr<ExponentialIndexRandomPicker<GPUUsageMode::ON_GPU>> gpu_impl;
    #else
    std::unique_ptr<ExponentialIndexRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif


public:
    explicit ExponentialIndexRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifndef HAS_CUDA
        if (gpu_usage != GPUUsageMode::ON_CPU) {
            throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        }
        #endif

        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::ON_GPU:
            gpu_impl = std::make_unique<ExponentialIndexRandomPicker<GPUUsageMode::ON_GPU>>();
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
        case GPUUsageMode::ON_GPU:
            return gpu_impl->pick_random(start, end, prioritize_end);
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
    std::unique_ptr<LinearRandomPicker<GPUUsageMode::ON_GPU>> gpu_impl;
    #else
    std::unique_ptr<LinearRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif


public:
    explicit LinearRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifndef HAS_CUDA
        if (gpu_usage != GPUUsageMode::ON_CPU) {
            throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        }
        #endif

        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::ON_GPU:
            gpu_impl = std::make_unique<LinearRandomPicker<GPUUsageMode::ON_GPU>>();
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
        case GPUUsageMode::ON_GPU:
            return gpu_impl->pick_random(start, end, prioritize_end);
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
    std::unique_ptr<UniformRandomPicker<GPUUsageMode::ON_GPU>> gpu_impl;
    #else
    std::unique_ptr<UniformRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif


public:
    explicit UniformRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifndef HAS_CUDA
        if (gpu_usage != GPUUsageMode::ON_CPU) {
            throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        }
        #endif

        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::ON_GPU:
            gpu_impl = std::make_unique<UniformRandomPicker<GPUUsageMode::ON_GPU>>();
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
        case GPUUsageMode::ON_GPU:
            return gpu_impl->pick_random(start, end, prioritize_end);
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
    std::unique_ptr<WeightBasedRandomPicker<GPUUsageMode::ON_GPU>> gpu_impl;
    #else
    std::unique_ptr<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>> cpu_impl;
    #endif

public:
    explicit WeightBasedRandomPickerProxy(GPUUsageMode gpu_usage): gpu_usage(gpu_usage)
    {
        #ifndef HAS_CUDA
        if (gpu_usage != GPUUsageMode::ON_CPU) {
            throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
        }
        #endif

        #ifdef HAS_CUDA
        switch(gpu_usage) {
        case GPUUsageMode::ON_GPU:
            gpu_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::ON_GPU>>();
            break;
        default:  // ON_CPU
            cpu_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>>();
        }
        #else
        cpu_impl = std::make_unique<WeightBasedRandomPicker<GPUUsageMode::ON_CPU>>();
        #endif
    }

    int pick_random(const std::vector<double>& cumulative_weights, int group_start, int group_end) {
        CommonVector<double, GPUUsageMode::ON_CPU> cumulative_weights_vector;
        cumulative_weights_vector.write_from_pointer(cumulative_weights.data(), cumulative_weights.size());
        return cpu_impl->pick_random(cumulative_weights_vector, group_start, group_end);
    }
};

#endif