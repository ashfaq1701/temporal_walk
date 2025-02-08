#ifndef RANDOM_PICKER_PROXIES_H
#define RANDOM_PICKER_PROXIES_H

#include <random/ExponentialIndexRandomPicker.cuh>
#include <random/LinearRandomPicker.cuh>
#include <random/UniformRandomPicker.cuh>
#include <random/WeightBasedRandomPicker.cuh>

class ExponentialIndexRandomPickerProxy
{
private:
    bool use_gpu;
    #ifdef USE_CUDA
    std::unique_ptr<ExponentialIndexRandomPicker<false>> cpu_impl;
    std::unique_ptr<ExponentialIndexRandomPicker<true>> gpu_impl;
    #else
    std::unique_ptr<ExponentialIndexRandomPicker<false>> cpu_impl;
    #endif

public:
    explicit ExponentialIndexRandomPickerProxy(bool use_gpu): use_gpu(use_gpu)
    {
        #ifdef USE_CUDA
        this->use_gpu = use_gpu;
        if (use_gpu) {
            gpu_impl = std::make_unique<ExponentialIndexRandomPicker<true>>();
        } else {
            cpu_impl = std::make_unique<ExponentialIndexRandomPicker<false>>();
        }
        #else
        cpu_impl = std::make_unique<ExponentialIndexRandomPicker<false>>();
        #endif
    }

    int pick_random(int start, int end, bool prioritize_end)
    {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->pick_random(start, end, prioritize_end);
        else return cpu_impl->pick_random(start, end, prioritize_end);
        #else
        return cpu_impl->pick_random(start, end, prioritize_end);
        #endif
    }
};


class LinearRandomPickerProxy {
private:
    bool use_gpu;
    #ifdef USE_CUDA
    std::unique_ptr<LinearRandomPicker<false>> cpu_impl;
    std::unique_ptr<LinearRandomPicker<true>> gpu_impl;
    #else
    std::unique_ptr<LinearRandomPicker<false>> cpu_impl;
    #endif

public:
   explicit LinearRandomPickerProxy(bool use_gpu): use_gpu(use_gpu) {
       #ifdef USE_CUDA
       this->use_gpu = use_gpu;
       if (use_gpu) {
           gpu_impl = std::make_unique<LinearRandomPicker<true>>();
       } else {
           cpu_impl = std::make_unique<LinearRandomPicker<false>>();
       }
       #else
       cpu_impl = std::make_unique<LinearRandomPicker<false>>();
       #endif
   }

   int pick_random(int start, int end, bool prioritize_end) {
       #ifdef USE_CUDA
       if (use_gpu) return gpu_impl->pick_random(start, end, prioritize_end);
       else return cpu_impl->pick_random(start, end, prioritize_end);
       #else
       return cpu_impl->pick_random(start, end, prioritize_end);
       #endif
   }
};


class UniformRandomPickerProxy {
private:
    bool use_gpu;
    #ifdef USE_CUDA
    std::unique_ptr<UniformRandomPicker<false>> cpu_impl;
    std::unique_ptr<UniformRandomPicker<true>> gpu_impl;
    #else
    std::unique_ptr<UniformRandomPicker<false>> cpu_impl;
    #endif

public:
   explicit UniformRandomPickerProxy(bool use_gpu): use_gpu(use_gpu) {
       #ifdef USE_CUDA
       this->use_gpu = use_gpu;
       if (use_gpu) {
           gpu_impl = std::make_unique<UniformRandomPicker<true>>();
       } else {
           cpu_impl = std::make_unique<UniformRandomPicker<false>>();
       }
       #else
       cpu_impl = std::make_unique<UniformRandomPicker<false>>();
       #endif
   }

   int pick_random(int start, int end, bool prioritize_end) {
       #ifdef USE_CUDA
       if (use_gpu) return gpu_impl->pick_random(start, end, prioritize_end);
       else return cpu_impl->pick_random(start, end, prioritize_end);
       #else
       return cpu_impl->pick_random(start, end, prioritize_end);
       #endif
   }
};

class WeightBasedRandomPickerProxy {
private:
    bool use_gpu;
    #ifdef USE_CUDA
    std::unique_ptr<WeightBasedRandomPicker<false>> cpu_impl;
    std::unique_ptr<WeightBasedRandomPicker<true>> gpu_impl;
    #else
    std::unique_ptr<WeightBasedRandomPicker<false>> cpu_impl;
    #endif

public:
    explicit WeightBasedRandomPickerProxy(bool use_gpu): use_gpu(use_gpu) {
        #ifdef USE_CUDA
        this->use_gpu = use_gpu;
        if (use_gpu) {
            gpu_impl = std::make_unique<WeightBasedRandomPicker<true>>();
        } else {
            cpu_impl = std::make_unique<WeightBasedRandomPicker<false>>();
        }
        #else
        cpu_impl = std::make_unique<WeightBasedRandomPicker<false>>();
        #endif
    }

    #ifdef USE_CUDA
    int pick_random(const thrust::host_vector<double>& cumulative_weights, int group_start, int group_end) {
        return gpu_impl->pick_random(cumulative_weights, group_start, group_end);
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