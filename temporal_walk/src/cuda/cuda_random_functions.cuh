#ifndef CUDA_RANDOM_FUNCTIONS_CUH
#define CUDA_RANDOM_FUNCTIONS_CUH

#include "DualVector.cuh"

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#endif

namespace cuda_random_functions {

    template<typename T>
    std::pair<T, size_t> pick_random_with_weights(
        const DualVector<T>& cumulative_weights,
        const size_t group_start,
        const size_t group_end,
        const bool use_gpu)
    {
        if (group_start >= group_end ||
            group_end > cumulative_weights.size())
        {
            return {T{}, static_cast<size_t>(-1)};
        }

        if (use_gpu) {
            #ifdef HAS_CUDA
            // Use CUDA random generator
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<T> dist(0.0, cumulative_weights.device_at(group_end - 1));
            const T rand_val = dist(rng);

            auto it = thrust::upper_bound(
                thrust::device,
                cumulative_weights.device_begin() + group_start,
                cumulative_weights.device_begin() + group_end,
                rand_val
            );

            const size_t index = group_start + thrust::distance(
                cumulative_weights.device_begin() + group_start,
                it
            );

            return {rand_val, index};
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        } else {
            // Use CPU random generator
            thread_local std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<T> dist(0.0, cumulative_weights.host_at(group_end - 1));
            const T rand_val = dist(rng);

            auto it = std::upper_bound(
                cumulative_weights.host_begin() + group_start,
                cumulative_weights.host_begin() + group_end,
                rand_val
            );

            const size_t index = group_start + std::distance(
                cumulative_weights.host_begin() + group_start,
                it
            );

            return {rand_val, index};
        }
    }

    template<typename T>
    double generate_uniform_random(const T start, const T end, const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<T> dist(start, end);
            return dist(rng);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        } else {
            thread_local std::mt19937 thread_local_gen(std::random_device{}());
            std::uniform_real_distribution<T> dist(start, end);
            return dist(thread_local_gen);
        }
    }

    inline int generate_uniform_random_int(const int start, const int end, const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::default_random_engine rng;
            thrust::uniform_int_distribution dist(start, end);
            return dist(rng);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        } else {
            thread_local std::mt19937 thread_local_gen(std::random_device{}());
            std::uniform_int_distribution<> dist(start, end);
            return dist(thread_local_gen);
        }
    }
}

#endif //CUDA_RANDOM_FUNCTIONS_CUH
