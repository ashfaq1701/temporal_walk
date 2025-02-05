#ifndef CUDA_FUNCTIONS_CUH
#define CUDA_FUNCTIONS_CUH

#include <algorithm>
#include <numeric>
#include "dual_vector.cuh"

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#endif

namespace cuda_functions {

    template<typename T>
    double get_value_at(const DualVector<T>& vec, int index, bool use_gpu) {
        #ifdef HAS_CUDA
        return use_gpu ? vec.device_at(index) : vec.host_at(index);
        #else
        return vec.host_at(index);
        #endif
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

#endif //CUDA_FUNCTIONS_CUH
