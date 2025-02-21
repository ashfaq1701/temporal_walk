#ifndef COMMON_VECTOR_H
#define COMMON_VECTOR_H

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include "../core/structs.h"
#include "macros.cuh"

template <typename T, GPUUsageMode GPUUsage>
struct DeviceVector {
    T* data;
    size_t size;

    // Constructor
    HOST DEVICE DeviceVector() : data(nullptr), size(0) {}

    // Allocate memory based on the template parameter
    HOST DEVICE void allocate(size_t n) {
        size = n;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaError_t err = cudaMalloc(&data, size * sizeof(T)); // For device memory
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
            }
        } else
        #endif
        {
            data = static_cast<T*>(malloc(size * sizeof(T)));
            if (!data) {
                throw std::runtime_error("Host malloc failed!");
            }
        }
    }


    // Free memory
    HOST DEVICE void deallocate() {
        if (data) {
            #ifdef HAS_CUDA
            if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
                cudaFree(data); // Free device memory
            } else
            #endif
            {
                free(data); // Free regular host memory
            }
            data = nullptr;
            size = 0;
        }
    }

    HOST DEVICE void resize(size_t new_size) {
        if (new_size == size) return;  // No need to resize if the size is the same

        T* new_data = nullptr;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // Allocate new device memory
            cudaError_t err = cudaMalloc(&new_data, new_size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
            }

            // Copy old data to the new device memory (only if there is existing data)
            if (data) {
                size_t copy_size = (new_size < size) ? new_size : size;
                cudaMemcpy(new_data, data, copy_size * sizeof(T), cudaMemcpyDeviceToDevice);
                deallocate();  // Deallocate old memory
            }
        } else
        #endif
        {
            // Allocate new host memory
            new_data = static_cast<T*>(malloc(new_size * sizeof(T)));
            if (!new_data) {
                throw std::runtime_error("Host malloc failed!");
            }

            // Copy old data to the new host memory (only if there is existing data)
            if (data) {
                size_t copy_size = (new_size < size) ? new_size : size;
                memcpy(new_data, data, copy_size * sizeof(T));
                deallocate();  // Deallocate old memory
            }
        }

        // Update data pointer and size
        data = new_data;
        size = new_size;
    }

    HOST void assign(size_t new_size, const T& value) {
        resize(new_size);  // Resize the vector to the new size

        // Fill the vector with the specified value
        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // For device memory, use CUDA kernel or memory setting
            cudaMemset(data, value, new_size * sizeof(T));  // Set all elements to `value` on device
        } else
        #endif
        {
            // For host memory, use std::fill
            std::fill(data, data + new_size, value);  // Fill all elements to `value` on host
        }
    }

    // Indexing operator
    HOST DEVICE int& operator[](size_t i) {
        return data[i];
    }

    HOST DEVICE const int& operator[](size_t i) const {
        return data[i];
    }
};

#endif // COMMON_VECTOR_H
