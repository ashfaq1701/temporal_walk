#ifndef COMMON_VECTOR_H
#define COMMON_VECTOR_H

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include "../core/structs.h"
#include "macros.cuh"

template <typename T, GPUUsageMode GPUUsage>
struct CommonVector {
    T* data;
    size_t size;
    size_t capacity;
    size_t initial_capacity;
    T default_value;

    // Constructor
    HOST DEVICE explicit CommonVector(size_t initial_cap = 100, T default_val = T())
        : data(nullptr)
        , size(0)
        , capacity(0)
        , initial_capacity(initial_cap)
        , default_value(default_val) {
        allocate(initial_cap);
    }

    // Destructor
    HOST DEVICE ~CommonVector() {
        deallocate();
    }

    // Copy constructor
    HOST DEVICE CommonVector(const CommonVector& other)
        : data(nullptr)
        , size(0)
        , capacity(0)
        , initial_capacity(other.initial_capacity)
        , default_value(other.default_value) {
        allocate(other.capacity);
        size = other.size;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaMemcpy(data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice);
        } else
        #endif
        {
            std::copy(other.data, other.data + size, data);
        }
    }

    // Move constructor
    HOST DEVICE CommonVector(CommonVector&& other) noexcept
        : data(other.data)
        , size(other.size)
        , capacity(other.capacity)
        , initial_capacity(other.initial_capacity)
        , default_value(other.default_value) {
        other.data = nullptr;
        other.size = 0;
        other.capacity = 0;
    }

    // Copy assignment
    HOST DEVICE CommonVector& operator=(const CommonVector& other) {
        if (this != &other) {
            deallocate();
            allocate(other.capacity);
            size = other.size;
            initial_capacity = other.initial_capacity;
            default_value = other.default_value;

            #ifdef HAS_CUDA
            if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
                cudaMemcpy(data, other.data, size * sizeof(T), cudaMemcpyDeviceToDevice);
            } else
            #endif
            {
                std::copy(other.data, other.data + size, data);
            }
        }
        return *this;
    }

    // Move assignment
    HOST DEVICE CommonVector& operator=(CommonVector&& other) noexcept {
        if (this != &other) {
            deallocate();
            data = other.data;
            size = other.size;
            capacity = other.capacity;
            initial_capacity = other.initial_capacity;
            default_value = other.default_value;

            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;
        }
        return *this;
    }

    // Allocate memory
    HOST DEVICE void allocate(size_t n) {
        if (n == 0) return;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaError_t err = cudaMalloc(&data, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed!");
            }
        } else
        #endif
        {
            data = static_cast<T*>(malloc(n * sizeof(T)));
            if (!data) {
                throw std::runtime_error("Host malloc failed!");
            }
        }
        capacity = n;
    }

    // Deallocate memory
    HOST DEVICE void deallocate() {
        if (data) {
            #ifdef HAS_CUDA
            if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
                cudaFree(data);
            } else
            #endif
            {
                free(data);
            }
            data = nullptr;
            size = 0;
            capacity = 0;
        }
    }

    // Clear the vector
    HOST DEVICE void clear() {
        deallocate();
        allocate(initial_capacity);
        size = 0;
    }

    // Assign value to all elements
    HOST DEVICE void assign(const T& value) {
        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            if (std::is_trivially_copyable<T>::value) {
                // Launch a CUDA kernel to fill the array
                fill_kernel<<<(size + 255)/256, 256>>>(data, value, size);
            } else {
                throw std::runtime_error("Non-trivial type assignment requires custom kernel");
            }
        } else
        #endif
        {
            std::fill(data, data + size, value);
        }
    }

    // Resize the vector
    HOST DEVICE void resize(size_t new_size) {
        if (new_size == size) return;

        if (new_size <= capacity) {
            size = new_size;
            return;
        }

        T* new_data = nullptr;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaError_t err = cudaMalloc(&new_data, new_size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed during resize!");
            }

            if (data && size > 0) {
                cudaMemcpy(new_data, data, size * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        } else
        #endif
        {
            new_data = static_cast<T*>(malloc(new_size * sizeof(T)));
            if (!new_data) {
                throw std::runtime_error("Host malloc failed during resize!");
            }

            if (data && size > 0) {
                std::copy(data, data + size, new_data);
            }
        }

        // Only deallocate after successful allocation
        deallocate();
        data = new_data;
        size = new_size;
        capacity = new_size;
    }

    // Add element to the end
    HOST DEVICE void push_back(const T& value) {
        if (size >= capacity) {
            const size_t new_capacity = (capacity == 0) ? initial_capacity : capacity * 2;
            resize(new_capacity);
        }
        data[size++] = value;
    }

    // Write from pointer
    HOST DEVICE void write_from_pointer(const T* ptr, size_t data_size) {
        if (!ptr) {
            throw std::invalid_argument("Null pointer in write_from_pointer");
        }

        resize(data_size);

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaMemcpy(data, ptr, data_size * sizeof(T), cudaMemcpyHostToDevice);
        } else
        #endif
        {
            std::copy(ptr, ptr + data_size, data);
        }
    }

    // Array access operators
    HOST DEVICE T& operator[](size_t i) {
        if (i >= size) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i];
    }

    HOST DEVICE const T& operator[](size_t i) const {
        if (i >= size) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i];
    }

    // Utility methods
    HOST DEVICE [[nodiscard]] bool empty() const { return size == 0; }
    HOST DEVICE [[nodiscard]] size_t get_size() const { return size; }
    HOST DEVICE [[nodiscard]] size_t get_capacity() const { return capacity; }
};

#endif // COMMON_VECTOR_H
