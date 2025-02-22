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
    size_t data_size;
    size_t capacity;
    size_t initial_capacity;
    T default_value;

    T* get_data() const __attribute__((used)) { return data; }
    size_t get_size() const __attribute__((used)) { return data_size; }

    HOST DEVICE CommonVector()
        : data(nullptr)
          , data_size(0)
          , capacity(0)
          , initial_capacity(100) // Default initial capacity
          , default_value(T())
    {
        allocate(initial_capacity);
    }

    // Constructor
    HOST DEVICE explicit CommonVector(size_t count, const T& fill_value = T(), size_t initial_cap = 100)
        : data(nullptr)
          , data_size(0)
          , capacity(0)
          , initial_capacity(initial_cap)
          , default_value(fill_value)
    {
        // Allocate at least the larger of count or initial_capacity
        const size_t alloc_size = std::max(count, initial_cap);
        allocate(alloc_size);
        resize(count, fill_value); // This will fill the elements with fill_value
    }

    // Constructor taking initializer list
    HOST DEVICE CommonVector(std::initializer_list<T> init)
        : data(nullptr)
        , data_size(0)
        , capacity(0)
        , initial_capacity(init.size())
        , default_value(T())
    {
        allocate(init.size());

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // For GPU, need to copy through host memory first
            cudaError_t err = cudaMemcpy(data,
                                       init.begin(),
                                       init.size() * sizeof(T),
                                       cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed in initializer list constructor!");
            }
        } else
        #endif
        {
            // For CPU, direct copy
            std::copy(init.begin(), init.end(), data);
        }
        data_size = init.size();
    }

    // Destructor
    HOST DEVICE ~CommonVector() {
        deallocate();
    }

    // Copy constructor
    HOST DEVICE CommonVector(const CommonVector& other)
        : data(nullptr)
        , data_size(0)
        , capacity(0)
        , initial_capacity(other.initial_capacity)
        , default_value(other.default_value) {
        allocate(other.capacity);
        data_size = other.data_size;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaMemcpy(data, other.data, data_size * sizeof(T), cudaMemcpyDeviceToDevice);
        } else
        #endif
        {
            std::copy(other.data, other.data + data_size, data);
        }
    }

    // Move constructor
    HOST DEVICE CommonVector(CommonVector&& other) noexcept
        : data(other.data)
        , data_size(other.data_size)
        , capacity(other.capacity)
        , initial_capacity(other.initial_capacity)
        , default_value(other.default_value) {
        other.data = nullptr;
        other.data_size = 0;
        other.capacity = 0;
    }

    // Copy assignment
    HOST DEVICE CommonVector& operator=(const CommonVector& other) {
        if (this != &other) {
            deallocate();
            allocate(other.capacity);
            data_size = other.data_size;
            initial_capacity = other.initial_capacity;
            default_value = other.default_value;

            #ifdef HAS_CUDA
            if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
                cudaMemcpy(data, other.data, data_size * sizeof(T), cudaMemcpyDeviceToDevice);
            } else
            #endif
            {
                std::copy(other.data, other.data + data_size, data);
            }
        }
        return *this;
    }

    // Move assignment
    HOST DEVICE CommonVector& operator=(CommonVector&& other) noexcept {
        if (this != &other) {
            deallocate();
            data = other.data;
            data_size = other.data_size;
            capacity = other.capacity;
            initial_capacity = other.initial_capacity;
            default_value = other.default_value;

            other.data = nullptr;
            other.data_size = 0;
            other.capacity = 0;
        }
        return *this;
    }

    // Allocate memory
    HOST DEVICE void allocate(size_t n)
    {
        if (n == 0) return;
        if (n <= capacity) return; // Only grow, never shrink

        T* new_data = nullptr;
        size_t old_size = data_size; // Save current size

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // Allocate new GPU memory
            cudaError_t err = cudaMalloc(&new_data, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed!");
            }

            // Copy existing data if we have any
            if (data && data_size > 0) {
                err = cudaMemcpy(new_data, data, data_size * sizeof(T), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    cudaFree(new_data);
                    throw std::runtime_error("CUDA memcpy failed!");
                }
            }

            // Initialize any extra space with default value
            if (std::is_trivially_copyable<T>::value) {
                fill_kernel<<<(n - old_size + 255)/256, 256>>>(
                    new_data + old_size,
                    default_value,
                    n - old_size
                );
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    cudaFree(new_data);
                    throw std::runtime_error("CUDA fill kernel failed!");
                }
            } else {
                cudaFree(new_data);
                throw std::runtime_error("Non-trivially copyable type requires custom kernel!");
            }
        } else
        #endif
        {
            // Allocate new host memory
            new_data = static_cast<T*>(malloc(n * sizeof(T)));
            if (!new_data)
            {
                throw std::runtime_error("Host malloc failed!");
            }

            // Copy existing data if we have any
            if (data && data_size > 0)
            {
                std::memcpy(new_data, data, data_size * sizeof(T));
            }

            // Initialize extra space with default value
            std::fill(new_data + old_size, new_data + n, default_value);
        }

        // Free old memory
        if (data)
        {
            #ifdef HAS_CUDA
            if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
                cudaFree(data);
            } else
            #endif
            {
                free(data);
            }
        }

        // Update pointer and capacity
        data = new_data;
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
            data_size = 0;
            capacity = 0;
        }
    }

    // Clear the vector
    HOST DEVICE void clear() {
        deallocate();
        allocate(initial_capacity);
    }

    // Assign value to all elements
    HOST DEVICE void assign(const T& value) {
        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            if (std::is_trivially_copyable<T>::value) {
                // Launch a CUDA kernel to fill the array
                fill_kernel<<<(data_size + 255)/256, 256>>>(data, value, data_size);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error("Fill kernel launch failed!");
                }
            } else {
                throw std::runtime_error("Non-trivial type assignment requires custom kernel");
            }
        } else
        #endif
        {
            std::fill(data, data + data_size, value);
        }
    }

    HOST DEVICE void assign(size_t count, const T& fill_value) {
        resize(count);

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            if (std::is_trivially_copyable<T>::value) {
                // Use fill kernel for GPU
                fill_kernel<<<(count + 255)/256, 256>>>(data, fill_value, count);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA fill kernel failed in assign!");
                }
            } else {
                throw std::runtime_error("Non-trivially copyable type assign requires custom kernel!");
            }
        } else
        #endif
        {
            std::fill(data, data + count, fill_value);
        }
    }

    HOST DEVICE void resize(size_t new_size)
    {
        // If size is the same, no action needed
        if (new_size == data_size)
        {
            return;
        }

        // Always allocate new memory of exactly the requested size
        T* new_data = nullptr;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // Allocate new GPU memory
            cudaError_t err = cudaMalloc(&new_data, new_size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed in resize!");
            }

            // Copy existing data if we have any
            if (data && data_size > 0) {
                // Copy the minimum of old and new size
                size_t copy_size = std::min(data_size, new_size);
                err = cudaMemcpy(new_data, data, copy_size * sizeof(T), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    cudaFree(new_data);  // Clean up on error
                    throw std::runtime_error("CUDA memcpy failed in resize!");
                }
            }

            // Initialize any new elements if growing
            if (new_size > data_size) {
                if (std::is_trivially_copyable<T>::value) {
                    fill_kernel<<<(new_size - data_size + 255)/256, 256>>>(
                        new_data + data_size,
                        default_value,
                        new_size - data_size
                    );
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        cudaFree(new_data);
                        throw std::runtime_error("CUDA fill kernel failed in resize!");
                    }
                } else {
                    cudaFree(new_data);
                    throw std::runtime_error("Non-trivially copyable type resize requires custom kernel!");
                }
            }
        } else
        #endif
        {
            // Allocate new host memory
            new_data = static_cast<T*>(malloc(new_size * sizeof(T)));
            if (!new_data)
            {
                throw std::runtime_error("Host malloc failed in resize!");
            }

            // Copy existing data if we have any
            if (data && data_size > 0)
            {
                // Copy the minimum of old and new size
                size_t copy_size = std::min(data_size, new_size);
                std::copy(data, data + copy_size, new_data);
            }

            // Initialize any new elements if growing
            if (new_size > data_size)
            {
                std::fill(new_data + data_size, new_data + new_size, default_value);
            }
        }

        // Free old memory
        deallocate();

        // Update member variables
        data = new_data;
        data_size = new_size;
        capacity = new_size;
    }

    // Resize with fill value
    HOST DEVICE void resize(size_t new_size, const T& fill_value)
    {
        // Remember old size for filling new elements
        size_t old_size = data_size;

        // Do the basic resize
        if (new_size == data_size) return;

        // Always allocate new memory of exactly the requested size
        T* new_data = nullptr;

        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            // Allocate new GPU memory
            cudaError_t err = cudaMalloc(&new_data, new_size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed in resize!");
            }

            // Copy existing data if we have any
            if (data && size > 0) {
                // Copy the minimum of old and new size
                size_t copy_size = std::min(size, new_size);
                err = cudaMemcpy(new_data, data, copy_size * sizeof(T), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    cudaFree(new_data);  // Clean up on error
                    throw std::runtime_error("CUDA memcpy failed in resize!");
                }
            }

            // Fill new elements with provided value
            if (new_size > old_size) {
                if (std::is_trivially_copyable<T>::value) {
                    fill_kernel<<<(new_size - old_size + 255)/256, 256>>>(
                        new_data + old_size,
                        fill_value,
                        new_size - old_size
                    );
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        cudaFree(new_data);
                        throw std::runtime_error("CUDA fill kernel failed in resize!");
                    }
                } else {
                    cudaFree(new_data);
                    throw std::runtime_error("Non-trivially copyable type resize requires custom kernel!");
                }
            }
        } else
        #endif
        {
            // Allocate new host memory
            new_data = static_cast<T*>(malloc(new_size * sizeof(T)));
            if (!new_data)
            {
                throw std::runtime_error("Host malloc failed in resize!");
            }

            // Copy existing data if we have any
            if (data && data_size > 0)
            {
                // Copy the minimum of old and new size
                size_t copy_size = std::min(data_size, new_size);
                std::copy(data, data + copy_size, new_data);
            }

            // Fill new elements with provided value
            if (new_size > old_size)
            {
                std::fill(new_data + old_size, new_data + new_size, fill_value);
            }
        }

        // Free old memory
        deallocate();

        // Update member variables
        data = new_data;
        data_size = new_size;
        capacity = new_size;
    }

    // Add element to the end
    HOST DEVICE void push_back(const T& value) {
        if (data_size >= capacity) {
            const size_t new_capacity = (capacity == 0) ? initial_capacity : capacity * 2;
            allocate(new_capacity);
        }
        data[data_size++] = value;
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

    HOST DEVICE void append_from_pointer(const T* ptr, size_t append_size) {
        if (!ptr) {
            throw std::invalid_argument("Null pointer in append_from_pointer");
        }

        // Calculate new total size needed
        size_t new_size = data_size + append_size;

        // Store old size for later use
        size_t old_size = data_size;

        // Resize to accommodate new elements
        resize(new_size);

        // Copy new elements to the end
        #ifdef HAS_CUDA
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            cudaError_t err = cudaMemcpy(
                data + old_size,          // Destination: end of existing data
                ptr,                      // Source: new data
                append_size * sizeof(T),  // Size of new data
                cudaMemcpyHostToDevice
            );
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed in append_from_pointer!");
            }
        } else
        #endif
        {
            std::copy(ptr, ptr + append_size, data + old_size);
        }
    }

    // Array access operators
    HOST DEVICE T& operator[](size_t i) {
        if (i >= data_size) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i];
    }

    HOST DEVICE const T& operator[](size_t i) const {
        if (i >= data_size) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[i];
    }

    // Get reference to first element
    HOST DEVICE T& front() {
        if (empty()) {
            throw std::out_of_range("Cannot access front() of empty vector");
        }
        return data[0];
    }

    // Get const reference to first element
    HOST DEVICE const T& front() const {
        if (empty()) {
            throw std::out_of_range("Cannot access front() of empty vector");
        }
        return data[0];
    }

    // Get reference to last element
    HOST DEVICE T& back() {
        if (empty()) {
            throw std::out_of_range("Cannot access back() of empty vector");
        }
        return data[data_size - 1];
    }

    // Get const reference to last element
    HOST DEVICE const T& back() const {
        if (empty()) {
            throw std::out_of_range("Cannot access back() of empty vector");
        }
        return data[data_size - 1];
    }

    // Get pointer to beginning of data
    HOST DEVICE T* begin() {
        return data;
    }

    // Get const pointer to beginning of data
    HOST DEVICE const T* begin() const {
        return data;
    }

    // Get pointer to end of data (one past last element)
    HOST DEVICE T* end() {
        return data + data_size;
    }

    // Get const pointer to end of data
    HOST DEVICE const T* end() const {
        return data + data_size;
    }

    // Utility methods
    HOST DEVICE [[nodiscard]] bool empty() const { return data_size == 0; }
    HOST DEVICE [[nodiscard]] size_t size() const { return data_size; }
    HOST DEVICE [[nodiscard]] size_t get_capacity() const { return capacity; }
    HOST DEVICE T* data_ptr() { return data; }
};

#endif // COMMON_VECTOR_H
