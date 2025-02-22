#ifndef COMMON_VECTOR_H
#define COMMON_VECTOR_H

#include <cstring>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <algorithm>
#include "macros.cuh"

template <typename T>
struct DeviceVector {
    T* data;
    size_t data_size;
    size_t capacity;
    size_t initial_capacity = 100;
    T default_value = T();
    mutable bool has_error = false;

    HOST DEVICE DeviceVector()
        : data(nullptr)
          , data_size(0)
          , capacity(0)
    {
        allocate(initial_capacity);
    }

    // Constructor
    HOST DEVICE explicit DeviceVector(const size_t count)
        : data(nullptr)
          , data_size(0)
          , capacity(0)
    {
        // Allocate at least the larger of count or initial_capacity
        const size_t alloc_size = std::max(count, initial_capacity);
        allocate(alloc_size);
        resize(count); // This will fill the elements with fill_value
    }

    // Constructor taking initializer list
    HOST DEVICE DeviceVector(std::initializer_list<T> init)
        : data(nullptr)
        , data_size(0)
        , capacity(0)
    {
        allocate(init.size());

        #ifdef HAS_CUDA
        // For GPU, need to copy through host memory first
        cudaError_t err = cudaMemcpy(data,
                                    init.begin(),
                                    init.size() * sizeof(T),
                                    cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            has_error = true;
            return;
        }
        #endif

        data_size = init.size();
    }

    // Destructor
    HOST DEVICE ~DeviceVector() {
        deallocate();
    }

    // Copy constructor
    HOST DEVICE DeviceVector(const DeviceVector& other)
        : data(nullptr)
        , data_size(0)
        , capacity(0) {
        allocate(other.capacity);
        data_size = other.data_size;

        #ifdef HAS_CUDA
        cudaMemcpy(data, other.data, data_size * sizeof(T), cudaMemcpyDeviceToDevice);
        #endif
    }

    // Move constructor
    HOST DEVICE DeviceVector(DeviceVector&& other) noexcept
        : data(other.data)
        , data_size(other.data_size)
        , capacity(other.capacity) {
        other.data = nullptr;
        other.data_size = 0;
        other.capacity = 0;
    }

    // Copy assignment
    HOST DEVICE DeviceVector& operator=(const DeviceVector& other) {
        if (this != &other) {
            deallocate();
            allocate(other.capacity);
            data_size = other.data_size;

            #ifdef HAS_CUDA
            cudaMemcpy(data, other.data, data_size * sizeof(T), cudaMemcpyDeviceToDevice);
            #endif
        }
        return *this;
    }

    // Move assignment
    HOST DEVICE DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            deallocate();
            data = other.data;
            data_size = other.data_size;
            capacity = other.capacity;

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
        // Allocate new GPU memory
        cudaError_t err = cudaMalloc(&new_data, n * sizeof(T));
        if (err != cudaSuccess) {
            has_error = true;
            return;
        }
        
        // Copy existing data if we have any
        if (data && data_size > 0) {
            err = cudaMemcpy(new_data, data, data_size * sizeof(T), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(new_data);
                has_error = true;
                return;
            }
        }

        // Initialize any extra space with default value
        if (std::is_trivially_copyable_v<T>) {
            cudaMemset(new_data + old_size, 0, (n - old_size) * sizeof(T));
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaFree(new_data);
                has_error = true;
                return;
            }
        } else {
            cudaFree(new_data);
            has_error = true;
            return;
        }
        #endif

        // Free old memory
        if (data)
        {
            #ifdef HAS_CUDA
            cudaFree(data);
            #endif
        }

        // Update pointer and capacity
        data = new_data;
        capacity = n;
    }

    // Deallocate memory
    HOST DEVICE void deallocate() {
        if (data) {
            #ifdef HAS_CUDA
            cudaFree(data);
            #endif
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
        // Allocate new GPU memory
        cudaError_t err = cudaMalloc(&new_data, new_size * sizeof(T));
        if (err != cudaSuccess) {
            has_error = true;
            return;
        }

        // Copy existing data if we have any
        if (data && data_size > 0) {
            // Copy the minimum of old and new size
            size_t copy_size = std::min(data_size, new_size);
            err = cudaMemcpy(new_data, data, copy_size * sizeof(T), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(new_data);  // Clean up on error
                has_error = true;
                return;
            }
        }

        // Initialize any new elements if growing
        if (new_size > data_size) {
            if (std::is_trivially_copyable_v<T>) {
                cudaMemset(new_data + data_size, 0, (new_size - data_size) * sizeof(T));
                if (cudaGetLastError() != cudaSuccess) {
                    cudaFree(new_data);
                    has_error = true;
                    return;
                }
            } else {
                cudaFree(new_data);
                has_error = true;
                return;
            }
        }
        #endif

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
            has_error = true;
            return;
        }

        resize(data_size);

        #ifdef HAS_CUDA
        cudaMemcpy(data, ptr, data_size * sizeof(T), cudaMemcpyHostToDevice);
        #endif
    }

    HOST DEVICE void append_from_pointer(const T* ptr, size_t append_size) {
        if (!ptr) {
            has_error = true;
            return;
        }

        // Calculate new total size needed
        size_t new_size = data_size + append_size;

        // Store old size for later use
        size_t old_size = data_size;

        // Resize to accommodate new elements
        resize(new_size);

        // Copy new elements to the end
        #ifdef HAS_CUDA
        cudaError_t err = cudaMemcpy(
            data + old_size,          // Destination: end of existing data
            ptr,                      // Source: new data
            append_size * sizeof(T),  // Size of new data
            cudaMemcpyHostToDevice
        );
        if (err != cudaSuccess) {
            has_error = true;
        }
        #endif
    }

    // Array access operators
    HOST DEVICE T& operator[](size_t i) {
        if (i >= data_size) {
            has_error = true;
            return default_value;
        }
        return data[i];
    }

    HOST DEVICE const T& operator[](size_t i) const {
        if (i >= data_size) {
            has_error = true;
            return default_value;
        }
        return data[i];
    }

    // Get reference to first element
    HOST DEVICE T& front() {
        if (empty()) {
            has_error = true;
            return default_value;
        }
        return data[0];
    }

    // Get const reference to first element
    HOST DEVICE const T& front() const {
        if (empty()) {
            has_error = true;
            return default_value;
        }
        return data[0];
    }

    // Get reference to last element
    HOST DEVICE T& back() {
        if (empty()) {
            has_error = true;
            return default_value;
        }
        return data[data_size - 1];
    }

    // Get const reference to last element
    HOST DEVICE const T& back() const {
        if (empty()) {
            has_error = true;
            return default_value;
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

    HOST DEVICE void fill(T value) {
        for (size_t i = 0; i < data_size; i++) {
            data[i] = value;
        }
    }

    // Utility methods
    HOST DEVICE [[nodiscard]] bool empty() const { return data_size == 0; }
    HOST DEVICE [[nodiscard]] size_t size() const { return data_size; }
    HOST DEVICE [[nodiscard]] size_t get_capacity() const { return capacity; }
    HOST DEVICE T* data_ptr() { return data; }
};

#endif // COMMON_VECTOR_H
