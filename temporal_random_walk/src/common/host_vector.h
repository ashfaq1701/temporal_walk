#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include <cstring>

#include <cstddef>
#include <algorithm>
#include "macros.cuh"

template <typename T>
struct HostVector {
    T* data;
    size_t data_size;
    size_t capacity;
    size_t initial_capacity = 100;
    T default_value = T();
    mutable bool has_error = false;

    T* get_data() const __attribute__((used)) { return data; }
    size_t get_size() const __attribute__((used)) { return data_size; }

    HOST DEVICE HostVector()
        : data(nullptr)
          , data_size(0)
          , capacity(0)
    {
        allocate(initial_capacity);
    }

    // Constructor
    HOST DEVICE explicit HostVector(size_t count)
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
    HOST DEVICE HostVector(std::initializer_list<T> init)
        : data(nullptr)
        , data_size(0)
        , capacity(0)
    {
        allocate(init.size());
        std::copy(init.begin(), init.end(), data);
        data_size = init.size();
    }

    // Destructor
    HOST DEVICE ~HostVector() {
        deallocate();
    }

    // Copy constructor
    HOST DEVICE HostVector(const HostVector& other)
        : data(nullptr)
        , data_size(0)
        , capacity(0) {
        allocate(other.capacity);
        data_size = other.data_size;

        std::copy(other.data, other.data + data_size, data);
    }

    // Move constructor
    HOST DEVICE HostVector(HostVector&& other) noexcept
        : data(other.data)
        , data_size(other.data_size)
        , capacity(other.capacity) {
        other.data = nullptr;
        other.data_size = 0;
        other.capacity = 0;
    }

    // Copy assignment
    HOST DEVICE HostVector& operator=(const HostVector& other) {
        if (this != &other) {
            deallocate();
            allocate(other.capacity);
            data_size = other.data_size;

            std::copy(other.data, other.data + data_size, data);
        }
        return *this;
    }

    // Move assignment
    HOST DEVICE HostVector& operator=(HostVector&& other) noexcept {
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

        // Allocate new host memory
        new_data = static_cast<T*>(malloc(n * sizeof(T)));
        if (!new_data)
        {
            has_error = true;
            return;
        }

        // Copy existing data if we have any
        if (data && data_size > 0)
        {
            std::memcpy(new_data, data, data_size * sizeof(T));
        }

        // Initialize extra space with default value
        std::fill(new_data + old_size, new_data + n, 0);

        // Free old memory
        if (data)
        {
            free(data);
        }

        // Update pointer and capacity
        data = new_data;
        capacity = n;
    }

    // Deallocate memory
    HOST DEVICE void deallocate() {
        if (data) {
            free(data);
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

        // Allocate new host memory
        new_data = static_cast<T*>(malloc(new_size * sizeof(T)));
        if (!new_data)
        {
            has_error = true;
            return;
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
            std::fill(new_data + data_size, new_data + new_size, 0);
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

        // Allocate new host memory
        new_data = static_cast<T*>(malloc(new_size * sizeof(T)));
        if (!new_data)
        {
            has_error = true;
            return;
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

        std::copy(ptr, ptr + data_size, data);
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
        std::copy(ptr, ptr + append_size, data + old_size);
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
        std::fill(data, data + data_size, value);
    }

    // Utility methods
    HOST DEVICE [[nodiscard]] bool empty() const { return data_size == 0; }
    HOST DEVICE [[nodiscard]] size_t size() const { return data_size; }
    HOST DEVICE [[nodiscard]] size_t get_capacity() const { return capacity; }
    HOST DEVICE T* data_ptr() { return data; }
};

#endif // HOST_VECTOR_H