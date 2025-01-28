#ifndef DUALVECTOR_H
#define DUALVECTOR_H

#include <vector>
#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#endif

template<typename T>
class DualVector {
public:
    // Type definitions
    using host_iterator = typename std::vector<T>::iterator;
    using host_const_iterator = typename std::vector<T>::const_iterator;
    #ifdef HAS_CUDA
    using device_iterator = typename thrust::device_vector<T>::iterator;
    using device_const_iterator = typename thrust::device_vector<T>::const_iterator;
    #endif

    // Constructor
    explicit DualVector(bool use_gpu_param);

    // Copy constructor
    DualVector(const DualVector& other);

    // Move semantics
    DualVector(DualVector&& other) noexcept = default;
    DualVector& operator=(DualVector&& other) noexcept = default;

    // Unified operations
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const { return size() == 0; }
    void push_back(const T& value);
    void clear();
    void reserve(size_t n);
    void resize(size_t n);
    void resize(size_t n, const T& val);
    void assign(size_t count, const T& value);
    [[nodiscard]] bool is_gpu() const { return use_gpu; }

    // Host operations
    host_iterator host_begin();
    host_const_iterator host_begin() const;
    host_iterator host_end();
    host_const_iterator host_end() const;
    T& host_at(size_t i);
    const T& host_at(size_t i) const;
    T& host_back();
    const T& host_back() const;

    // Device operations
    #ifdef HAS_CUDA
    device_iterator device_begin();
    device_const_iterator device_begin() const;
    device_iterator device_end();
    device_const_iterator device_end() const;
    T& device_at(size_t i);
    const T& device_at(size_t i) const;
    T& device_back();
    const T& device_back() const;
    #endif

    // Default access operators
    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    // Data movement
    void move(size_t dest_pos, size_t src_pos, size_t count);
    std::vector<T> to_vector() const;

private:
    bool use_gpu;
    std::vector<T> h_vec;
    #ifdef HAS_CUDA
    thrust::device_vector<T> d_vec;
    #endif
};

#endif //DUALVECTOR_H
