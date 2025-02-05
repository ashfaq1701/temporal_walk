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
    DualVector(DualVector&& other) noexcept;
    DualVector& operator=(DualVector&& other) noexcept;

    DualVector& operator=(const DualVector& other);

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

    // Setters for vector assignment
    void set_host_vector(const std::vector<T>& vec);
    void set_host_vector(std::vector<T>&& vec);  // Move version

    // Device operations
    #ifdef HAS_CUDA
    device_iterator device_begin();
    device_const_iterator device_begin() const;
    device_iterator device_end();
    device_const_iterator device_end() const;

    T device_at(size_t i) const;
    void device_at_set(size_t i, const T& value);

    T& device_back();
    const T& device_back() const;

    thrust::device_vector<T>& get_device_vector();
    const thrust::device_vector<T>& get_device_vector() const;

    thrust::device_ptr<T> device_data();
    thrust::device_ptr<const T> device_data() const;

    void set_device_vector(const thrust::device_vector<T>& vec);
    void set_device_vector(thrust::device_vector<T>&& vec);  // Move version
    #endif

    #ifdef HAS_CUDA
    // Device-side direct access (for use in kernels)
    __device__ T& device_at_kernel(size_t i) {
        return thrust::raw_pointer_cast(d_vec.data())[i];
    }

    __device__ const T& device_at_kernel(size_t i) const {
        return thrust::raw_pointer_cast(d_vec.data())[i];
    }

    // Get raw device pointer (for use in kernels)
    __device__ T* device_ptr_kernel() {
        return thrust::raw_pointer_cast(d_vec.data());
    }

    __device__ const T* device_ptr_kernel() const {
        return thrust::raw_pointer_cast(d_vec.data());
    }

    class DeviceElementProxy {
    private:
        DualVector<T>& vec;
        size_t index;
        bool is_gpu;

    public:
        DeviceElementProxy(DualVector<T>& v, const size_t i) : vec(v), index(i), is_gpu(v.is_gpu()) {}

        operator T() const {
            return is_gpu ? vec.device_at(index) : vec.host_at(index);
        }

        DeviceElementProxy& operator=(const T& value) {
            if (is_gpu) {
                vec.device_at_set(index, value);
            } else {
                vec.host_at(index) = value;
            }
            return *this;
        }

        DeviceElementProxy& operator/=(const T& value) {
            if (is_gpu) {
                T current = vec.device_at(index);
                vec.device_at_set(index, current / value);
            } else {
                vec.host_at(index) /= value;
            }
            return *this;
        }

        DeviceElementProxy& operator+=(const T& value) {
            if (is_gpu) {
                T current = vec.device_at(index);
                vec.device_at_set(index, current + value);
            } else {
                vec.host_at(index) += value;
            }
            return *this;
        }

        DeviceElementProxy& operator++() {
            if (is_gpu) {
                T current = vec.device_at(index);
                vec.device_at_set(index, current + 1);
            } else {
                ++vec.host_at(index);
            }
            return *this;
        }

        T operator++(int) {
            if (is_gpu) {
                T old_value = vec.device_at(index);
                vec.device_at_set(index, old_value + 1);
                return old_value;
            } else {
                return vec.host_at(index)++;
            }
        }
    };
    #endif

    T& back();
    const T& back() const;

    // Default access operators
    #ifdef HAS_CUDA
    DeviceElementProxy operator[](size_t i);
    #else
    T& operator[](size_t i);
    #endif

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

template<typename T>
DualVector<T>::DualVector(const bool use_gpu_param) : use_gpu(use_gpu_param) {
    #ifndef HAS_CUDA
    if (use_gpu) throw std::runtime_error("GPU support not compiled in");
    #endif
}

// Copy constructor
template<typename T>
DualVector<T>::DualVector(const DualVector& other) : use_gpu(other.use_gpu) {
    #ifndef HAS_CUDA
    if (use_gpu) throw std::runtime_error("GPU support not compiled in");
    #endif

    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec = other.d_vec;
        #endif
    } else {
        h_vec = other.h_vec;
    }
}

template<typename T>
DualVector<T>::DualVector(DualVector&& other) noexcept
    : use_gpu(other.use_gpu), h_vec(std::move(other.h_vec))
{
    #ifdef HAS_CUDA
    if (use_gpu) {
        d_vec = std::move(other.d_vec);
    }
    #endif
    other.use_gpu = false;  // Reset other to CPU mode
}

template<typename T>
DualVector<T>& DualVector<T>::operator=(const DualVector& other) {
    if (this != &other) {
        use_gpu = other.use_gpu;
        if (use_gpu) {
            #ifdef HAS_CUDA
            d_vec = other.d_vec;
            #endif
        } else {
            h_vec = other.h_vec;
        }
    }
    return *this;
}

template<typename T>
DualVector<T>& DualVector<T>::operator=(DualVector&& other) noexcept {
    if (this != &other) {
        use_gpu = other.use_gpu;
        h_vec = std::move(other.h_vec);

        #ifdef HAS_CUDA
        if (use_gpu) {
            d_vec = std::move(other.d_vec);
        }
        #endif
        other.use_gpu = false;  // Reset other to CPU mode
    }
    return *this;
}

// Unified operations
template<typename T>
size_t DualVector<T>::size() const {
    return use_gpu ?
        #ifdef HAS_CUDA
        d_vec.size()
        #else
        0
        #endif
        : h_vec.size();
}

template<typename T>
void DualVector<T>::push_back(const T& value) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec.push_back(value);
        #endif
    } else {
        h_vec.push_back(value);
    }
}

template<typename T>
void DualVector<T>::clear() {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec.clear();
        #endif
    } else {
        h_vec.clear();
    }
}

template<typename T>
void DualVector<T>::reserve(size_t n) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec.reserve(n);
        #endif
    } else {
        h_vec.reserve(n);
    }
}

template<typename T>
void DualVector<T>::resize(size_t n) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec.resize(n);
        #endif
    } else {
        h_vec.resize(n);
    }
}

template<typename T>
void DualVector<T>::resize(size_t n, const T& val) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec.resize(n, val);
        #endif
    } else {
        h_vec.resize(n, val);
    }
}

template<typename T>
void DualVector<T>::assign(size_t count, const T& value) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec.assign(count, value);
        #endif
    } else {
        h_vec.assign(count, value);
    }
}

// Host operations
template<typename T>
typename DualVector<T>::host_iterator DualVector<T>::host_begin() {
    if (use_gpu) throw std::runtime_error("Using host iterator in device mode");
    return h_vec.begin();
}

template<typename T>
typename DualVector<T>::host_const_iterator DualVector<T>::host_begin() const {
    if (use_gpu) throw std::runtime_error("Using host iterator in device mode");
    return h_vec.begin();
}

template<typename T>
typename DualVector<T>::host_iterator DualVector<T>::host_end() {
    if (use_gpu) throw std::runtime_error("Using host iterator in device mode");
    return h_vec.end();
}

template<typename T>
typename DualVector<T>::host_const_iterator DualVector<T>::host_end() const {
    if (use_gpu) throw std::runtime_error("Using host iterator in device mode");
    return h_vec.end();
}

template<typename T>
T& DualVector<T>::host_at(size_t i) {
    if (use_gpu) throw std::runtime_error("Using host access in device mode");
    return h_vec[i];
}

template<typename T>
const T& DualVector<T>::host_at(size_t i) const {
    if (use_gpu) throw std::runtime_error("Using host access in device mode");
    return h_vec[i];
}

template<typename T>
T& DualVector<T>::host_back() {
    if (use_gpu) throw std::runtime_error("Using host access in device mode");
    return h_vec.back();
}

template<typename T>
const T& DualVector<T>::host_back() const {
    if (use_gpu) throw std::runtime_error("Using host access in device mode");
    return h_vec.back();
}

// In DualVector.cu
template<typename T>
void DualVector<T>::set_host_vector(const std::vector<T>& vec) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec = vec;  // thrust::device_vector has constructor from std::vector
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    } else {
        h_vec = vec;
    }
}

template<typename T>
void DualVector<T>::set_host_vector(std::vector<T>&& vec) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        d_vec = vec;  // thrust::device_vector will copy from the moved vector
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    } else {
        h_vec = std::move(vec);
    }
}

// Device operations
#ifdef HAS_CUDA
template<typename T>
typename DualVector<T>::device_iterator DualVector<T>::device_begin() {
    if (!use_gpu) throw std::runtime_error("Using device iterator in host mode");
    return d_vec.begin();
}

template<typename T>
typename DualVector<T>::device_const_iterator DualVector<T>::device_begin() const {
    if (!use_gpu) throw std::runtime_error("Using device iterator in host mode");
    return d_vec.begin();
}

template<typename T>
typename DualVector<T>::device_iterator DualVector<T>::device_end() {
    if (!use_gpu) throw std::runtime_error("Using device iterator in host mode");
    return d_vec.end();
}

template<typename T>
typename DualVector<T>::device_const_iterator DualVector<T>::device_end() const {
    if (!use_gpu) throw std::runtime_error("Using device iterator in host mode");
    return d_vec.end();
}

template<typename T>
T DualVector<T>::device_at(size_t i) const {
    if (!use_gpu) throw std::runtime_error("Using device access in host mode");
    T value;
    thrust::copy_n(d_vec.begin() + i, 1, &value);
    return value;
}

// Host-side write access
template<typename T>
void DualVector<T>::device_at_set(size_t i, const T& value) {
    if (!use_gpu) throw std::runtime_error("Using device access in host mode");
    thrust::copy_n(&value, 1, d_vec.begin() + i);
}


template<typename T>
T& DualVector<T>::device_back() {
    if (!use_gpu) throw std::runtime_error("Using device access in host mode");
    return thrust::raw_reference_cast(d_vec.back());
}

template<typename T>
const T& DualVector<T>::device_back() const {
    if (!use_gpu) throw std::runtime_error("Using device access in host mode");
    return thrust::raw_reference_cast(d_vec.back());
}

template<typename T>
thrust::device_vector<T>& DualVector<T>::get_device_vector() {
    if (!use_gpu) throw std::runtime_error("Using device vector in host mode");
    return d_vec;
}

template<typename T>
const thrust::device_vector<T>& DualVector<T>::get_device_vector() const {
    if (!use_gpu) throw std::runtime_error("Using device vector in host mode");
    return d_vec;
}

// Get raw device pointer
template<typename T>
thrust::device_ptr<T> DualVector<T>::device_data() {
    if (!use_gpu) throw std::runtime_error("Using device pointer in host mode");
    return d_vec.data();
}

template<typename T>
thrust::device_ptr<const T> DualVector<T>::device_data() const {
    if (!use_gpu) throw std::runtime_error("Using device pointer in host mode");
    return d_vec.data();
}

template<typename T>
void DualVector<T>::set_device_vector(const thrust::device_vector<T>& vec) {
    if (!use_gpu) throw std::runtime_error("Using device vector in host mode");
    d_vec = vec;
}

template<typename T>
void DualVector<T>::set_device_vector(thrust::device_vector<T>&& vec) {
    if (!use_gpu) throw std::runtime_error("Using device vector in host mode");
    d_vec = std::move(vec);
}
#endif

template<typename T>
T& DualVector<T>::back() {
    if (use_gpu) {
        #ifdef HAS_CUDA
        return device_back();
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }
    return host_back();
}

template<typename T>
const T& DualVector<T>::back() const {
    if (use_gpu) {
        #ifdef HAS_CUDA
        return device_back();
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }
    return host_back();
}

template<typename T>
#ifdef HAS_CUDA
typename DualVector<T>::DeviceElementProxy DualVector<T>::operator[](size_t i) {
    return DeviceElementProxy(*this, i);
}
#else
T& DualVector<T>::operator[](size_t i) {
    return host_at(i);
}
#endif

template<typename T>
const T& DualVector<T>::operator[](const size_t i) const {
    if (use_gpu) {
        #ifdef HAS_CUDA
        thread_local T value;  // Make it thread-safe with thread_local
        value = device_at(i);
        return value;
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }
    return host_at(i);
}

// Data movement
template<typename T>
void DualVector<T>::move(size_t dest_pos, size_t src_pos, size_t count) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        thrust::copy(d_vec.begin() + src_pos, d_vec.begin() + src_pos + count,
                    d_vec.begin() + dest_pos);
        #endif
    } else {
        std::move(h_vec.begin() + src_pos, h_vec.begin() + src_pos + count,
                h_vec.begin() + dest_pos);
    }
}

template<typename T>
std::vector<T> DualVector<T>::to_vector() const {
    if (use_gpu) {
        #ifdef HAS_CUDA
        std::vector<T> result(d_vec.size());
        thrust::copy(d_vec.begin(), d_vec.end(), result.begin());
        return result;
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }
    return h_vec;  // Already a vector, just return a copy
}

#endif //DUALVECTOR_H
