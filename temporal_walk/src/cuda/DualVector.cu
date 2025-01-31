#include "DualVector.cuh"

#include <stdexcept>

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
T& DualVector<T>::device_at(size_t i) {
    if (!use_gpu) throw std::runtime_error("Using device access in host mode");
    return thrust::raw_reference_cast(d_vec[i]);
}

template<typename T>
const T& DualVector<T>::device_at(size_t i) const {
    if (!use_gpu) throw std::runtime_error("Using device access in host mode");
    return thrust::raw_reference_cast(d_vec[i]);
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

// Default access operators
template<typename T>
T& DualVector<T>::operator[](size_t i) {
    if (use_gpu) {
        #ifdef HAS_CUDA
        return device_at(i);
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }
    return host_at(i);
}

template<typename T>
const T& DualVector<T>::operator[](size_t i) const {
    if (use_gpu) {
        #ifdef HAS_CUDA
        return device_at(i);
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
