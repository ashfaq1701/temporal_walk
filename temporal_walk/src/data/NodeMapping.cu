#include "NodeMapping.cuh"
#include <algorithm>
#include <stdexcept>

constexpr short ITEM_DELETED = 1;
constexpr short ITEM_NOT_DELETED = 1;

NodeMapping::NodeMapping(const bool use_gpu):
    use_gpu(use_gpu), sparse_to_dense(use_gpu), dense_to_sparse(use_gpu), is_deleted(use_gpu) {}


void NodeMapping::clear() {
    sparse_to_dense.clear();
    dense_to_sparse.clear();
    is_deleted.clear();
}

void NodeMapping::reserve(const size_t size) {
    sparse_to_dense.reserve(size);
    dense_to_sparse.reserve(size);
    is_deleted.reserve(size);
}

void NodeMapping::mark_node_deleted(const int sparse_id) {
    if (sparse_id < is_deleted.size()) {
        is_deleted[sparse_id] = ITEM_DELETED;
    }
}

void NodeMapping::update(const EdgeData& edges, const size_t start_idx, const size_t end_idx) {
    // First pass: find max node ID
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({max_node_id, edges.sources[i], edges.targets[i]});
    }

    // Extend sparse_to_dense if needed
    if (max_node_id >= sparse_to_dense.size()) {
        sparse_to_dense.resize(max_node_id + 1, -1);
        is_deleted.resize(max_node_id + 1, ITEM_DELETED);
    }

    // Map unmapped nodes
    for (size_t i = start_idx; i < end_idx; i++) {
        is_deleted[edges.sources[i]] = ITEM_NOT_DELETED;
        is_deleted[edges.targets[i]] = ITEM_NOT_DELETED;

        if (sparse_to_dense[edges.sources[i]] == -1) {
            sparse_to_dense[edges.sources[i]] = static_cast<int>(dense_to_sparse.size());
            dense_to_sparse.push_back(edges.sources[i]);
        }
        if (sparse_to_dense[edges.targets[i]] == -1) {
            sparse_to_dense[edges.targets[i]] = static_cast<int>(dense_to_sparse.size());
            dense_to_sparse.push_back(edges.targets[i]);
        }
    }
}

int NodeMapping::to_dense(const int sparse_id) const {
    return sparse_id < sparse_to_dense.size() ? sparse_to_dense[sparse_id] : -1;
}

int NodeMapping::to_sparse(const int dense_idx) const {
    return dense_idx < dense_to_sparse.size() ? dense_to_sparse[dense_idx] : -1;
}

size_t NodeMapping::size() const {
    return dense_to_sparse.size();
}

size_t NodeMapping::active_size() const {
    if (is_deleted.is_gpu()) {
        #ifdef HAS_CUDA
        // Count the number of non-deleted items (zeros)
        return thrust::count(
            thrust::device,
            is_deleted.device_begin(),
            is_deleted.device_end(),
            ITEM_NOT_DELETED
        );
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }
    return std::count(is_deleted.host_begin(), is_deleted.host_end(), ITEM_NOT_DELETED);
}

std::vector<int> NodeMapping::get_active_node_ids() const {
    if (dense_to_sparse.is_gpu()) {
        #ifdef HAS_CUDA
        // Create device vector for results
        thrust::device_vector<int> d_result(dense_to_sparse.size());

        // Use copy_if directly with device vectors
        const auto end = thrust::copy_if(
            thrust::device,
            dense_to_sparse.device_begin(),
            dense_to_sparse.device_end(),
            d_result.begin(),
            [is_deleted = is_deleted.get_device_vector().data()] __device__ (int sparse_id) {
                return is_deleted[sparse_id] == ITEM_NOT_DELETED;
            }
        );

        // Copy results back to host
        std::vector<int> active_ids(thrust::distance(d_result.begin(), end));
        thrust::copy(d_result.begin(), end, active_ids.begin());
        return active_ids;
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    }

    // CPU version using explicit iterators
    std::vector<int> active_ids;
    active_ids.reserve(dense_to_sparse.size());
    const auto begin = dense_to_sparse.host_begin();
    const auto end = dense_to_sparse.host_end();
    for (auto it = begin; it != end; ++it) {
        if (int sparse_id = *it; is_deleted[sparse_id] == ITEM_NOT_DELETED) {
            active_ids.push_back(sparse_id);
        }
    }
    return active_ids;
}

bool NodeMapping::has_node(const int sparse_id) const {
    return sparse_id < sparse_to_dense.size() && sparse_to_dense[sparse_id] != -1;
}

std::vector<int> NodeMapping::get_all_sparse_ids() const {
    return dense_to_sparse.to_vector();
}
