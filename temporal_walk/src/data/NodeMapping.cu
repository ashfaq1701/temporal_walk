#include "NodeMapping.cuh"
#include <algorithm>
#include <stdexcept>
#include <thrust/extrema.h>

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

void NodeMapping::host_mark_node_deleted(const int sparse_id) {
    if (sparse_id < is_deleted.size()) {
        is_deleted[sparse_id] = ITEM_DELETED;
    }
}

void NodeMapping::update(const EdgeData &edges, const size_t start_idx, const size_t end_idx) {
    // First pass: find max node ID
    int max_node_id = 0;
    if (use_gpu) {
        #ifdef HAS_CUDA
        // Use thrust::max_element directly on the edge ranges
        const auto max_src = thrust::max_element(
            thrust::device,
            edges.sources.device_begin() + static_cast<int>(start_idx),
            edges.sources.device_begin() + static_cast<int>(end_idx)
        );
        const auto max_tgt = thrust::max_element(
            thrust::device,
            edges.targets.device_begin() + static_cast<int>(start_idx),
            edges.targets.device_begin() + static_cast<int>(end_idx)
        );

        max_node_id = std::max(
            *max_src,
            *max_tgt
        );
        #endif
    } else {
        for (size_t i = start_idx; i < end_idx; i++) {
            max_node_id = std::max({
                max_node_id,
                edges.sources.host_at(i),
                edges.targets.host_at(i)
            });
        }
    }

    // Extend sparse_to_dense if needed
    if (max_node_id >= sparse_to_dense.size()) {
        sparse_to_dense.resize(max_node_id + 1, -1);
        is_deleted.resize(max_node_id + 1, ITEM_DELETED);
    }

    // Map unmapped nodes
    if (use_gpu) {
        #ifdef HAS_CUDA
        // Get source and target node IDs in parallel
        const int *sources_ptr = edges.sources.device_data().get();
        const int *targets_ptr = edges.targets.device_data().get();
        int *s2d_ptr = sparse_to_dense.device_data().get();

        // Create temporary storage for nodes needing mapping
        thrust::device_vector<int> new_nodes(1);  // Start with size 1 for the counter
        thrust::fill(new_nodes.begin(), new_nodes.end(), 0);  // Initialize counter to 0
        new_nodes.resize(1 + 2 * (end_idx - start_idx));  // Space for counter + max possible new nodes

        // First identify and collect nodes needing mapping
        thrust::for_each(
            thrust::device,
            thrust::counting_iterator<size_t>(start_idx),
            thrust::counting_iterator<size_t>(end_idx),
            [sources = sources_ptr,
                targets = targets_ptr,
                s2d = s2d_ptr,
                d2s_size = dense_to_sparse.size(),
                new_nodes_ptr = thrust::raw_pointer_cast(new_nodes.data())] __device__ (size_t i) {
                const int src = sources[i];
                const int tgt = targets[i];

                // Use atomic operation to ensure thread safety
                if (atomicCAS(&s2d[src], -1, static_cast<int>(d2s_size)) == -1) {
                    // If this thread won the race to map this node, add it to new nodes
                    const size_t pos = atomicAdd(&new_nodes_ptr[0], 1) + 1;  // +1 to skip counter
                    new_nodes_ptr[pos] = src;
                }
                if (atomicCAS(&s2d[tgt], -1, static_cast<int>(d2s_size)) == -1) {
                    const size_t pos = atomicAdd(&new_nodes_ptr[0], 1) + 1;  // +1 to skip counter
                    new_nodes_ptr[pos] = tgt;
                }
            }
        );

        // Copy new nodes to host and update dense_to_sparse
        std::vector<int> h_new_nodes(new_nodes.size());
        thrust::copy(new_nodes.begin(), new_nodes.end(), h_new_nodes.begin());

        // Update dense_to_sparse sequentially on host
        for (int node: h_new_nodes) {
            dense_to_sparse.push_back(node);
        }
        #endif
    } else {
        // CPU version
        for (size_t i = start_idx; i < end_idx; i++) {
            const int src = edges.sources.host_at(i);
            const int tgt = edges.targets.host_at(i);

            if (sparse_to_dense[src] == -1) {
                sparse_to_dense[src] = static_cast<int>(dense_to_sparse.size());
                dense_to_sparse.push_back(src);
            }
            if (sparse_to_dense[tgt] == -1) {
                sparse_to_dense[tgt] = static_cast<int>(dense_to_sparse.size());
                dense_to_sparse.push_back(tgt);
            }
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
