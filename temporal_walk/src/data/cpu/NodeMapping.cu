#include "NodeMapping.cuh"
#include <algorithm>

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::clear() {
    sparse_to_dense.clear();
    dense_to_sparse.clear();
    is_deleted.clear();
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::reserve(const size_t size) {
    sparse_to_dense.reserve(size);
    dense_to_sparse.reserve(size);
    is_deleted.reserve(size);
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::mark_node_deleted(const int sparse_id) {
    if (sparse_id < is_deleted.size()) {
        is_deleted[sparse_id] = true;
    }
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::update(const EdgeData<GPUUsage>& edges, const size_t start_idx, const size_t end_idx) {
    // First pass: find max node ID
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({
            max_node_id,
            static_cast<int>(edges.sources[i]),
            static_cast<int>(edges.targets[i])
        });
    }

    // Extend sparse_to_dense if needed
    if (max_node_id >= sparse_to_dense.size()) {
        sparse_to_dense.resize(max_node_id + 1, -1);
        is_deleted.resize(max_node_id + 1, true);
    }

    // Map unmapped nodes
    for (size_t i = start_idx; i < end_idx; i++) {
        is_deleted[edges.sources[i]] = false;
        is_deleted[edges.targets[i]] = false;

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

template<GPUUsageMode GPUUsage>
int NodeMapping<GPUUsage>::to_dense(const int sparse_id) const {
    return sparse_id < sparse_to_dense.size() ? sparse_to_dense[sparse_id] : -1;
}

template<GPUUsageMode GPUUsage>
int NodeMapping<GPUUsage>::to_sparse(const int dense_idx) const {
    return dense_idx < dense_to_sparse.size() ? dense_to_sparse[dense_idx] : -1;
}

template<GPUUsageMode GPUUsage>
size_t NodeMapping<GPUUsage>::size() const {
    return dense_to_sparse.size();
}

template<GPUUsageMode GPUUsage>
size_t NodeMapping<GPUUsage>::active_size() const {
    return std::count(is_deleted.begin(), is_deleted.end(), false);
}

template<GPUUsageMode GPUUsage>
std::vector<int> NodeMapping<GPUUsage>::get_active_node_ids() const {
    std::vector<int> active_ids;
    active_ids.reserve(dense_to_sparse.size());
    for (int sparse_id : dense_to_sparse) {
        if (!is_deleted[sparse_id]) {
            active_ids.push_back(sparse_id);
        }
    }
    return active_ids;
}

template<GPUUsageMode GPUUsage>
bool NodeMapping<GPUUsage>::has_node(int sparse_id) const {
    return sparse_id < sparse_to_dense.size() && sparse_to_dense[sparse_id] != -1;
}

template<GPUUsageMode GPUUsage>
typename NodeMapping<GPUUsage>::IntVector NodeMapping<GPUUsage>::get_all_sparse_ids() const {
    return dense_to_sparse;
}

template class NodeMapping<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeMapping<GPUUsageMode::DATA_ON_GPU>;
template class NodeMapping<GPUUsageMode::DATA_ON_HOST>;
#endif
