#include "NodeMapping.cuh"
#include <algorithm>
#include <stdexcept>
#include "../cuda/cuda_functions.cuh"


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

void NodeMapping::update(const EdgeData& edges, const size_t start_idx, const size_t end_idx) {
    int max_node_id = cuda_functions::find_max_node_id(edges.sources, edges.targets, start_idx, end_idx, use_gpu);
    // Extend sparse_to_dense if needed
    if (max_node_id >= sparse_to_dense.size()) {
        sparse_to_dense.resize(max_node_id + 1, -1);
        is_deleted.resize(max_node_id + 1, ITEM_DELETED);
    }

    cuda_functions::mark_nodes_not_deleted(
        edges.sources,
        edges.targets,
        is_deleted,
        start_idx,
        end_idx,
        use_gpu,
        ITEM_NOT_DELETED);
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
    return cuda_functions::count_matching(is_deleted, ITEM_NOT_DELETED, use_gpu);
}

std::vector<int> NodeMapping::get_active_node_ids() const {
    return cuda_functions::filter_active_nodes(
        dense_to_sparse,
        is_deleted,
        ITEM_NOT_DELETED,
        use_gpu);
}

bool NodeMapping::has_node(const int sparse_id) const {
    return sparse_id < sparse_to_dense.size() && sparse_to_dense[sparse_id] != -1;
}

std::vector<int> NodeMapping::get_all_sparse_ids() const {
    return dense_to_sparse.to_vector();
}
