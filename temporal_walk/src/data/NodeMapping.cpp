#include "NodeMapping.h"

void NodeMapping::update(const EdgeData& edges, size_t start_idx, size_t end_idx) {
    // First pass: find max node ID
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({max_node_id, edges.sources[i], edges.targets[i]});
    }

    // Extend sparse_to_dense if needed
    if (max_node_id >= sparse_to_dense.size()) {
        sparse_to_dense.resize(max_node_id + 1, -1);
    }

    // Map unmapped nodes
    for (size_t i = start_idx; i < end_idx; i++) {
        if (sparse_to_dense[edges.sources[i]] == -1) {
            sparse_to_dense[edges.sources[i]] = dense_to_sparse.size();
            dense_to_sparse.push_back(edges.sources[i]);
        }
        if (sparse_to_dense[edges.targets[i]] == -1) {
            sparse_to_dense[edges.targets[i]] = dense_to_sparse.size();
            dense_to_sparse.push_back(edges.targets[i]);
        }
    }
}

int NodeMapping::to_dense(int sparse_id) const {
    return sparse_id < sparse_to_dense.size() ? sparse_to_dense[sparse_id] : -1;
}

int NodeMapping::to_sparse(int dense_idx) const {
    return dense_idx < dense_to_sparse.size() ? dense_to_sparse[dense_idx] : -1;
}

size_t NodeMapping::size() const {
    return dense_to_sparse.size();
}