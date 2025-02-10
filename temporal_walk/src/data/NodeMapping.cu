#include "NodeMapping.cuh"
#include <algorithm>

NodeMapping::NodeMapping(bool use_gpu): use_gpu(use_gpu) {
    sparse_to_dense = VectorTypes<int>::select(use_gpu);
    dense_to_sparse = VectorTypes<int>::select(use_gpu);
    is_deleted = VectorTypes<bool>::select(use_gpu);
}

void NodeMapping::clear() {
    std::visit([&](auto& sparse_to_dense_vec, auto& dense_to_sparse_vec, auto& is_deleted_vec)
    {
        sparse_to_dense_vec.clear();
        dense_to_sparse_vec.clear();
        is_deleted_vec.clear();
    }, sparse_to_dense, dense_to_sparse, is_deleted);
}

void NodeMapping::reserve(const size_t size) {
    std::visit([&](auto& sparse_to_dense_vec, auto& dense_to_sparse_vec, auto& is_deleted_vec)
    {
        sparse_to_dense_vec.reserve(size);
        dense_to_sparse_vec.reserve(size);
        is_deleted_vec.reserve(size);
    }, sparse_to_dense, dense_to_sparse, is_deleted);
}

void NodeMapping::mark_node_deleted(const int sparse_id) {
    return std::visit([&](auto& is_deleted_vec)
    {
        if (sparse_id < is_deleted_vec.size())
        {
            is_deleted_vec[sparse_id] = true;
        }
    }, is_deleted);
}

void NodeMapping::update(const EdgeData& edges, const size_t start_idx, const size_t end_idx) {
    return std::visit([&](const auto& sources_vec, const auto& targets_vec, auto& sparse_to_dense_vec, auto& dense_to_sparse_vec, auto& is_deleted_vec)
    {
        // First pass: find max node ID
        int max_node_id = 0;
        for (size_t i = start_idx; i < end_idx; i++) {
            max_node_id = std::max({max_node_id, sources_vec[i], targets_vec[i]});
        }

        // Extend sparse_to_dense if needed
        if (max_node_id >= sparse_to_dense_vec.size()) {
            sparse_to_dense_vec.resize(max_node_id + 1, -1);
            is_deleted_vec.resize(max_node_id + 1, true);
        }

        // Map unmapped nodes
        for (size_t i = start_idx; i < end_idx; i++) {
            is_deleted_vec[sources_vec[i]] = false;
            is_deleted_vec[targets_vec[i]] = false;

            if (sparse_to_dense_vec[sources_vec[i]] == -1) {
                sparse_to_dense_vec[sources_vec[i]] = static_cast<int>(dense_to_sparse_vec.size());
                dense_to_sparse_vec.push_back(sources_vec[i]);
            }
            if (sparse_to_dense_vec[targets_vec[i]] == -1) {
                sparse_to_dense_vec[targets_vec[i]] = static_cast<int>(dense_to_sparse_vec.size());
                dense_to_sparse_vec.push_back(targets_vec[i]);
            }
        }
    }, edges.sources, edges.targets, sparse_to_dense, dense_to_sparse, is_deleted);
}

int NodeMapping::to_dense(const int sparse_id) const {
    return std::visit([&](const auto& sparse_to_dense_vec)
    {
        return sparse_id < sparse_to_dense_vec.size() ? sparse_to_dense_vec[sparse_id] : -1;
    }, sparse_to_dense);
}

int NodeMapping::to_sparse(const int dense_idx) const {
    return std::visit([&](const auto& dense_to_sparse_vec)
    {
        return dense_idx < dense_to_sparse_vec.size() ? dense_to_sparse_vec[dense_idx] : -1;
    }, dense_to_sparse);
}

size_t NodeMapping::size() const {
    return std::visit([&](const auto& dense_to_sparse_vec)
    {
        return dense_to_sparse_vec.size();
    }, dense_to_sparse);
}

size_t NodeMapping::active_size() const {
    return std::visit([&](const auto& is_deleted_vec)
    {
        return std::count(is_deleted_vec.begin(), is_deleted_vec.end(), false);
    }, is_deleted);
}

std::vector<int> NodeMapping::get_active_node_ids() const {
    return std::visit([&](const auto& dense_to_sparse_vec, const auto& is_deleted_vec)
    {
        std::vector<int> active_ids;
        active_ids.reserve(dense_to_sparse_vec.size());
        for (int sparse_id : dense_to_sparse_vec) {
            if (!is_deleted_vec[sparse_id]) {
                active_ids.push_back(sparse_id);
            }
        }
        return active_ids;
    }, dense_to_sparse, is_deleted);
}

bool NodeMapping::has_node(int sparse_id) const
{
    return std::visit([&](auto& sparse_to_dense_vec)
    {
        return sparse_id < sparse_to_dense_vec.size() && sparse_to_dense_vec[sparse_id] != -1;
    }, sparse_to_dense);
}

std::vector<int> NodeMapping::get_all_sparse_ids() const
{
    return std::visit([&](auto& dense_to_sparse_vec) { return dense_to_sparse_vec; }, dense_to_sparse);
}
