#ifndef NODEMAPPING_H
#define NODEMAPPING_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "EdgeData.h"

struct NodeMapping {
    std::vector<int> sparse_to_dense;    // Maps sparse ID to dense index
    std::vector<int> dense_to_sparse;    // Maps dense index back to sparse ID

    void update(const EdgeData& edges, size_t start_idx, size_t end_idx);
    int to_dense(int sparse_id) const;
    int to_sparse(int dense_idx) const;
    size_t size() const;
};

#endif //NODEMAPPING_H
