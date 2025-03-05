#ifndef I_NODEMAPPING_H
#define I_NODEMAPPING_H

#include "../../data/enums.h"
#include "../../cuda_common/types.cuh"

#include "../interfaces/IEdgeData.cuh"

template<GPUUsageMode GPUUsage>
class INodeMapping {

protected:
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using BoolVector = typename SelectVectorType<bool, GPUUsage>::type;

public:
    virtual ~INodeMapping() = default;

    IntVector sparse_to_dense{};    // Maps sparse ID to dense index
    IntVector dense_to_sparse{};    // Maps dense index back to sparse ID

    BoolVector is_deleted{};        // Tracks deleted status of nodes

    /**
    * HOST METHODS
    */
    virtual HOST void update(const IEdgeData<GPUUsage>* edges, size_t start_idx, size_t end_idx) {}
    [[nodiscard]] virtual HOST int to_dense(int sparse_id) const;
    [[nodiscard]] virtual HOST int to_sparse(int dense_idx) const;
    [[nodiscard]] virtual HOST size_t size() const;
    [[nodiscard]] virtual HOST size_t active_size() const;

    // Helper methods
    [[nodiscard]] virtual HOST IntVector get_active_node_ids() const;
    virtual HOST void clear();
    virtual HOST void reserve(size_t size);
    virtual HOST void mark_node_deleted(int sparse_id);
    [[nodiscard]] virtual HOST bool has_node(int sparse_id) const;
    [[nodiscard]] virtual HOST IntVector get_all_sparse_ids() const;
};

#endif //I_NODEMAPPING_H