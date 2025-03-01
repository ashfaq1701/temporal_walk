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
    [[nodiscard]] virtual HOST int to_dense_host(int sparse_id) const { return -1; }
    [[nodiscard]] virtual HOST int to_sparse_host(int dense_idx) const { return -1; }
    [[nodiscard]] virtual HOST size_t size_host() const { return 0; }
    [[nodiscard]] virtual HOST size_t active_size_host() const { return 0; }

    // Helper methods
    [[nodiscard]] virtual HOST IntVector get_active_node_ids_host() const { return IntVector(); }
    virtual HOST void clear_host() {}
    virtual HOST void reserve_host(size_t size) {}
    virtual HOST void mark_node_deleted_host(int sparse_id) {}
    [[nodiscard]] virtual HOST bool has_node_host(int sparse_id) const { return false; }
    [[nodiscard]] virtual HOST IntVector get_all_sparse_ids_host() const { return IntVector(); }

    /**
    * DEVICE METHODS
    */
    [[nodiscard]] virtual DEVICE int to_dense_device(int sparse_id) const { return -1; }
    [[nodiscard]] virtual DEVICE int to_sparse_device(int dense_idx) const { return -1; }
    [[nodiscard]] virtual DEVICE size_t size_device() const { return 0; }
    [[nodiscard]] virtual DEVICE size_t active_size_device() const { return 0; }

    // Helper methods
    [[nodiscard]] virtual DEVICE IntVector get_active_node_ids_device() const { return IntVector(); }
    virtual DEVICE void clear_device() {}
    virtual DEVICE void reserve_device(size_t size) {}
    virtual DEVICE void mark_node_deleted_device(int sparse_id) {}
    [[nodiscard]] virtual DEVICE bool has_node_device(int sparse_id) const { return false; }
    [[nodiscard]] virtual DEVICE IntVector get_all_sparse_ids_device() const { return IntVector(); }
};

#endif //I_NODEMAPPING_H