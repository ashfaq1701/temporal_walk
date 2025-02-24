#ifndef NODEEDGEINDEX_CPU_H
#define NODEEDGEINDEX_CPU_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "../../structs/enums.h"
#include "../interfaces/NodeEdgeIndex.cuh"
#include "../interfaces/NodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCPU : public NodeEdgeIndex<GPUUsage>
{
public:
    ~NodeEdgeIndexCPU() override = default;

    HOST void clear_host() override;
    HOST void rebuild_host(const EdgeData<GPUUsage>& edges, const NodeMapping<GPUUsage>& mapping, bool is_directed) override;

    // Core access methods
    [[nodiscard]] HOST SizeRange get_edge_range_host(int dense_node_id, bool forward, bool is_directed) const override;
    [[nodiscard]] HOST SizeRange get_timestamp_group_range_host(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const override;
    [[nodiscard]] HOST size_t get_timestamp_group_count_host(int dense_node_id, bool forward, bool directed) const override;

    HOST void update_temporal_weights_host(const EdgeData<GPUUsage>& edges, double timescale_bound) override;

protected:
    HOST typename NodeEdgeIndex<GPUUsage>::SizeVector get_timestamp_offset_vector_host(bool forward, bool directed) const override;
};

#endif //NODEEDGEINDEX_CPU_H
