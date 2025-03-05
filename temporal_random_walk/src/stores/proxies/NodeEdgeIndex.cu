#include "NodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
NodeEdgeIndex<GPUUsage>::NodeEdgeIndex(): node_edge_index(new BaseType()) {}

template<GPUUsageMode GPUUsage>
NodeEdgeIndex<GPUUsage>::NodeEdgeIndex(INodeEdgeIndex<GPUUsage>* node_edge_index): node_edge_index(node_edge_index) {}

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::clear()
{
    node_edge_index->clear();
}

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed)
{
    node_edge_index->rebuild(edges, mapping, is_directed);
}

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> NodeEdgeIndex<GPUUsage>::get_edge_range(int dense_node_id, bool forward, bool is_directed) const
{
    auto range = node_edge_index->get_edge_range(dense_node_id, forward, is_directed);
    return {range.from, range.to};
}

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> NodeEdgeIndex<GPUUsage>::get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const
{
    auto range = node_edge_index->get_timestamp_group_range(dense_node_id, group_idx, forward, is_directed);
    return {range.from, range.to};
}

template<GPUUsageMode GPUUsage>
size_t NodeEdgeIndex<GPUUsage>::get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const
{
    return node_edge_index->get_timestamp_group_count(dense_node_id, forward, directed);
}

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::update_temporal_weights(const IEdgeData<GPUUsage>* edges, double timescale_bound)
{
    node_edge_index->update_temporal_weights(edges, timescale_bound);
}

template class NodeEdgeIndex<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeEdgeIndex<GPUUsageMode::ON_GPU>;
#endif