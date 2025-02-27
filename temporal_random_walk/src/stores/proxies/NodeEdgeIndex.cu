#include "NodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::clear()
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        node_edge_index.clear_host();
    }
    else
    {
        node_edge_index.clear_device();
    }
}

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        node_edge_index.rebuild_host(edges, mapping, is_directed);
    }
    else
    {
        node_edge_index.rebuild_device(edges, mapping, is_directed);
    }
}

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> NodeEdgeIndex<GPUUsage>::get_edge_range(int dense_node_id, bool forward, bool is_directed) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto range = node_edge_index.get_edge_range_host(dense_node_id, forward, is_directed);
        return {range.from, range.to};
    }
    else
    {
        auto range = node_edge_index.get_edge_range_device(dense_node_id, forward, is_directed);
        return {range.from, range.to};
    }
}

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> NodeEdgeIndex<GPUUsage>::get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto range = node_edge_index.get_timestamp_group_range_host(dense_node_id, group_idx, forward, is_directed);
        return {range.from, range.to};
    }
    else
    {
        auto range = node_edge_index.get_timestamp_group_range_device(dense_node_id, group_idx, forward, is_directed);
        return {range.from, range.to};
    }
}

template<GPUUsageMode GPUUsage>
size_t NodeEdgeIndex<GPUUsage>::get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return node_edge_index.get_timestamp_group_count_host(dense_node_id, forward, directed);
    }
    else
    {
        return node_edge_index.get_timestamp_group_count_device(dense_node_id, forward, directed);
    }
}

template class NodeEdgeIndex<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeEdgeIndex<GPUUsageMode::ON_GPU>;
#endif