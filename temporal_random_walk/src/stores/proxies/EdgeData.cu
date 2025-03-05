#include "EdgeData.cuh"

template<GPUUsageMode GPUUsage>
EdgeData<GPUUsage>::EdgeData(): edge_data(new BaseType()) {}

template<GPUUsageMode GPUUsage>
EdgeData<GPUUsage>::EdgeData(IEdgeData<GPUUsage>* edge_data): edge_data(edge_data) {}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::reserve(size_t size)
{
    edge_data->reserve(size);
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::clear()
{
    edge_data->clear();
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::size() const
{
    return edge_data->size();
}

template <GPUUsageMode GPUUsage>
bool EdgeData<GPUUsage>::empty() const
{
    return edge_data->empty();
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::resize(size_t new_size)
{
    edge_data->resize(new_size);
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::add_edges(int* src, int* tgt, int64_t* ts, size_t size)
{
    edge_data->add_edges(src, tgt, ts, size);
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::push_back(int src, int tgt, int64_t ts)
{
    edge_data->push_back(src, tgt, ts);
}

template <GPUUsageMode GPUUsage>
std::vector<Edge> EdgeData<GPUUsage>::get_edges()
{
    std::vector<Edge> results;
    auto edges = edge_data->get_edges();
    for (auto edge : edges)
    {
        results.push_back(edge);
    }

    return results;
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::update_timestamp_groups()
{
    edge_data->update_timestamp_groups();
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::update_temporal_weights(double timescale_bound)
{
    edge_data->update_temporal_weights(timescale_bound);
}

template <GPUUsageMode GPUUsage>
std::pair<size_t, size_t> EdgeData<GPUUsage>::get_timestamp_group_range(size_t group_idx)
{
    auto group_range = edge_data->get_timestamp_group_range(group_idx);
    return {group_range.from, group_range.to};
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::get_timestamp_group_count() const
{
    return edge_data->get_timestamp_group_count();
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const
{
    return edge_data->find_group_after_timestamp(timestamp);
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const
{
    return edge_data->find_group_before_timestamp(timestamp);
}

template class EdgeData<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class EdgeData<GPUUsageMode::ON_GPU>;
#endif

