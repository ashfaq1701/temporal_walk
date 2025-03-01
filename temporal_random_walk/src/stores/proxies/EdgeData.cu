#include "EdgeData.cuh"

template<GPUUsageMode GPUUsage>
EdgeData<GPUUsage>::EdgeData(): edge_data(new BaseType()) {}

template<GPUUsageMode GPUUsage>
EdgeData<GPUUsage>::EdgeData(IEdgeData<GPUUsage>* edge_data): edge_data(edge_data) {}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::reserve(size_t size)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        edge_data->reserve_host(size);
    } else
    {
        edge_data->reserve_device(size);
    }
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::clear()
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        edge_data->clear_host();
    } else
    {
        edge_data->clear_device();
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::size() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return edge_data->size_host();
    } else
    {
        return edge_data->size_device();
    }
}

template <GPUUsageMode GPUUsage>
bool EdgeData<GPUUsage>::empty() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return edge_data->empty_host();
    } else
    {
        return edge_data->empty_device();
    }
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::resize(size_t new_size)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        edge_data->resize_host(new_size);
    } else
    {
        edge_data->resize_device(new_size);
    }
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::add_edges(int* src, int* tgt, int64_t* ts, size_t size)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        edge_data->add_edges_host(src, tgt, ts, size);
    } else
    {
        edge_data->add_edges_device(src, tgt, ts, size);
    }
}

template <GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::push_back(int src, int tgt, int64_t ts)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        edge_data->push_back_host(src, tgt, ts);
    } else
    {
        edge_data->push_back_device(src, tgt, ts);
    }
}

template <GPUUsageMode GPUUsage>
std::vector<Edge> EdgeData<GPUUsage>::get_edges()
{
    std::vector<Edge> results;

    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto edges = edge_data->get_edges_host();
        for (auto edge : edges)
        {
            results.push_back(edge);
        }
    } else
    {
        auto edges = edge_data->get_edges_device();
        for (auto edge : edges)
        {
            results.push_back(edge);
        }
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
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto group_range = edge_data->get_timestamp_group_range_host(group_idx);
        return {group_range.from, group_range.to};
    } else
    {
        auto group_range = edge_data->get_timestamp_group_range_device(group_idx);
        return {group_range.from, group_range.to};
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::get_timestamp_group_count() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return edge_data->get_timestamp_group_count_host();
    } else
    {
        return edge_data->get_timestamp_group_count_device();
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return edge_data->find_group_after_timestamp_host(timestamp);
    } else
    {
        return edge_data->find_group_after_timestamp_device(timestamp);
    }
}

template <GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return edge_data->find_group_before_timestamp_host(timestamp);
    } else
    {
        return edge_data->find_group_before_timestamp_device(timestamp);
    }
}

template class EdgeData<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class EdgeData<GPUUsageMode::ON_GPU>;
#endif

