#include "NodeMapping.cuh"

template<GPUUsageMode GPUUsage>
NodeMapping<GPUUsage>::NodeMapping(): node_mapping(new BaseType()) {}

template<GPUUsageMode GPUUsage>
NodeMapping<GPUUsage>::NodeMapping(INodeMapping<GPUUsage>* node_mapping): node_mapping(node_mapping) {}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::update(const IEdgeData<GPUUsage>* edges, size_t start_idx, size_t end_idx)
{
    node_mapping->update(edges, start_idx, end_idx);
}

template<GPUUsageMode GPUUsage>
int NodeMapping<GPUUsage>::to_dense(int sparse_id) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return node_mapping->to_dense_host(sparse_id);
    }
    else
    {
        return node_mapping->to_dense_device(sparse_id);
    }
}

template<GPUUsageMode GPUUsage>
int NodeMapping<GPUUsage>::to_sparse(int dense_idx) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return node_mapping->to_sparse_host(dense_idx);
    }
    else
    {
        return node_mapping->to_sparse_device(dense_idx);
    }
}

template<GPUUsageMode GPUUsage>
size_t NodeMapping<GPUUsage>::size() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return node_mapping->size_host();
    }
    else
    {
        return node_mapping->size_device();
    }
}

template<GPUUsageMode GPUUsage>
size_t NodeMapping<GPUUsage>::active_size() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return node_mapping->active_size_host();
    }
    else
    {
        return node_mapping->active_size_device();
    }
}

template<GPUUsageMode GPUUsage>
HOST std::vector<int> NodeMapping<GPUUsage>::get_active_node_ids() const
{
    std::vector<int> result;
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto active_ids = node_mapping->get_active_node_ids_host();
        for (int i = 0; i < active_ids.size(); i++)
        {
            result.push_back(active_ids[i]);
        }
    }
    else
    {
        auto active_ids = node_mapping->get_active_node_ids_device();
        for (int i = 0; i < active_ids.size(); i++)
        {
            result.push_back(active_ids[i]);
        }
    }
    return result;
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::clear()
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        node_mapping->clear_host();
    }
    else
    {
        node_mapping->clear_device();
    }
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::reserve(size_t size)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        node_mapping->reserve_host(size);
    }
    else
    {
        node_mapping->reserve_device(size);
    }
}

template<GPUUsageMode GPUUsage>
void NodeMapping<GPUUsage>::mark_node_deleted(int sparse_id)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        node_mapping->mark_node_deleted_host(sparse_id);
    }
    else
    {
        node_mapping->mark_node_deleted_device(sparse_id);
    }
}

template<GPUUsageMode GPUUsage>
bool NodeMapping<GPUUsage>::has_node(int sparse_id)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return node_mapping->has_node_host(sparse_id);
    }
    else
    {
        return node_mapping->has_node_device(sparse_id);
    }
}

template<GPUUsageMode GPUUsage>
std::vector<int> NodeMapping<GPUUsage>::get_all_sparse_ids() const
{
    std::vector<int> result;
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto all_ids = node_mapping->get_all_sparse_ids_host();
        for (int i = 0; i < all_ids.size(); i++)
        {
            result.push_back(all_ids[i]);
        }
    }
    else
    {
        auto all_ids = node_mapping->get_all_sparse_ids_device();
        for (int i = 0; i < all_ids.size(); i++)
        {
            result.push_back(all_ids[i]);
        }
    }
    return result;
}

template class NodeMapping<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeMapping<GPUUsageMode::ON_GPU>;
#endif
