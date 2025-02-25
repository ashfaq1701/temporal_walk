#ifndef I_TEMPORAL_RANDOM_WALK_H
#define I_TEMPORAL_RANDOM_WALK_H

#include<vector>
#include "../structs/structs.cuh"
#include "../structs/enums.h"
#include "../../libs/thread-pool/ThreadPool.h"
#include "../random/RandomPicker.h"
#include "../data/cpu/TemporalGraphCPU.cuh"
#include "../random/WeightBasedRandomPicker.cuh"

template<GPUUsageMode GPUUsage>
class ITemporalRandomWalk {

public:
    using EdgeVector = typename SelectVectorType<Edge, GPUUsage>::type;
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;

    bool is_directed = false;

    int64_t max_time_capacity = -1;

    bool enable_weight_computation = false;

    double timescale_bound = -1;

    int64_t max_edge_time = 0;

    ITemporalRandomWalk<GPUUsage>* temporal_graph;

    /**
     * HOST FUNCTIONS
     */

    HOST RandomPicker* get_random_picker_host(const RandomPickerType* picker_type) const { return nullptr; }

    [[nodiscard]] HOST WalkSet<GPUUsage> get_random_walks_and_times_for_all_nodes_host(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) { return WalkSet<GPUUsage>(); }

    [[nodiscard]] HOST WalkSet<GPUUsage> get_random_walks_and_times_host(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) { return WalkSet<GPUUsage>(); }

    HOST void add_multiple_edges_host(const EdgeVector& edge_infos) const {}

    [[nodiscard]] HOST size_t get_node_count_host() const { return 0; }

    [[nodiscard]] HOST size_t get_edge_count_host() const { return 0; }

    [[nodiscard]] HOST IntVector get_node_ids_host() const { return {}; }

    [[nodiscard]] HOST EdgeVector get_edges_host() const { return {}; }

    [[nodiscard]] HOST bool get_is_directed_host() const { return false; }

    HOST void clear_host() {}

    /**
     * DEVICE FUNCTIONS
     */

    DEVICE RandomPicker* get_random_picker_device(const RandomPickerType* picker_type) const { return nullptr; }

    [[nodiscard]] DEVICE WalkSet<GPUUsage> get_random_walks_and_times_for_all_nodes_device(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) { return WalkSet<GPUUsage>(); }

    [[nodiscard]] DEVICE WalkSet<GPUUsage> get_random_walks_and_times_device(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) { return WalkSet<GPUUsage>(); }

    DEVICE void add_multiple_edges_device(const EdgeVector& edge_infos) const {}

    [[nodiscard]] DEVICE size_t get_node_count_device() const { return 0; }

    [[nodiscard]] DEVICE size_t get_edge_count_device() const { return 0; }

    [[nodiscard]] DEVICE IntVector get_node_ids_device() const { return {}; }

    [[nodiscard]] DEVICE EdgeVector get_edges_device() const { return {}; }

    [[nodiscard]] DEVICE bool get_is_directed_device() const { return false; }

    DEVICE void clear_device() {}
};

#endif //I_TEMPORAL_RANDOM_WALK_H
