#ifndef I_TEMPORAL_RANDOM_WALK_H
#define I_TEMPORAL_RANDOM_WALK_H

#include "../config/constants.h"
#include "../structs/structs.cuh"
#include "../structs/enums.h"
#include "../random/RandomPicker.h"
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

    ITemporalRandomWalk<GPUUsage>* temporal_graph = nullptr;

    explicit HOST ITemporalRandomWalk(
        bool is_directed,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND)
    : is_directed(is_directed), max_time_capacity(max_time_capacity),
        enable_weight_computation(enable_weight_computation), timescale_bound(timescale_bound) {}

    virtual ~ITemporalRandomWalk() = default;

    virtual HOST RandomPicker* get_random_picker(const RandomPickerType* picker_type) const { return nullptr; }

    [[nodiscard]] virtual HOST WalkSet<GPUUsage> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) { return WalkSet<GPUUsage>(); }

    [[nodiscard]] virtual HOST WalkSet<GPUUsage> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) { return WalkSet<GPUUsage>(); }

    virtual HOST void add_multiple_edges(const EdgeVector& edge_infos) const {}

    [[nodiscard]] virtual HOST size_t get_node_count() const { return 0; }

    [[nodiscard]] virtual HOST size_t get_edge_count() const { return 0; }

    [[nodiscard]] virtual HOST IntVector get_node_ids() const { return {}; }

    [[nodiscard]] virtual HOST EdgeVector get_edges() const { return {}; }

    [[nodiscard]] virtual HOST bool get_is_directed() const { return false; }

    virtual HOST void clear() {}
};

#endif //I_TEMPORAL_RANDOM_WALK_H
