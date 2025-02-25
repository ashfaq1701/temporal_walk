#ifndef TEMPORAL_RANDOM_WALK_CPU_H
#define TEMPORAL_RANDOM_WALK_CPU_H

#include<vector>
#include "ITemporalRandomWalk.cuh"
#include "../structs/structs.cuh"
#include "../structs/enums.h"
#include "../config/constants.h"
#include "../../libs/thread-pool/ThreadPool.h"
#include "../random/RandomPicker.h"
#include "../random/WeightBasedRandomPickerGPU.cuh"

template<GPUUsageMode GPUUsage>
class TemporalRandomWalkCPU : public ITemporalRandomWalk<GPUUsage> {
protected:

    size_t n_threads;
    ThreadPool thread_pool;

    HOST void generate_random_walk_and_time(
        int walk_idx,
        WalkSet<GPUUsage>& walk_set,
        const RandomPicker* edge_picker,
        const RandomPicker* start_picker,
        int max_walk_len,
        bool should_walk_forward,
        int start_node_id=-1) const;

    HOST RandomPicker* get_random_picker(const RandomPickerType* picker_type) const override;

public:
    explicit HOST TemporalRandomWalkCPU(
        bool is_directed,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND,
        size_t n_threads=std::thread::hardware_concurrency());

    [[nodiscard]] HOST WalkSet<GPUUsage> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) override;

    [[nodiscard]] HOST WalkSet<GPUUsage> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) override;

    HOST void add_multiple_edges(const typename ITemporalRandomWalk<GPUUsage>::EdgeVector& edge_infos) const override;

    [[nodiscard]] HOST size_t get_node_count() const override;

    [[nodiscard]] HOST size_t get_edge_count() const override;

    [[nodiscard]] HOST ITemporalRandomWalk<GPUUsage>::IntVector get_node_ids() const override;

    [[nodiscard]] HOST ITemporalRandomWalk<GPUUsage>::EdgeVector get_edges() const override;

    [[nodiscard]] HOST bool get_is_directed() const override;

    HOST void clear() override;
};

#endif //TEMPORAL_RANDOM_WALK_CPU_H
