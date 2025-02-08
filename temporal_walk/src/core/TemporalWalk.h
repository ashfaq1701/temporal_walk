#ifndef TEMPORAL_WALK_H
#define TEMPORAL_WALK_H

#include<vector>
#include "../../libs/thread-pool/ThreadPool.h"
#include "../random/RandomPicker.h"
#include "../data/TemporalGraph.cuh"

enum RandomPickerType {
    Uniform,
    Linear,
    ExponentialIndex,
    ExponentialWeight
};

enum WalkDirection {
    Forward_In_Time,
    Backward_In_Time
};

struct NodeWithTime {
    int node;
    int64_t timestamp;
};

template<bool UseGPU>
class TemporalWalk {
    bool is_directed;

    int64_t max_time_capacity;

    int n_threads;

    bool enable_weight_computation;

    double timescale_bound;

    int64_t max_edge_time = 0;

    #ifdef USE_CUDA
    using TemporalGraphType = std::conditional_t<true, TemporalGraph<true>, TemporalGraph<false>>;
    #else
    using TemporalGraphType = TemporalGraph<false>;
    #endif
    std::unique_ptr<TemporalGraphType> temporal_graph;

    ThreadPool thread_pool;

    void generate_random_walk_and_time(
        std::vector<NodeWithTime>* walk,
        const std::shared_ptr<RandomPicker<UseGPU>>& edge_picker,
        const std::shared_ptr<RandomPicker<UseGPU>>& start_picker,
        int max_walk_len,
        bool should_walk_forward,
        int start_node_id=-1) const;

    std::shared_ptr<RandomPicker<UseGPU>> get_random_picker(const RandomPickerType* picker_type) const;

    [[nodiscard]] long estimate_cw_count(int num_walks_per_node, int max_walk_len, int min_walk_len) const;

public:
    explicit TemporalWalk(
        bool is_directed,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND,
        size_t n_threads=std::thread::hardware_concurrency());

    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD);

    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edge_infos) const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const;

    [[nodiscard]] bool get_is_directed() const;

    void clear();
};

#endif //TEMPORAL_WALK_H
