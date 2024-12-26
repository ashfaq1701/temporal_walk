#ifndef TEMPORAL_WALK_H
#define TEMPORAL_WALK_H

#include<vector>
#include "../../libs/thread-pool/ThreadPool.h"
#include "../random/RandomPicker.h"
#include "../models/TemporalGraph.h"

constexpr float DEFAULT_SUCCESS_THRESHOLD = 0.01;

enum RandomPickerType {
    Uniform,
    Linear,
    Exponential
};

enum WalkInitEdgeTimeBias {
    Bias_Earliest_Time,
    Bias_Latest_Time
};

enum WalkDirection {
    Forward_In_Time,
    Backward_In_Time
};

struct EdgeInfo {
    int u;
    int i;
    int64_t t;

    EdgeInfo(int u, int i, int64_t t);
};

struct NodeWithTime {
    int node;
    int64_t timestamp;
};

class TemporalWalk {
    int64_t max_time_capacity;

    int64_t max_edge_time = 0;

    std::unique_ptr<TemporalGraph> temporal_graph;

    ThreadPool thread_pool;

    void generate_random_walk_with_time(
        std::vector<NodeWithTime>* walk,
        const std::shared_ptr<RandomPicker>& edge_picker,
        const std::shared_ptr<RandomPicker>& start_picker,
        int max_walk_len,
        bool should_walk_forward,
        bool init_edge_picker_end_prioritization) const;

    void add_edge(int u, int i, int64_t t);

    static std::shared_ptr<RandomPicker> get_random_picker(const RandomPickerType* picker_type);

    [[nodiscard]] long estimate_cw_count(int num_walks_per_node, int max_walk_len, int min_walk_len) const;

public:
    explicit TemporalWalk(int64_t max_time_capacity=-1, size_t n_threads=std::thread::hardware_concurrency());

    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_with_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias walk_init_edge_time_bias=WalkInitEdgeTimeBias::Bias_Earliest_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD);

    void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos);

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<EdgeInfo> get_edges() const;

    void clear();
};

#endif //TEMPORAL_WALK_H