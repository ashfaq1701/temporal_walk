#ifndef TEMPORAL_WALK_H
#define TEMPORAL_WALK_H

#include<vector>
#include "../../libs/thread-pool/include/BS_thread_pool.hpp"
#include "../random/RandomPicker.h"
#include "../models/TemporalGraph.h"

enum RandomPickerType {
    Uniform,
    Linear,
    Exponential
};

enum WalkStartAt {
    Begin,
    End,
    Random
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

    BS::thread_pool thread_pool;

    void generate_random_walk_with_time(std::vector<NodeWithTime>* walk, bool begin_from_end, int len_walk, const std::shared_ptr<RandomPicker>& edge_picker, const std::shared_ptr<RandomPicker>& start_picker, int end_node=-1) const;

    void add_edge(int u, int i, int64_t t);

    static std::shared_ptr<RandomPicker> get_random_picker(const RandomPickerType* picker_type);

public:
    explicit TemporalWalk(int64_t max_time_capacity=-1);

    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_with_times(WalkStartAt walk_start_at, int num_walks, int len_walk, const RandomPickerType* edge_picker_type, int end_node=-1, const RandomPickerType* start_picker_type=nullptr);
    [[nodiscard]] std::unordered_map<int, std::vector<std::vector<NodeWithTime>>> get_random_walks_for_nodes_with_times(WalkStartAt walk_start_at, const std::vector<int>& end_nodes, int num_walks, int len_walk, const RandomPickerType* edge_picker_type, const RandomPickerType* start_picker_type=nullptr);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks(WalkStartAt walk_start_at, int num_walks, int len_walk, const RandomPickerType* edge_picker_type, int end_node=-1, const RandomPickerType* start_picker_type=nullptr);
    [[nodiscard]] std::unordered_map<int, std::vector<std::vector<int>>> get_random_walks_for_nodes(WalkStartAt walk_start_at, const std::vector<int>& end_nodes, int num_walks, int len_walk, const RandomPickerType* edge_picker_type, const RandomPickerType* start_picker_type=nullptr);

    void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos);

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;

    [[nodiscard]] std::vector<EdgeInfo> get_edges() const;

    void clear();
};

#endif //TEMPORAL_WALK_H