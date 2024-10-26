#ifndef TEMPORAL_WALK_H
#define TEMPORAL_WALK_H

#include<vector>

#include "../random/RandomPicker.h"
#include "../models/TemporalGraph.h"
#include "../libs/thread_pool.h"

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
};

struct NodeWithTime {
    int node;
    int64_t timestamp;
};

class TemporalWalk {
    int num_walks;
    int len_walk;

    std::unique_ptr<TemporalGraph> temporal_graph;
    std::unique_ptr<RandomPicker> random_picker;

    ThreadPool thread_pool;

    void generate_random_walk_with_time(std::vector<NodeWithTime>* walk, bool begin_from_end, int end_node=-1) const;

public:
    TemporalWalk(int num_walks, int len_walk, RandomPickerType picker_type);

    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_with_times(WalkStartAt walk_start_at, int end_node=-1);
    [[nodiscard]] std::unordered_map<int, std::vector<std::vector<NodeWithTime>>> get_random_walks_for_nodes_with_times(WalkStartAt walk_start_at, const std::vector<int>& end_nodes);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks(WalkStartAt walk_start_at, int end_node=-1);
    [[nodiscard]] std::unordered_map<int, std::vector<std::vector<int>>> get_random_walks_for_nodes(WalkStartAt walk_start_at, const std::vector<int>& end_nodes);

    void add_edge(int u, int i, int64_t t) const;

    void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos) const;

    [[nodiscard]] int get_len_walk() const;

    [[nodiscard]] size_t get_node_count() const;

    [[nodiscard]] size_t get_edge_count() const;

    [[nodiscard]] std::vector<int> get_node_ids() const;
};

#endif //TEMPORAL_WALK_H