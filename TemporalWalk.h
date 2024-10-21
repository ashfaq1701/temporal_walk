#ifndef TEMPORAL_WALK_H
#define TEMPORAL_WALK_H

#include<vector>

#include "random/RandomPicker.h"
#include "models/TemporalGraph.h"
#include "libs/thread_pool.h"

enum RandomPickerType {
    Uniform,
    Linear,
    Exponential
};

struct EdgeInfo {
    int u;
    int v;
    int64_t t;
};

class TemporalWalk {
    int num_walks;
    int len_walk;

    std::unique_ptr<TemporalGraph> temporal_graph;
    std::unique_ptr<RandomPicker> random_picker;

    ThreadPool thread_pool;

public:
    TemporalWalk(int num_walks, int len_walk, RandomPickerType picker_type);

    [[nodiscard]] std::vector<std::vector<int>> get_random_walks(int start_node=-1);

    void generate_random_walk(std::vector<int>* walk, int start_node=-1) const;

    void add_edge(int u, int v, int64_t t) const;

    void add_multiple_edges(const std::vector<EdgeInfo>& edge_infos) const;
};

#endif //TEMPORAL_WALK_H