#ifndef TEMPORAL_WALK_H
#define TEMPORAL_WALK_H

#include<vector>

#include "random/RandomPicker.h"
#include "models/TemporalGraph.h"

enum RandomPickerType {
    Uniform,
    Linear,
    Exponential
};

class TemporalWalk {
    int num_walks;
    int len_walk;

    std::unique_ptr<TemporalGraph> temporal_graph;
    std::unique_ptr<RandomPicker> random_picker;

public:
    TemporalWalk(int num_walks, int len_walk, RandomPickerType picker_type);

    std::vector<std::vector<int>> get_random_walks(int start_node=-1);

    void generate_random_walk(std::vector<int>* walk, int start_node=-1);
};

#endif //TEMPORAL_WALK_H