#ifndef TEMPORAL_WALK_STRUCTS_H
#define TEMPORAL_WALK_STRUCTS_H

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

#endif //TEMPORAL_WALK_STRUCTS_H
