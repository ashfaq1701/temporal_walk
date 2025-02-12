#ifndef TEMPORAL_WALK_STRUCTS_H
#define TEMPORAL_WALK_STRUCTS_H

#include <cstdint>

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

enum GPUUsageMode {
    ON_CPU,
    DATA_ON_GPU,
    DATA_ON_HOST
};

struct NodeWithTime {
    int node;
    int64_t timestamp;
};

#endif //TEMPORAL_WALK_STRUCTS_H
