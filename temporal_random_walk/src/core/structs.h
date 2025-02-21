#ifndef TEMPORAL_RANDOM_WALK_STRUCTS_H
#define TEMPORAL_RANDOM_WALK_STRUCTS_H

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
    ON_HOST_USING_THRUST,
    ON_GPU_USING_CUDA
};

struct NodeWithTime {
    int node;
    int64_t timestamp;
};

#endif //TEMPORAL_RANDOM_WALK_STRUCTS_H
