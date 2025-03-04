#ifndef ENUMS_H
#define ENUMS_H

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
    ON_GPU
};

#endif // ENUMS_H
