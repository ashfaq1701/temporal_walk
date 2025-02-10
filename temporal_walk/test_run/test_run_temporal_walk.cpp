#include <vector>

#include "../src/core/TemporalWalk.cuh"
#include "test_utils.h"

constexpr bool USE_GPU = false;

int main() {
    const std::vector<std::tuple<int, int, int64_t>> edges {
        {4, 5, 71},
        {3, 5, 82},
        {1, 3, 19},
        {4, 2, 34},
        {4, 3, 79},
        {2, 5, 19},
        {2, 3, 70},
        {5, 4, 97},
        {4, 6, 57},
        {6, 4, 27},
        {2, 6, 80},
        {6, 1, 42},
        {4, 6, 98},
        {1, 4, 17},
        {5, 4, 32}
    };

    TemporalWalk temporal_walk(true, USE_GPU);
    temporal_walk.add_multiple_edges(edges);

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;

    const auto walks_forward = temporal_walk.get_random_walks_and_times_for_all_nodes(20, &linear_picker_type, 10, &exponential_picker_type, WalkDirection::Forward_In_Time);
    std::cout << "Forward walks:" << std::endl;
    print_temporal_walks_with_times(walks_forward);

    const auto walks_backward = temporal_walk.get_random_walks_and_times_for_all_nodes(20, &linear_picker_type, 10, &exponential_picker_type, WalkDirection::Backward_In_Time);
    std::cout << "Backward walks:" << std::endl;
    print_temporal_walks_with_times(walks_backward);

    return 0;
}
