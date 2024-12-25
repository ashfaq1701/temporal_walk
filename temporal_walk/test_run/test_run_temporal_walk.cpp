#include <vector>

#include "../src/core/TemporalWalk.h"
#include "print_walks.h"

int main() {
    const std::vector<EdgeInfo> edges {
        EdgeInfo{4, 5, 71},
        EdgeInfo{3, 5, 82},
        EdgeInfo{1, 3, 19},
        EdgeInfo{4, 2, 34},
        EdgeInfo{4, 3, 79},
        EdgeInfo{2, 5, 19},
        EdgeInfo{2, 3, 70},
        EdgeInfo{5, 4, 97},
        EdgeInfo{4, 6, 57},
        EdgeInfo{6, 4, 27},
        EdgeInfo{2, 6, 80},
        EdgeInfo{6, 1, 42},
        EdgeInfo{4, 6, 98},
        EdgeInfo{1, 4, 17},
        EdgeInfo{5, 4, 32}
    };

    TemporalWalk temporal_walk;
    temporal_walk.add_multiple_edges(edges);

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::Exponential;

    const auto walks_forward = temporal_walk.get_random_walks_with_times(20, &linear_picker_type, -1, 10, &exponential_picker_type, WalkDirection::Forward_In_Time, WalkInitEdgeTimeBias::Bias_Earliest_Time);
    std::cout << "Forward walks:" << std::endl;
    print_temporal_walks_with_times(walks_forward);

    const auto walks_backward = temporal_walk.get_random_walks_with_times(20, &linear_picker_type, -1, 10, &exponential_picker_type, WalkDirection::Backward_In_Time, WalkInitEdgeTimeBias::Bias_Latest_Time);
    std::cout << "Backward walks:" << std::endl;
    print_temporal_walks_with_times(walks_backward);

    return 0;
}
