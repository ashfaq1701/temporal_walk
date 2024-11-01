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

    TemporalWalk temporal_walk(5, 10, RandomPickerType::Linear);
    temporal_walk.add_multiple_edges(edges);

    constexpr int selected_node = 2;

    const auto walks_starting_at = temporal_walk.get_random_walks_with_times(WalkStartAt::Begin, selected_node);
    std::cout << "Walks starting at " << selected_node << std::endl;
    print_temporal_walks_with_times(walks_starting_at);

    const auto walks_ending_at = temporal_walk.get_random_walks_with_times(WalkStartAt::End, selected_node);
    std::cout << "Walks ending at " << selected_node << std::endl;
    print_temporal_walks_with_times(walks_ending_at);

    return 0;
}
