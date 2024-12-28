#include <vector>

#include "../src/core/TemporalWalk.h"
#include "print_walks.h"
#include "../test/test_utils.h"

int main() {
    const auto start = std::chrono::high_resolution_clock::now();

    const auto edge_infos = read_edges_from_csv("../../data/sample_data.csv");
    std::cout << edge_infos.size() << std::endl;

    TemporalWalk temporal_walk(false);
    temporal_walk.add_multiple_edges(edge_infos);

    constexpr int selected_node = 2;
    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::Exponential;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    const auto walks_forward = temporal_walk.get_random_walks_with_times(
        80,
        &exponential_picker_type,
        -1,
        10,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time,
        WalkInitEdgeTimeBias::Bias_Earliest_Time,
        10);
    std::cout << "Walks forward: " << walks_forward.size() << std::endl;

    const auto walks_backward = temporal_walk.get_random_walks_with_times(
        80,
        &exponential_picker_type,
        -1,
        10,
        nullptr,
        WalkDirection::Backward_In_Time,
        WalkInitEdgeTimeBias::Bias_Latest_Time,
        10);
    std::cout << "Walks backward: " << walks_backward.size() << std::endl;

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;
    std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;

    return 0;
}
