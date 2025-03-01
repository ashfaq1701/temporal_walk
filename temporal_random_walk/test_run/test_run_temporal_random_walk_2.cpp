#include <vector>

#include "../src/core/TemporalRandomWalk.cuh"
#include "test_utils.h"
#include "../test/test_utils.h"

constexpr GPUUsageMode GPU_USAGE_MODE = GPUUsageMode::ON_CPU;

int main(int argc, char* argv[]) {
    std::string file_path = "../../data/sample_data.csv";

    if (argc > 1) {
        file_path = argv[1];
    }

    const auto edge_infos = read_edges_from_csv(file_path);
    std::cout << edge_infos.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    TemporalRandomWalk<GPU_USAGE_MODE> temporal_random_walk(false, -1, true, 34, 1);
    temporal_random_walk.add_multiple_edges(edge_infos);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Edge addition time: " << duration.count() << " seconds" << std::endl;

    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    start = std::chrono::high_resolution_clock::now();

    const auto walks_backward_for_all_nodes = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        80,
        &exponential_picker_type,
        10,
        &uniform_picker_type,
        WalkDirection::Backward_In_Time);

    const auto walks_forward_for_all_nodes = temporal_random_walk.get_random_walks_and_times_for_all_nodes(
        80,
        &exponential_picker_type,
        10,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time);
    std::cout << "Walks forward: " << walks_forward_for_all_nodes.size() << ", average length " << get_average_walk_length(walks_forward_for_all_nodes) << std::endl;

    std::cout << "Walks backward: " << walks_backward_for_all_nodes.size() << ", average length " << get_average_walk_length(walks_backward_for_all_nodes) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Walk generation time: " << duration.count() << " seconds" << std::endl;

    std::vector<std::vector<NodeWithTime>> first_100_walks_forward;
    first_100_walks_forward.assign(walks_forward_for_all_nodes.begin(), walks_forward_for_all_nodes.begin() + 100);

    print_temporal_random_walks_with_times(first_100_walks_forward);

    return 0;
}
