#include <vector>

#include "../src/core/TemporalRandomWalk.cuh"
#include "test_utils.h"
#include "../test/test_utils.h"

constexpr GPUUsageMode GPU_USAGE_MODE = GPUUsageMode::ON_CPU;

int main() {
    const auto edge_infos = read_edges_from_csv("../../data/sample_data.csv");
    std::cout << edge_infos.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    TemporalRandomWalk<GPU_USAGE_MODE> temporal_random_walk(false, -1, true, 34);
    temporal_random_walk.add_multiple_edges(edge_infos);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Edge addition time: " << duration.count() << " seconds" << std::endl;


    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::ExponentialIndex;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    start = std::chrono::high_resolution_clock::now();

    const auto walks_backward_with_specific_number_of_contexts = temporal_random_walk.get_random_walks_and_times_with_specific_number_of_contexts(
        80,
        &exponential_picker_type,
        -1,
        10,
        &uniform_picker_type,
        WalkDirection::Backward_In_Time,
        3);

    const auto walks_forward_with_specific_number_of_contexts = temporal_random_walk.get_random_walks_and_times_with_specific_number_of_contexts(
        80,
        &exponential_picker_type,
        -1,
        10,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time,
        5);
    std::cout << "Walks forward (with specific number of contexts): " << walks_forward_with_specific_number_of_contexts.size() << ", average length " << get_average_walk_length(walks_forward_with_specific_number_of_contexts) << std::endl;

    std::cout << "Walks backward (with specific number of contexts): " << walks_backward_with_specific_number_of_contexts.size() << ", average length " << get_average_walk_length(walks_backward_with_specific_number_of_contexts) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Runtime (with specific number of contexts): " << duration.count() << " seconds" << std::endl;

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
    std::cout << "Walks forward (for all nodes): " << walks_forward_for_all_nodes.size() << ", average length " << get_average_walk_length(walks_forward_for_all_nodes) << std::endl;

    std::cout << "Walks backward (for all nodes): " << walks_backward_for_all_nodes.size() << ", average length " << get_average_walk_length(walks_backward_for_all_nodes) << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Runtime (for all nodes): " << duration.count() << " seconds" << std::endl;

    std::vector<std::vector<NodeWithTime>> first_100_walks_forward;
    first_100_walks_forward.assign(walks_forward_for_all_nodes.begin(), walks_forward_for_all_nodes.begin() + 100);

    print_temporal_random_walks_with_times(first_100_walks_forward);

    return 0;
}
