#include <vector>

#include "../src/core/TemporalWalk.h"
#include "print_walks.h"
#include "../test/test_utils.h"

int main() {
    const auto edge_infos = read_edges_from_csv("../../data/sample_data.csv");
    std::cout << edge_infos.size() << std::endl;

    TemporalWalk temporal_walk(false);
    temporal_walk.add_multiple_edges(edge_infos);

    constexpr int selected_node = 2;
    constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;
    constexpr RandomPickerType exponential_picker_type = RandomPickerType::Exponential;
    constexpr RandomPickerType uniform_picker_type = RandomPickerType::Uniform;

    auto start = std::chrono::high_resolution_clock::now();

    const auto walks_forward_with_specific_number_of_contexts = temporal_walk.get_random_walks_and_times_with_specific_number_of_contexts(
        80,
        &exponential_picker_type,
        -1,
        10,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time,
        10);
    std::cout << "Walks forward (with specific number of contexts): " << walks_forward_with_specific_number_of_contexts.size() << std::endl;

    const auto walks_backward_with_specific_number_of_contexts = temporal_walk.get_random_walks_and_times_with_specific_number_of_contexts(
        80,
        &exponential_picker_type,
        -1,
        10,
        nullptr,
        WalkDirection::Backward_In_Time,
        10);
    std::cout << "Walks backward (with specific number of contexts): " << walks_backward_with_specific_number_of_contexts.size() << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Runtime (with specific number of contexts): " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    const auto walks_forward_for_all_nodes = temporal_walk.get_random_walks_and_times_for_all_nodes(
        80,
        &exponential_picker_type,
        10,
        &uniform_picker_type,
        WalkDirection::Forward_In_Time);
    std::cout << "Walks forward (for all nodes): " << walks_forward_for_all_nodes.size() << std::endl;

    const auto walks_backward_for_all_nodes = temporal_walk.get_random_walks_and_times_for_all_nodes(
        80,
        &exponential_picker_type,
        10,
        nullptr,
        WalkDirection::Backward_In_Time);
    std::cout << "Walks backward (for all nodes): " << walks_backward_for_all_nodes.size() << std::endl;

    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Runtime (for all nodes): " << duration.count() << " seconds" << std::endl;

    return 0;
}
