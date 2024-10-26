#include <iostream>
#include <fstream>
#include <vector>
#include "test_utils.h"
#include "../core/TemporalWalk.h"

constexpr int NUM_WALKS = 20;
constexpr int LEN_WALK = 500;
constexpr RandomPickerType RANDOM_PICKER_TYPE = RandomPickerType::Linear;

void print_temporal_walks(const std::vector<std::vector<int>>& walks) {
    for (auto & walk : walks) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << node << ", ";
        }

        std::cout << std::endl;
    }
}

void print_temporal_walks_with_times(const std::vector<std::vector<NodeWithTime>>& walks_with_times) {
    for (auto & walk : walks_with_times) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << "(" << node.node << ", " << node.timestamp << "), ";
        }

        std::cout << std::endl;
    }
}

void print_temporal_walks_for_nodes(const std::unordered_map<int, std::vector<std::vector<int>>>& walks_for_nodes) {
    for (const auto& [node, walks] : walks_for_nodes) {
        std::cout << "Walk for node " << node << std::endl;
        print_temporal_walks(walks);
        std::cout << std::endl;
        std::cout << "------------------------------";
        std::cout << std::endl;
    }
}

void print_temporal_walks_for_nodes_with_times(const std::unordered_map<int, std::vector<std::vector<NodeWithTime>>>& walks_for_nodes_with_times) {
    for (const auto& [node, walks_with_times] : walks_for_nodes_with_times) {
        std::cout << "Walk with times for node " << node << std::endl;
        print_temporal_walks_with_times(walks_with_times);
        std::cout << std::endl;
        std::cout << "------------------------------";
        std::cout << std::endl;
    }
}

int main() {
    const auto start = std::chrono::high_resolution_clock::now();

    const auto edge_infos = read_edges_from_csv("../../data/sample_data.csv");
    std::cout << edge_infos.size() << std::endl;

    TemporalWalk temporal_walk(NUM_WALKS, LEN_WALK, RANDOM_PICKER_TYPE);
    temporal_walk.add_multiple_edges(edge_infos);

    const std::vector<int> nodes = temporal_walk.get_node_ids();
    std::cout << "Total node count: " << nodes.size() << std::endl;

    const auto selected_nodes = std::vector<int>(nodes.begin(), nodes.begin() + 100);

    const auto walks_for_nodes = temporal_walk.get_random_walks_for_nodes(WalkStartAt::Random, selected_nodes);
    print_temporal_walks_for_nodes(walks_for_nodes);

    const auto walks_for_nodes_with_times = temporal_walk.get_random_walks_for_nodes_with_times(WalkStartAt::Random, selected_nodes);
    print_temporal_walks_for_nodes_with_times(walks_for_nodes_with_times);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;


    const auto random_walks_from_begin = temporal_walk.get_random_walks(WalkStartAt::Begin);
    const auto random_walks_from_end = temporal_walk.get_random_walks(WalkStartAt::End);

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Random Walks from beginning to end" << std::endl;
    print_temporal_walks(random_walks_from_begin);
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "Random Walks from end to beginning" << std::endl;
    print_temporal_walks(random_walks_from_end);
    std::cout << "------------------------------------------------------" << std::endl;

    std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;

    return 0;
}
