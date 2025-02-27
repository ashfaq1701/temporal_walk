#ifndef TEST_RUN_UTILS_H
#define TEST_RUN_UTILS_H

#include <iostream>

inline void print_temporal_random_walks(const std::vector<std::vector<int>>& walks) {
    for (auto & walk : walks) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << node << ", ";
        }

        std::cout << std::endl;
    }
}

inline void print_temporal_random_walks_with_times(const std::vector<std::vector<NodeWithTime>>& walks_with_times) {
    for (auto & walk : walks_with_times) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << "(" << node.node << ", " << node.timestamp << "), ";
        }

        std::cout << std::endl;
    }
}

inline void print_temporal_random_walks_for_nodes(const std::unordered_map<int, std::vector<std::vector<int>>>& walks_for_nodes) {
    for (const auto& [node, walks] : walks_for_nodes) {
        std::cout << "Walk for node " << node << std::endl;
        print_temporal_random_walks(walks);
        std::cout << std::endl;
        std::cout << "------------------------------";
        std::cout << std::endl;
    }
}

inline void print_temporal_random_walks_for_nodes_with_times(const std::unordered_map<int, std::vector<std::vector<NodeWithTime>>>& walks_for_nodes_with_times) {
    for (const auto& [node, walks_with_times] : walks_for_nodes_with_times) {
        std::cout << "Walk with times for node " << node << std::endl;
        print_temporal_random_walks_with_times(walks_with_times);
        std::cout << std::endl;
        std::cout << "------------------------------";
        std::cout << std::endl;
    }
}

inline double get_average_walk_length(const std::vector<std::vector<NodeWithTime>>& walks_for_nodes) {
    if (walks_for_nodes.empty()) return 0.0;

    size_t total_length = 0;
    size_t num_walks = 0;

    for (const auto& walks : walks_for_nodes) {
        total_length += walks.size();
        num_walks++;
    }

    return static_cast<double>(total_length) / static_cast<double>(num_walks);
}

#endif //TEST_RUN_UTILS_H
