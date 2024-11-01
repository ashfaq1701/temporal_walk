#ifndef PRINT_WALKS_H
#define PRINT_WALKS_H

#include <iostream>

inline void print_temporal_walks(const std::vector<std::vector<int>>& walks) {
    for (auto & walk : walks) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << node << ", ";
        }

        std::cout << std::endl;
    }
}

inline void print_temporal_walks_with_times(const std::vector<std::vector<NodeWithTime>>& walks_with_times) {
    for (auto & walk : walks_with_times) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << "(" << node.node << ", " << node.timestamp << "), ";
        }

        std::cout << std::endl;
    }
}

inline void print_temporal_walks_for_nodes(const std::unordered_map<int, std::vector<std::vector<int>>>& walks_for_nodes) {
    for (const auto& [node, walks] : walks_for_nodes) {
        std::cout << "Walk for node " << node << std::endl;
        print_temporal_walks(walks);
        std::cout << std::endl;
        std::cout << "------------------------------";
        std::cout << std::endl;
    }
}

inline void print_temporal_walks_for_nodes_with_times(const std::unordered_map<int, std::vector<std::vector<NodeWithTime>>>& walks_for_nodes_with_times) {
    for (const auto& [node, walks_with_times] : walks_for_nodes_with_times) {
        std::cout << "Walk with times for node " << node << std::endl;
        print_temporal_walks_with_times(walks_with_times);
        std::cout << std::endl;
        std::cout << "------------------------------";
        std::cout << std::endl;
    }
}

#endif //PRINT_WALKS_H
