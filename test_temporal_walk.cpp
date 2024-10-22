#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "TemporalWalk.h"

constexpr int NUM_WALKS = 500;
constexpr int LEN_WALK = 500;

std::vector<EdgeInfo> read_edges_from_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<EdgeInfo> edges;
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string u_str, i_str, t_str;

        std::getline(ss, u_str, ',');
        std::getline(ss, i_str, ',');
        std::getline(ss, t_str, ',');

        EdgeInfo edge{};
        edge.u = std::stoi(u_str);
        edge.i = std::stoi(i_str);
        edge.t = std::stoll(t_str);

        edges.push_back(edge);
    }

    return edges;
}

void print_temporal_walks(const std::vector<std::vector<int>>& walks) {
    for (auto & walk : walks) {
        std::cout << "Length: " << walk.size() << ", Walk: ";

        for (const auto node : walk) {
            std::cout << node << ", ";
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

int main() {
    const auto start = std::chrono::high_resolution_clock::now();

    const auto edge_infos = read_edges_from_csv("../../data/sample_data.csv");
    std::cout << edge_infos.size() << std::endl;

    TemporalWalk temporal_walk(NUM_WALKS, LEN_WALK, RandomPickerType::Linear);
    temporal_walk.add_multiple_edges(edge_infos);

    std::vector<int> start_nodes;
    for (int i = 0; i < 1000; i++) {
        start_nodes.push_back(i); // NOLINT(*-inefficient-vector-operation)
    }
    auto walks_for_nodes = temporal_walk.get_random_walks_for_nodes(start_nodes);
    print_temporal_walks_for_nodes(walks_for_nodes);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;
    std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;

    return 0;
}
