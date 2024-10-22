#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "TemporalWalk.h"

constexpr int NUM_WALKS = 10000;
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

int main() {
    const auto start = std::chrono::high_resolution_clock::now();

    const auto edge_infos = read_edges_from_csv("../data/sample_data.csv");

    TemporalWalk temporal_walk(NUM_WALKS, LEN_WALK, RandomPickerType::Linear);
    temporal_walk.add_multiple_edges(edge_infos);

    const auto walks = temporal_walk.get_random_walks();

    print_temporal_walks(walks);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;
    std::cout << "Runtime: " << duration.count() << " seconds" << std::endl;

    return 0;
}
