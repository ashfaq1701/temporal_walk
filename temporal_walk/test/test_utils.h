#include <sstream>
#include <string>
#include <fstream>
#include "../src/core/TemporalWalk.h"

inline std::vector<std::tuple<int, int, int64_t>> read_edges_from_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::tuple<int, int, int64_t>> edges;
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string u_str, i_str, t_str;

        std::getline(ss, u_str, ',');
        std::getline(ss, i_str, ',');
        std::getline(ss, t_str, ',');
        edges.emplace_back(std::stoi(u_str), std::stoi(i_str), std::stoll(t_str));
    }

    return edges;
}
