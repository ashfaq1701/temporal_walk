#include <sstream>
#include <string>
#include <fstream>
#include "../core/TemporalWalk.h"

inline std::vector<EdgeInfo> read_edges_from_csv(const std::string& filename) {
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
