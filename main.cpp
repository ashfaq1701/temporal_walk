#include <iostream>
#include "models/TemporalGraph.h"

int main() {
    TemporalGraph temporal_graph;
    temporal_graph.addEdge(1, 2, 3);
    temporal_graph.addEdge(1, 2, 4);

    std::cout << "Hello, World! " << temporal_graph.get_node_count() << " " << temporal_graph.get_edge_count() << std::endl;
    return 0;
}
