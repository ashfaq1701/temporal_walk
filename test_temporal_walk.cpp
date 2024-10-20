#include <iostream>
#include "models/TemporalGraph.h"
#include "random/ExponentialRandomPicker.h"
#include "random/LinearRandomPicker.h"
#include "random/UniformRandomPicker.h"

int main() {
    TemporalGraph temporal_graph;
    temporal_graph.add_edge(1, 2, 3);
    temporal_graph.add_edge(1, 2, 4);
    temporal_graph.add_edge(2, 3, 5);
    temporal_graph.add_edge(4, 2, 7);

    std::cout << "Hello, World! " << temporal_graph.get_node_count() << " " << temporal_graph.get_edge_count() << std::endl;

    auto random_picker = new LinearRandomPicker();
    for (int i = 0; i < 10; i++) {
        std::cout << random_picker->pick_random(100, 300) << std::endl;
    }

    return 0;
}
