#include <iostream>
#include "TemporalWalk.h"
#include "random/LinearRandomPicker.h"

int main() {
    auto random_picker = new LinearRandomPicker();
    TemporalWalk temporal_walk(3, 4, RandomPickerType::Linear);
    temporal_walk.add_edge(1, 2, 3);
    temporal_walk.add_edge(1, 2, 4);
    temporal_walk.add_edge(2, 3, 5);
    temporal_walk.add_edge(4, 2, 7);

    for (int i = 0; i < 10; i++) {
        std::cout << random_picker->pick_random(100, 300) << std::endl;
    }

    return 0;
}
