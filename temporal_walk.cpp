#include "temporal_walk.h"
#include "random/UniformRandomPicker.h"
#include "random/LinearRandomPicker.h"
#include "random/ExponentialRandomPicker.h"

TemporalWalk::TemporalWalk(const int num_walks, const int len_walk, RandomPickerType picker_type)
    : num_walks(num_walks), len_walk(len_walk) {
    temporal_graph = std::make_unique<TemporalGraph>();

    switch (picker_type) {
        case Uniform:
            random_picker = std::make_unique<UniformRandomPicker>();
        break;
        case Linear:
            random_picker = std::make_unique<LinearRandomPicker>();
        break;
        case Exponential:
            random_picker = std::make_unique<ExponentialRandomPicker>();
        break;
        default:
            throw std::invalid_argument("Invalid picker type");
    }
}

std::vector<std::vector<int>> TemporalWalk::get_random_walks(const int start_node) {
    std::vector walks(num_walks, std::vector<int>(len_walk));

    for (auto & walk : walks) {
        generate_random_walk(&walk, start_node);
    }

    return walks;
}

void TemporalWalk::generate_random_walk(std::vector<int>* walk, const int start_node) {

}
