#include "ExponentialWeightRandomPicker.h"

int ExponentialWeightRandomPicker::pick_random(
    const std::vector<double>& group_probs,
    const std::vector<int>& group_aliases,
    int group_start,
    int group_length) {

    // Validate inputs
    if (group_length <= 0 || group_start < 0 ||
        group_start >= static_cast<int>(group_probs.size()) ||
        group_start >= static_cast<int>(group_aliases.size()) ||
        group_start + group_length > static_cast<int>(group_probs.size()) ||
        group_start + group_length > static_cast<int>(group_aliases.size())) {
        return -1;
    }

    // Generate random column index
    const int col = group_start + get_random_number(group_length);

    // Perform alias table lookup
    // If random value < stored probability, return this index
    // Otherwise return the alias index for this position
    return (get_random_double() < group_probs[col]) ? col : group_aliases[col];
}
