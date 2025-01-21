#include "ExponentialWeightRandomPicker.h"

int ExponentialWeightRandomPicker::pick_random(
    const std::vector<double>& prob,
    const std::vector<int>& alias,
    int start,
    int length) {

    // Validate inputs
    if (length <= 0 || start < 0 ||
        start >= static_cast<int>(prob.size()) ||
        start >= static_cast<int>(alias.size()) ||
        start + length > static_cast<int>(prob.size()) ||
        start + length > static_cast<int>(alias.size())) {
        return -1;
    }

    // Generate random column index
    const int col = start + get_random_number(length);

    // Perform alias table lookup
    // If random value < stored probability, return this index
    // Otherwise return the alias index for this position
    return (get_random_double() < prob[col]) ? col : alias[col];
}
