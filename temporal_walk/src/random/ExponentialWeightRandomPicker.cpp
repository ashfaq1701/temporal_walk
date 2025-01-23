#include "ExponentialWeightRandomPicker.h"

int ExponentialWeightRandomPicker::pick_random(
    const std::vector<double>& weights,
    int group_start,
    int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(weights.size()))
    {
        return -1;
    }

    // Create discrete distribution directly with iterator range
    std::discrete_distribution<int> dist(
        weights.begin() + group_start,
        weights.begin() + group_end
    );

    // Select random index within the group
    return group_start + dist(get_random_generator());
}
