#include "ExponentialWeightRandomPicker.h"

int ExponentialWeightRandomPicker::pick_random(
    const std::vector<double>& cumulative_weights,
    int group_start,
    int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size()))
    {
        return -1;
    }

    // Get start and end sums
    const double start_sum = (group_start > 0) ? cumulative_weights[group_start - 1] : 0.0;
    const double end_sum = cumulative_weights[group_end - 1];

    if (end_sum <= start_sum) return -1;

    // Generate random value between [start_sum, end_sum]
    double random_val = start_sum + get_random_double() * (end_sum - start_sum);

    // Find the group where random_val falls
    return static_cast<int>(std::lower_bound(
        cumulative_weights.begin() + group_start,
        cumulative_weights.begin() + group_end,
        random_val) - cumulative_weights.begin());
}
