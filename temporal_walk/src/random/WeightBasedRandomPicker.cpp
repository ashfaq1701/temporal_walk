#include "WeightBasedRandomPicker.h"

template<typename T>
int WeightBasedRandomPicker::pick_random(
    const std::vector<T>& cumulative_weights,
    const int group_start,
    const int group_end)
{
    // Validate inputs
    if (group_start < 0 || group_end <= group_start ||
        group_end > static_cast<int>(cumulative_weights.size()))
    {
        return -1;
    }

    // Get start and end sums
    const T start_sum = (group_start > 0) ? cumulative_weights[group_start - 1] : T{0};
    const T end_sum = cumulative_weights[group_end - 1];

    if (end_sum < start_sum) return -1;

    // Generate random value between [start_sum, end_sum]
    const T random_val = start_sum + generate_random_value(T{0}, end_sum - start_sum);

    // Find the group where random_val falls
    return static_cast<int>(std::lower_bound(
        cumulative_weights.begin() + group_start,
        cumulative_weights.begin() + group_end,
        random_val) - cumulative_weights.begin());
}

template int WeightBasedRandomPicker::pick_random<double>(
    const std::vector<double>&, const int, const int);
