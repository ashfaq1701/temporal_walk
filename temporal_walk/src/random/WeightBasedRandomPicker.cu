#include "WeightBasedRandomPicker.cuh"

WeightBasedRandomPicker::WeightBasedRandomPicker(bool use_gpu): use_gpu(use_gpu) {}

int WeightBasedRandomPicker::pick_random(
    const VectorTypes<double>::Vector& cumulative_weights,
    const int group_start,
    const int group_end)
{
    return std::visit([&](const auto& weights_vec) {
        // Validate inputs
        if (group_start < 0 || group_end <= group_start ||
            group_end > static_cast<int>(weights_vec.size())) {
            return -1;
        }

        // Get start and end sums
        double start_sum = 0.0;
        if (group_start > 0 && weights_vec[group_start] >= weights_vec[group_start - 1]) {
            start_sum = weights_vec[group_start - 1];
        }
        const double end_sum = weights_vec[group_end - 1];

        if (end_sum < start_sum) {
            return -1;
        }

        // Generate random value between [start_sum, end_sum]
        const double random_val = generate_random_value(start_sum, end_sum);

        // Find the group where random_val falls
        return static_cast<int>(std::lower_bound(
            weights_vec.begin() + group_start,
            weights_vec.begin() + group_end,
            random_val) - weights_vec.begin());
    }, cumulative_weights);
}