#include "ExponentialRandomPicker.h"
#include <random>
#include <cmath>
#include <stdexcept>

int ExponentialRandomPicker::pick_random(const int start, const int end, const bool prioritize_end) {
    // Ensure the input range is valid
    if (start >= end) {
        throw std::invalid_argument("Start must be less than end.");
    }

    const int len_seq = end - start;

    // Generate a random value 'u' uniformly distributed in the range [0, 1)
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double u = dist(thread_local_gen);

    // Original formula:
    // log_index = log1p(u * (exp(len_seq) - 1))
    // To avoid computing exp(len_seq) directly (which can overflow), rewrite it using logarithmic properties:
    // log_index = log((1 - u) * exp(len_seq) + u) = len_seq + log1p(-u) + log1p(u * exp(-shifted_log))

    // Compute log(1 - u), which is always negative since 0 ≤ u < 1
    // This term helps avoid direct computation of exp(len_seq), which can overflow
    const double log_term = std::log1p(-u);

    // Shift the log by adding len_seq, effectively computing len_seq + log(1 - u)
    // This value represents a logarithmic offset to avoid dealing with large numbers directly
    const double shifted_log = log_term + len_seq;

    // Add another adjustment to account for the scaled contribution of u in log-space
    // This avoids overflow by computing log1p(u * exp(-shifted_log))
    // log_index represents the position in the sequence in log-space
    const double log_index = shifted_log + std::log1p(u * std::exp(-shifted_log));

    if (prioritize_end) {
        // When prioritizing end, use the computed log_index directly
        // Ensure the index does not exceed len_seq - 1
        return start + std::min(static_cast<int>(log_index), len_seq - 1);
    } else {
        // When prioritizing start, reverse the computed index
        // Ensure the reversed index does not drop below 0
        const int revered_index = len_seq - 1 - static_cast<int>(log_index);
        return start + std::max(0, revered_index);
    }
}
