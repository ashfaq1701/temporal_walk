#include "EdgeData.cuh"
#include <algorithm>
#include <iostream>

EdgeData::EdgeData(bool use_gpu): use_gpu(use_gpu) {
    sources = VectorTypes<int>::select(use_gpu);
    targets = VectorTypes<int>::select(use_gpu);
    timestamps = VectorTypes<int64_t>::select(use_gpu);

    timestamp_group_offsets = VectorTypes<size_t>::select(use_gpu);
    unique_timestamps = VectorTypes<int64_t>::select(use_gpu);

    forward_cumulative_weights_exponential = VectorTypes<double>::select(use_gpu);
    backward_cumulative_weights_exponential = VectorTypes<double>::select(use_gpu);
}


void EdgeData::reserve(size_t size)
{
    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec,
        auto& timestamp_group_offsets_vec, auto& unique_timestamps_vec)
    {
        sources_vec.reserve(size);
        targets_vec.reserve(size);
        timestamps_vec.reserve(size);
        timestamp_group_offsets_vec.reserve(size / 4 + 1); // Estimate for group count
        unique_timestamps_vec.reserve(size / 4);
    }, sources, targets, timestamps, timestamp_group_offsets, unique_timestamps);
}

void EdgeData::clear() {
    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec,
        auto& timestamp_group_offsets_vec, auto& unique_timestamps_vec)
    {
        sources_vec.clear();
        targets_vec.clear();
        timestamps_vec.clear();
        timestamp_group_offsets_vec.clear(); // Estimate for group count
        unique_timestamps_vec.clear();
    }, sources, targets, timestamps, timestamp_group_offsets, unique_timestamps);
}

size_t EdgeData::size() const {
    return std::visit([&](const auto& timestamps_vec) {
        return timestamps_vec.size();
    }, timestamps);
}

bool EdgeData::empty() const {
    return std::visit([&](const auto& timestamps_vec) {
        return timestamps_vec.empty();
    }, timestamps);
}

void EdgeData::resize(size_t new_size) {
    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec)
    {
        sources_vec.resize(new_size);
        targets_vec.resize(new_size);
        timestamps_vec.resize(new_size);
    }, sources, targets, timestamps);
}

void EdgeData::push_back(int src, int tgt, int64_t ts) {
    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec)
    {
        sources_vec.push_back(src);
        targets_vec.push_back(tgt);
        timestamps_vec.push_back(ts);
    }, sources, targets, timestamps);
}

std::vector<std::tuple<int, int, int64_t>> EdgeData::get_edges() {
    std::vector<std::tuple<int, int, int64_t>> edges;

    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec) {
        edges.reserve(sources_vec.size());

        for (int i = 0; i < sources_vec.size(); i++) {
            edges.emplace_back(sources_vec[i], targets_vec[i], timestamps_vec[i]);
        }

        return edges;
    }, sources, targets, timestamps);

    return edges;
}

void EdgeData::update_timestamp_groups() {
    std::visit([&](auto& ts_vec, auto& unique_ts_vec, auto& ts_group_offsets_vec) {
        if (ts_vec.empty()) {
            ts_group_offsets_vec.clear();
            unique_ts_vec.clear();
            return;
        }

        ts_group_offsets_vec.clear();
        unique_ts_vec.clear();

        ts_group_offsets_vec.push_back(0);
        unique_ts_vec.push_back(ts_vec[0]);

        for (size_t i = 1; i < ts_vec.size(); i++) {
            if (ts_vec[i] != ts_vec[i-1]) {
                ts_group_offsets_vec.push_back(i);
                unique_ts_vec.push_back(ts_vec[i]);
            }
        }

        ts_group_offsets_vec.push_back(ts_vec.size());
    }, timestamps, unique_timestamps, timestamp_group_offsets);
}

void EdgeData::update_temporal_weights(const double timescale_bound)
{
    std::visit([&](auto& ts_vec, auto& ts_group_offsets_vec, auto& forward_cumulative_weights_vec,
                   auto& backward_cumulative_weights_vec)
               {
                   if (ts_vec.empty())
                   {
                       forward_cumulative_weights_vec.clear();
                       backward_cumulative_weights_vec.clear();
                       return;
                   }

                   const int64_t min_timestamp = ts_vec[0];
                   const int64_t max_timestamp = ts_vec.back();
                   const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
                   const double time_scale = (timescale_bound > 0 && time_diff > 0) ? timescale_bound / time_diff : 1.0;

                   const size_t num_groups = get_timestamp_group_count();
                   forward_cumulative_weights_vec.resize(num_groups);
                   backward_cumulative_weights_vec.resize(num_groups);

                   double forward_sum = 0.0, backward_sum = 0.0;

                   // First calculate all weights and total sums
                   for (size_t group = 0; group < num_groups; group++)
                   {
                       const size_t start = ts_group_offsets_vec[group];
                       const int64_t group_timestamp = ts_vec[start];

                       const auto time_diff_forward = static_cast<double>(max_timestamp - group_timestamp);
                       const auto time_diff_backward = static_cast<double>(group_timestamp - min_timestamp);

                       const double forward_scaled = timescale_bound > 0
                                                         ? time_diff_forward * time_scale
                                                         : time_diff_forward;
                       const double backward_scaled = timescale_bound > 0
                                                          ? time_diff_backward * time_scale
                                                          : time_diff_backward;

                       const double forward_weight = exp(forward_scaled);
                       const double backward_weight = exp(backward_scaled);

                       forward_sum += forward_weight;
                       backward_sum += backward_weight;

                       forward_cumulative_weights_vec[group] = forward_weight;
                       backward_cumulative_weights_vec[group] = backward_weight;
                   }

                   // Then normalize and compute cumulative sums
                   double forward_cumsum = 0.0, backward_cumsum = 0.0;
                   for (size_t group = 0; group < num_groups; group++)
                   {
                       forward_cumulative_weights_vec[group] /= forward_sum;
                       backward_cumulative_weights_vec[group] /= backward_sum;

                       // Update with cumulative sums
                       forward_cumsum += forward_cumulative_weights_vec[group];
                       backward_cumsum += backward_cumulative_weights_vec[group];

                       forward_cumulative_weights_vec[group] = forward_cumsum;
                       backward_cumulative_weights_vec[group] = backward_cumsum;
                   }
               }, timestamps, timestamp_group_offsets, forward_cumulative_weights_exponential,
               backward_cumulative_weights_exponential);
}

std::pair<size_t, size_t> EdgeData::get_timestamp_group_range(size_t group_idx) const {
    return std::visit([&] (auto& unique_ts_vec, auto& ts_group_offsets_vec)
    {
        if (group_idx >= unique_ts_vec.size()) {
            return std::pair<size_t, size_t>{0, 0};
        }

        return std::pair<size_t, size_t>{ts_group_offsets_vec[group_idx], ts_group_offsets_vec[group_idx + 1]};
    }, unique_timestamps, timestamp_group_offsets);
}

size_t EdgeData::get_timestamp_group_count() const {
    return std::visit([](const auto& vec) { return vec.size(); }, unique_timestamps);
}

size_t EdgeData::find_group_after_timestamp(int64_t timestamp) const {
    return std::visit([&](const auto& unique_ts_vec) {
        if (unique_ts_vec.empty()) return 0L;

        auto it = std::upper_bound(unique_ts_vec.begin(), unique_ts_vec.end(), timestamp);
        return it - unique_ts_vec.begin();
    }, unique_timestamps);
}

size_t EdgeData::find_group_before_timestamp(int64_t timestamp) const {
    return std::visit([&](const auto& unique_ts_vec)
    {
        if (unique_ts_vec.empty()) return 0L;

        auto it = std::lower_bound(unique_ts_vec.begin(), unique_ts_vec.end(), timestamp);
        return (it - unique_ts_vec.begin()) - 1;
    }, unique_timestamps);
}
