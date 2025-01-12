#include "TemporalGraph.h"
#include <algorithm>
#include <numeric>


// TemporalGraph implementation
TemporalGraph::TemporalGraph(bool directed, int64_t window)
    : is_directed(directed), time_window(window), latest_timestamp(0) {}

void TemporalGraph::sort_and_merge_edges(size_t start_idx) {
    // Sort new edges
    if (start_idx < edges.size()) {
        std::vector<size_t> indices(edges.size() - start_idx);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
            [this, start_idx](size_t a, size_t b) {
                return edges.timestamps[start_idx + a] < edges.timestamps[start_idx + b];
            });

        EdgeData temp;
        temp.reserve(edges.size() - start_idx);
        for (size_t idx : indices) {
            temp.push_back(
                edges.sources[start_idx + idx],
                edges.targets[start_idx + idx],
                edges.timestamps[start_idx + idx]
            );
        }

        // Copy back sorted edges
        for (size_t i = 0; i < temp.size(); i++) {
            edges.sources[start_idx + i] = temp.sources[i];
            edges.targets[start_idx + i] = temp.targets[i];
            edges.timestamps[start_idx + i] = temp.timestamps[i];
        }

        // Merge with existing edges if needed
        if (start_idx > 0) {
            std::inplace_merge(
                edges.timestamps.begin(),
                edges.timestamps.begin() + start_idx,
                edges.timestamps.end(),
                std::less<>()
            );

            std::vector<size_t> merge_indices(edges.size());
            std::iota(merge_indices.begin(), merge_indices.end(), 0);
            std::sort(merge_indices.begin(), merge_indices.end(),
                [this](size_t a, size_t b) {
                    return edges.timestamps[a] < edges.timestamps[b];
                });

            EdgeData merged;
            merged.reserve(edges.size());
            for (size_t idx : merge_indices) {
                merged.push_back(
                    edges.sources[idx],
                    edges.targets[idx],
                    edges.timestamps[idx]
                );
            }
            edges = std::move(merged);
        }
    }
}

void TemporalGraph::add_multiple_edges(
    const std::vector<std::tuple<int, int, int64_t>>& new_edges) {

    if (new_edges.empty()) return;

    size_t start_idx = edges.size();
    edges.reserve(start_idx + new_edges.size());

    // Add new edges and track max timestamp
    for (const auto& [src, tgt, ts] : new_edges) {
        if (!is_directed && src > tgt) {
            edges.push_back(tgt, src, ts);
        } else {
            edges.push_back(src, tgt, ts);
        }
        latest_timestamp = std::max(latest_timestamp, ts);
    }

    // Sort and merge edges
    sort_and_merge_edges(start_idx);

    // Update node mappings for new nodes
    node_mapping.update(edges, start_idx, edges.size());

    // Remove old edges if time window is set
    if (time_window > 0) {
        delete_old_edges();
    }

    // Rebuild node edge indices
    node_index.rebuild(edges, node_mapping, is_directed);
}

void TemporalGraph::delete_old_edges() {
    if (time_window <= 0 || edges.empty()) return;

    int64_t cutoff_time = latest_timestamp - time_window;
    auto it = std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), cutoff_time);
    if (it == edges.timestamps.begin()) return;

    size_t delete_count = it - edges.timestamps.begin();
    size_t remaining = edges.size() - delete_count;

    if (remaining > 0) {
        std::move(edges.sources.begin() + delete_count, edges.sources.end(), edges.sources.begin());
        std::move(edges.targets.begin() + delete_count, edges.targets.end(), edges.targets.begin());
        std::move(edges.timestamps.begin() + delete_count, edges.timestamps.end(), edges.timestamps.begin());
    }

    edges.resize(remaining);
}

size_t TemporalGraph::count_edges_less_than(int64_t timestamp) const {
    return std::lower_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp)
           - edges.timestamps.begin();
}

size_t TemporalGraph::count_edges_greater_than(int64_t timestamp) const {
    return edges.timestamps.end() -
           std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp);
}

size_t TemporalGraph::count_node_edges_less_than(int node_id, int64_t timestamp, bool forward) const {
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    auto [start_offset, end_offset] = node_index.get_edge_range(dense_idx, forward, is_directed);
    const auto& indices = forward ? node_index.outbound_indices : node_index.inbound_indices;

    return std::lower_bound(
        indices.begin() + start_offset,
        indices.begin() + end_offset,
        timestamp,
        [this](size_t idx, int64_t ts) {
            return edges.timestamps[idx] < ts;
        }
    ) - (indices.begin() + start_offset);
}

size_t TemporalGraph::count_node_edges_greater_than(int node_id, int64_t timestamp, bool forward) const {
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    auto [start_offset, end_offset] = node_index.get_edge_range(dense_idx, forward, is_directed);
    const auto& indices = forward ? node_index.outbound_indices : node_index.inbound_indices;

    return (indices.begin() + end_offset) -
           std::upper_bound(
               indices.begin() + start_offset,
               indices.begin() + end_offset,
               timestamp,
               [this](int64_t ts, size_t idx) {
                   return ts < edges.timestamps[idx];
               }
           );
}

std::tuple<int, int, int64_t> TemporalGraph::get_edge_at(
   size_t index, int64_t timestamp, bool forward) const {

   if (edges.empty()) return {-1, -1, -1};

   if (timestamp != -1) {
       if (forward) {
           // Find first edge after timestamp
           auto it = std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp);
           size_t start_idx = it - edges.timestamps.begin();
           size_t available_edges = edges.size() - start_idx;

           if (index >= available_edges) return {-1, -1, -1};
           return {
               edges.sources[start_idx + index],
               edges.targets[start_idx + index],
               edges.timestamps[start_idx + index]
           };
       } else {
           // Find edges before timestamp
           auto it = std::lower_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp);
           size_t available_edges = it - edges.timestamps.begin();

           if (index >= available_edges) return {-1, -1, -1};
           // Move backwards from cutoff point
           return {
               edges.sources[available_edges - index - 1],
               edges.targets[available_edges - index - 1],
               edges.timestamps[available_edges - index - 1]
           };
       }
   } else {
       // No timestamp constraint
       if (index >= edges.size()) return {-1, -1, -1};
       return {
           edges.sources[index],
           edges.targets[index],
           edges.timestamps[index]
       };
   }
}

std::tuple<int, int, int64_t> TemporalGraph::get_node_edge_at(
   int node_id, size_t index, int64_t timestamp, bool forward) const {

   int dense_idx = node_mapping.to_dense(node_id);
   if (dense_idx < 0) return {-1, -1, -1};

   auto [start_offset, end_offset] = node_index.get_edge_range(dense_idx, forward, is_directed);
   const auto& indices = forward ? node_index.outbound_indices : node_index.inbound_indices;

   if (timestamp != -1) {
       if (forward) {
           // Find first edge after timestamp
           auto it = std::upper_bound(
               indices.begin() + start_offset,
               indices.begin() + end_offset,
               timestamp,
               [this](int64_t ts, size_t idx) {
                   return ts < edges.timestamps[idx];
               }
           );

           size_t available_edges = end_offset - (it - indices.begin());
           if (available_edges == 0 || index >= available_edges) return {-1, -1, -1};

           size_t edge_idx = *(it + index);
           return {
               edges.sources[edge_idx],
               edges.targets[edge_idx],
               edges.timestamps[edge_idx]
           };
       } else {
           // Find first edge at or after timestamp
           auto it = std::lower_bound(
               indices.begin() + start_offset,
               indices.begin() + end_offset,
               timestamp,
               [this](size_t idx, int64_t ts) {
                   return edges.timestamps[idx] < ts;
               }
           );

           size_t available_edges = (it - indices.begin()) - start_offset;
           if (index >= available_edges) return {-1, -1, -1};

           // Move backwards from cutoff point
           size_t edge_idx = *(it - index - 1);
           return {
               edges.sources[edge_idx],
               edges.targets[edge_idx],
               edges.timestamps[edge_idx]
           };
       }
   } else {
       // No timestamp constraint
       if (index >= (end_offset - start_offset)) return {-1, -1, -1};

       if (forward) {
           size_t edge_idx = indices[start_offset + index];
           return {
               edges.sources[edge_idx],
               edges.targets[edge_idx],
               edges.timestamps[edge_idx]
           };
       } else {
           // For backward walks without timestamp, still walk backwards
           size_t edge_idx = indices[end_offset - index - 1];
           return {
               edges.sources[edge_idx],
               edges.targets[edge_idx],
               edges.timestamps[edge_idx]
           };
       }
   }
}
