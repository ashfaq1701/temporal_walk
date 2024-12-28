#include "TemporalEdge.h"

TemporalEdge::TemporalEdge(Node* u, Node* i, const int64_t ts)
    : u(u), i(i), timestamp(ts) {}

Node* TemporalEdge::pick_random_endpoint() const {
    const bool is_u = get_random_boolean();
    return is_u ? u : i;
}

[[nodiscard]] Node* TemporalEdge::select_other_endpoint(const Node* node) const {
    return node == u ? i : u;
}
