#include <gtest/gtest.h>
#include "../src/data/TemporalGraph.cuh"
#include "../src/random/IndexBasedRandomPicker.h"

// Test-specific picker implementations
class FirstIndexPicker : public IndexBasedRandomPicker {
public:
    [[nodiscard]] int pick_random(int start, int end, bool prioritize_end) override {
        return start;
    }
};

class LastIndexPicker : public IndexBasedRandomPicker {
public:
    [[nodiscard]] int pick_random(int start, int end, bool prioritize_end) override {
        return end - 1;
    }
};

class TemporalGraphGetNodeEdgeAtTest : public ::testing::TestWithParam<bool> {
protected:
    std::unique_ptr<TemporalGraph> graph;
    std::unique_ptr<FirstIndexPicker> first_picker;
    std::unique_ptr<LastIndexPicker> last_picker;

    void SetUp() override {
        graph = std::make_unique<TemporalGraph>(true, GetParam()); // directed graph
        first_picker = std::make_unique<FirstIndexPicker>();
        last_picker = std::make_unique<LastIndexPicker>();
    }

    // Helper to verify edge fields
    static void verify_edge(const std::tuple<int, int, int64_t>& edge,
                          int expected_src, int expected_tgt, int64_t expected_ts) {
        EXPECT_EQ(std::get<0>(edge), expected_src);
        EXPECT_EQ(std::get<1>(edge), expected_tgt);
        EXPECT_EQ(std::get<2>(edge), expected_ts);
    }
};

// Test forward walks from a node
TEST_P(TemporalGraphGetNodeEdgeAtTest, ForwardWalkTest) {
    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {10, 20, 100}, // Node 10 outbound group 1
        {10, 30, 100},
        {10, 40, 102}, // Node 10 outbound group 2
        {10, 50, 104}, // Node 10 outbound group 3
        {20, 10, 101}, // Node 10 inbound (should be ignored for forward)
        {30, 10, 103}  // Node 10 inbound (should be ignored for forward)
    };
    graph->add_multiple_edges(edges);

    // Test no timestamp constraint
    auto edge = graph->get_node_edge_at(10, *first_picker, -1, true);
    EXPECT_EQ(std::get<2>(edge), 100); // Should select from first group
    EXPECT_EQ(std::get<0>(edge), 10);  // Source should be 10

    edge = graph->get_node_edge_at(10, *last_picker, -1, true);
    EXPECT_EQ(std::get<2>(edge), 104); // Should select from last group
    EXPECT_EQ(std::get<0>(edge), 10);  // Source should be 10

    // Test with timestamp constraints
    edge = graph->get_node_edge_at(10, *first_picker, 102, true);
    EXPECT_EQ(std::get<2>(edge), 104); // Should select first group after 102

    edge = graph->get_node_edge_at(10, *first_picker, 103, true);
    EXPECT_EQ(std::get<2>(edge), 104); // Should select first group after 103

    // Test no available groups after timestamp
    edge = graph->get_node_edge_at(10, *first_picker, 104, true);
    verify_edge(edge, -1, -1, -1); // No groups after 104
}

TEST_P(TemporalGraphGetNodeEdgeAtTest, BackwardWalkTest) {
    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {20, 10, 100}, // Node 10 inbound: ts 100 -> 103
        {30, 10, 101}, // From upstream (source) nodes: 20,30,40,50
        {40, 10, 102}, // To downstream node: 10
        {50, 10, 103},
        {10, 20, 101}, // Node 10 outbound (should be ignored for backward)
        {10, 30, 102}  // Node 10 outbound (should be ignored for backward)
    };
    graph->add_multiple_edges(edges);

    // Test no timestamp constraint
    auto edge = graph->get_node_edge_at(10, *first_picker, -1, false);
    verify_edge(edge, 20, 10, 100); // Should select from first group with first_picker

    edge = graph->get_node_edge_at(10, *last_picker, -1, false);
    verify_edge(edge, 50, 10, 103); // Should select from last group with last_picker

    // Test with timestamp constraints
    edge = graph->get_node_edge_at(10, *first_picker, 104, false);
    verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 104

    edge = graph->get_node_edge_at(10, *first_picker, 103, false);
    verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 103

    edge = graph->get_node_edge_at(10, *first_picker, 102, false);
    verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 102

    edge = graph->get_node_edge_at(10, *first_picker, 101, false);
    verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 101

    // Test with timestamp constraints
    edge = graph->get_node_edge_at(10, *last_picker, 104, false);
    verify_edge(edge, 50, 10, 103); // Should select 103 as highest timestamp < 104

    edge = graph->get_node_edge_at(10, *last_picker, 103, false);
    verify_edge(edge, 40, 10, 102); // Should select 102 as highest timestamp < 103

    edge = graph->get_node_edge_at(10, *last_picker, 102, false);
    verify_edge(edge, 30, 10, 101); // Should select 101 as highest timestamp < 102

    edge = graph->get_node_edge_at(10, *first_picker, 101, false);
    verify_edge(edge, 20, 10, 100); // Should select 100 as highest timestamp < 101

    // Test edge cases
    edge = graph->get_node_edge_at(10, *first_picker, 100, false);
    verify_edge(edge, -1, -1, -1); // No timestamps < 100

    edge = graph->get_node_edge_at(10, *first_picker, 99, false);
    verify_edge(edge, -1, -1, -1); // No timestamps < 99
}

// Test edge cases and invalid inputs
TEST_P(TemporalGraphGetNodeEdgeAtTest, EdgeCasesTest) {
    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {10, 20, 100}, // Node 10 outbound: ts 100,101
        {10, 30, 101}, // Node 10 -> Nodes 20,30
    };
    graph->add_multiple_edges(edges);

    // Test invalid node ID
    auto edge = graph->get_node_edge_at(-1, *first_picker, -1, true);
    verify_edge(edge, -1, -1, -1);

    // Test non-existent node
    edge = graph->get_node_edge_at(999, *first_picker, -1, true);
    verify_edge(edge, -1, -1, -1);

    // Test node with no edges in requested direction
    edge = graph->get_node_edge_at(20, *first_picker, -1, true); // Node 20 has no outbound edges
    verify_edge(edge, -1, -1, -1);

    // Test backward walk for node with no inbound edges
    edge = graph->get_node_edge_at(10, *first_picker, -1, false); // Node 10 has no inbound edges
    verify_edge(edge, -1, -1, -1);

    // Test empty graph
    graph = std::make_unique<TemporalGraph>(true, GetParam());
    edge = graph->get_node_edge_at(10, *first_picker, -1, true);
    verify_edge(edge, -1, -1, -1);
}

// Test random selection within timestamp groups
TEST_P(TemporalGraphGetNodeEdgeAtTest, RandomSelectionTest) {
    const auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {10, 20, 100}, // Group 1: ts 100
        {10, 30, 100},
        {10, 40, 100},
        {10, 50, 101}  // Group 2: ts 101
    };
    graph->add_multiple_edges(edges);

    // Make multiple selections from first timestamp group
    std::set<int> seen_targets;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto edge = graph->get_node_edge_at(10, *first_picker, -1, true);
        EXPECT_EQ(std::get<2>(edge), 100); // Should always be from first group
        seen_targets.insert(std::get<1>(edge));
    }

    // Should see more than one target due to random selection within group ts=100
    EXPECT_GT(seen_targets.size(), 1);
}

// Test exact timestamp matching
TEST_P(TemporalGraphGetNodeEdgeAtTest, ExactTimestampTest) {
    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        // Edges for backward walks (upstream -> node 10)
        {20, 10, 100}, // Upstream nodes 20,30,40 -> downstream node 10
        {30, 10, 101},
        {40, 10, 102},
        // Edges for forward walks (node 10 -> downstream)
        {10, 50, 100}, // Upstream node 10 -> downstream nodes 50,60,70
        {10, 60, 101},
        {10, 70, 102}
    };
    graph->add_multiple_edges(edges);

    // Forward direction (node 10 to downstream)
    auto edge = graph->get_node_edge_at(10, *first_picker, 100, true);
    verify_edge(edge, 10, 60, 101); // Should get next timestamp going downstream

    // Backward direction (node 10 looking upstream)
    edge = graph->get_node_edge_at(10, *first_picker, 101, false);
    verify_edge(edge, 20, 10, 100); // Should get previous timestamp from upstream
}

// Test exact timestamp matching for undirected graphs
TEST_P(TemporalGraphGetNodeEdgeAtTest, ExactTimestampUndirectedTest) {
    // Create undirected graph
    graph = std::make_unique<TemporalGraph>(false, GetParam());

    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        // Edges connecting to node 10
        {10, 20, 100}, // Will be normalized (stored as min source, max target)
        {30, 10, 101}, // These edges connect node 10 with 20,30,40,50,60,70
        {10, 40, 102}, // In both directions since it's undirected
        {50, 10, 100},
        {10, 60, 101},
        {70, 10, 102},
        {20, 30, 104}
    };
    graph->add_multiple_edges(edges);

    // Forward direction should work same as backward
    auto edge = graph->get_node_edge_at(10, *first_picker, 100, true);
    EXPECT_EQ(std::get<2>(edge), 101);
    EXPECT_TRUE(std::get<0>(edge) == 10 && (std::get<1>(edge) == 30 || std::get<1>(edge) == 60));

    // Backward direction
    edge = graph->get_node_edge_at(10, *first_picker, 101, false);
    EXPECT_EQ(std::get<2>(edge), 100);
    EXPECT_TRUE(std::get<0>(edge) == 10 && (std::get<1>(edge) == 20 || std::get<1>(edge) == 50));

    // Try from other node's perspective
    edge = graph->get_node_edge_at(20, *first_picker, 100, true);
    verify_edge(edge, 20, 30, 104);
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    TemporalGraphGetNodeEdgeAtTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    TemporalGraphGetNodeEdgeAtTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
