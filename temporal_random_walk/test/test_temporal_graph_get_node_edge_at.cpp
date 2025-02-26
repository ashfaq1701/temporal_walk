#include <gtest/gtest.h>
#include "../src/stores/proxies/TemporalGraph.cuh"
#include "../src/stores/cuda/TemporalGraphCUDA.cuh"
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

template<typename T>
class TemporalGraphGetNodeEdgeAtTest : public ::testing::Test {
protected:
    using TemporalGraphType = std::conditional_t<
        T::value == GPUUsageMode::ON_CPU,
        TemporalGraph<T::value>,
        TemporalGraphCUDA<T::value>
    >;

    std::unique_ptr<TemporalGraphType> graph;
    std::unique_ptr<FirstIndexPicker> first_picker;
    std::unique_ptr<LastIndexPicker> last_picker;

    void SetUp() override {
        graph = std::make_unique<TemporalGraphType>(true); // directed graph
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

#ifdef HAS_CUDA
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_CPU>,
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_GPU>
>;
#else
using GPU_USAGE_TYPES = ::testing::Types<
    std::integral_constant<GPUUsageMode, GPUUsageMode::ON_CPU>
>;
#endif

TYPED_TEST_SUITE(TemporalGraphGetNodeEdgeAtTest, GPU_USAGE_TYPES);

// Test forward walks from a node
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ForwardWalkTest) {
    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {10, 20, 100}, // Node 10 outbound group 1
        {10, 30, 100},
        {10, 40, 102}, // Node 10 outbound group 2
        {10, 50, 104}, // Node 10 outbound group 3
        {20, 10, 101}, // Node 10 inbound (should be ignored for forward)
        {30, 10, 103}  // Node 10 inbound (should be ignored for forward)
    };
    this->graph->add_multiple_edges(edges);

    // Test no timestamp constraint
    auto edge = this->graph->get_node_edge_at(10, *this->first_picker, -1, true);
    EXPECT_EQ(std::get<2>(edge), 100); // Should select from first group
    EXPECT_EQ(std::get<0>(edge), 10);  // Source should be 10

    edge = this->graph->get_node_edge_at(10, *this->last_picker, -1, true);
    EXPECT_EQ(std::get<2>(edge), 104); // Should select from last group
    EXPECT_EQ(std::get<0>(edge), 10);  // Source should be 10

    // Test with timestamp constraints
    edge = this->graph->get_node_edge_at(10, *this->first_picker, 102, true);
    EXPECT_EQ(std::get<2>(edge), 104); // Should select first group after 102

    edge = this->graph->get_node_edge_at(10, *this->first_picker, 103, true);
    EXPECT_EQ(std::get<2>(edge), 104); // Should select first group after 103

    // Test no available groups after timestamp
    edge = this->graph->get_node_edge_at(10, *this->first_picker, 104, true);
    this->verify_edge(edge, -1, -1, -1); // No groups after 104
}

TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, BackwardWalkTest) {
    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {20, 10, 100}, // Node 10 inbound: ts 100 -> 103
        {30, 10, 101}, // From upstream (source) nodes: 20,30,40,50
        {40, 10, 102}, // To downstream node: 10
        {50, 10, 103},
        {10, 20, 101}, // Node 10 outbound (should be ignored for backward)
        {10, 30, 102}  // Node 10 outbound (should be ignored for backward)
    };
    this->graph->add_multiple_edges(edges);

    // Test no timestamp constraint
    auto edge = this->graph->get_node_edge_at(10, *this->first_picker, -1, false);
    this->verify_edge(edge, 20, 10, 100); // Should select from first group with first_picker

    edge = this->graph->get_node_edge_at(10, *this->last_picker, -1, false);
    this->verify_edge(edge, 50, 10, 103); // Should select from last group with last_picker

    // Test with timestamp constraints
    edge = this->graph->get_node_edge_at(10, *this->first_picker, 104, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 104

    edge = this->graph->get_node_edge_at(10, *this->first_picker, 103, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 103

    edge = this->graph->get_node_edge_at(10, *this->first_picker, 102, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 102

    edge = this->graph->get_node_edge_at(10, *this->first_picker, 101, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as lowest timestamp < 101

    // Test with timestamp constraints
    edge = this->graph->get_node_edge_at(10, *this->last_picker, 104, false);
    this->verify_edge(edge, 50, 10, 103); // Should select 103 as highest timestamp < 104

    edge = this->graph->get_node_edge_at(10, *this->last_picker, 103, false);
    this->verify_edge(edge, 40, 10, 102); // Should select 102 as highest timestamp < 103

    edge = this->graph->get_node_edge_at(10, *this->last_picker, 102, false);
    this->verify_edge(edge, 30, 10, 101); // Should select 101 as highest timestamp < 102

    edge = this->graph->get_node_edge_at(10, *this->first_picker, 101, false);
    this->verify_edge(edge, 20, 10, 100); // Should select 100 as highest timestamp < 101

    // Test edge cases
    edge = this->graph->get_node_edge_at(10, *this->first_picker, 100, false);
    this->verify_edge(edge, -1, -1, -1); // No timestamps < 100

    edge = this->graph->get_node_edge_at(10, *this->first_picker, 99, false);
    this->verify_edge(edge, -1, -1, -1); // No timestamps < 99
}

// Test edge cases and invalid inputs
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, EdgeCasesTest) {
    using TemporalGraphType = typename TestFixture::TemporalGraphType;

    auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {10, 20, 100}, // Node 10 outbound: ts 100,101
        {10, 30, 101}, // Node 10 -> Nodes 20,30
    };
    this->graph->add_multiple_edges(edges);

    // Test invalid node ID
    auto edge = this->graph->get_node_edge_at(-1, *this->first_picker, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    // Test non-existent node
    edge = this->graph->get_node_edge_at(999, *this->first_picker, -1, true);
    this->verify_edge(edge, -1, -1, -1);

    // Test node with no edges in requested direction
    edge = this->graph->get_node_edge_at(20, *this->first_picker, -1, true); // Node 20 has no outbound edges
    this->verify_edge(edge, -1, -1, -1);

    // Test backward walk for node with no inbound edges
    edge = this->graph->get_node_edge_at(10, *this->first_picker, -1, false); // Node 10 has no inbound edges
    this->verify_edge(edge, -1, -1, -1);

    // Test empty graph
    this->graph = std::make_unique<TemporalGraphType>(true);
    edge = this->graph->get_node_edge_at(10, *this->first_picker, -1, true);
    this->verify_edge(edge, -1, -1, -1);
}

// Test random selection within timestamp groups
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, RandomSelectionTest) {
    const auto edges = std::vector<std::tuple<int, int, int64_t>>{
        {10, 20, 100}, // Group 1: ts 100
        {10, 30, 100},
        {10, 40, 100},
        {10, 50, 101}  // Group 2: ts 101
    };
    this->graph->add_multiple_edges(edges);

    // Make multiple selections from first timestamp group
    std::set<int> seen_targets;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto edge = this->graph->get_node_edge_at(10, *this->first_picker, -1, true);
        EXPECT_EQ(std::get<2>(edge), 100); // Should always be from first group
        seen_targets.insert(std::get<1>(edge));
    }

    // Should see more than one target due to random selection within group ts=100
    EXPECT_GT(seen_targets.size(), 1);
}

// Test exact timestamp matching
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ExactTimestampTest) {
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
    this->graph->add_multiple_edges(edges);

    // Forward direction (node 10 to downstream)
    auto edge = this->graph->get_node_edge_at(10, *this->first_picker, 100, true);
    this->verify_edge(edge, 10, 60, 101); // Should get next timestamp going downstream

    // Backward direction (node 10 looking upstream)
    edge = this->graph->get_node_edge_at(10, *this->first_picker, 101, false);
    this->verify_edge(edge, 20, 10, 100); // Should get previous timestamp from upstream
}

// Test exact timestamp matching for undirected graphs
TYPED_TEST(TemporalGraphGetNodeEdgeAtTest, ExactTimestampUndirectedTest) {
    using TemporalGraphType = typename TestFixture::TemporalGraphType;
    // Create undirected graph
    this->graph = std::make_unique<TemporalGraphType>(false);

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
    this->graph->add_multiple_edges(edges);

    // Forward direction should work same as backward
    auto edge = this->graph->get_node_edge_at(10, *this->first_picker, 100, true);
    EXPECT_EQ(std::get<2>(edge), 101);
    EXPECT_TRUE(std::get<0>(edge) == 10 && (std::get<1>(edge) == 30 || std::get<1>(edge) == 60));

    // Backward direction
    edge = this->graph->get_node_edge_at(10, *this->first_picker, 101, false);
    EXPECT_EQ(std::get<2>(edge), 100);
    EXPECT_TRUE(std::get<0>(edge) == 10 && (std::get<1>(edge) == 20 || std::get<1>(edge) == 50));

    // Try from other node's perspective
    edge = this->graph->get_node_edge_at(20, *this->first_picker, 100, true);
    this->verify_edge(edge, 20, 30, 104);
}
