#include <gtest/gtest.h>
#include "../src/data/TemporalGraph.cuh"
#include "../src/random/IndexBasedRandomPicker.h"

// Test-specific picker that always selects first element
class FirstIndexPicker : public IndexBasedRandomPicker {
public:
    [[nodiscard]] int pick_random(int start, int end, bool prioritize_end) override {
        return start;
    }
};

// Test-specific picker that always selects last element
class LastIndexPicker : public IndexBasedRandomPicker {
public:
    [[nodiscard]] int pick_random(int start, int end, bool prioritize_end) override {
        return end - 1;
    }
};

class TemporalGraphTest : public ::testing::TestWithParam<bool> {
protected:
    std::unique_ptr<TemporalGraph> graph;
    std::unique_ptr<FirstIndexPicker> first_picker;
    std::unique_ptr<LastIndexPicker> last_picker;

    void SetUp() override {
        // Create directed graph by default
        graph = std::make_unique<TemporalGraph>(true, GetParam());
        first_picker = std::make_unique<FirstIndexPicker>();
        last_picker = std::make_unique<LastIndexPicker>();
    }

    // Helper to create edge tuples
    static std::vector<std::tuple<int, int, int64_t>> create_edges(
        std::initializer_list<std::tuple<int, int, int64_t>> edges) {
        return edges;
    }
};

// Test empty state
TEST_P(TemporalGraphTest, EmptyStateTest) {
    EXPECT_EQ(graph->get_node_ids().size(), 0);
    EXPECT_TRUE(graph->get_edges().empty());
}

// Test basic edge addition
TEST_P(TemporalGraphTest, BasicEdgeAdditionTest) {
    auto edges = create_edges({
        {1, 2, 100},
        {2, 3, 200},
        {3, 1, 300}
    });

    graph->add_multiple_edges(edges);

    auto graph_edges = graph->get_edges();
    EXPECT_EQ(graph_edges.size(), 3);
    EXPECT_EQ(graph->get_node_ids().size(), 3);
}

TEST_P(TemporalGraphTest, MaintainSortedOrderTest) {
    // First addition
    auto edges1 = create_edges({
        {10, 20, 200},  // Out of order timestamps
        {20, 30, 100}
    });
    graph->add_multiple_edges(edges1);

    // Check first addition is sorted
    auto sorted_edges = graph->get_edges();
    EXPECT_EQ(std::get<2>(sorted_edges[0]), 100);
    EXPECT_EQ(std::get<2>(sorted_edges[1]), 200);

    // Second addition with timestamps that need to be merged
    auto edges2 = create_edges({
        {30, 40, 150},
        {40, 50, 250}
    });
    graph->add_multiple_edges(edges2);

    // Verify all timestamps are still sorted
    sorted_edges = graph->get_edges();
    EXPECT_EQ(sorted_edges.size(), 4);
    EXPECT_EQ(std::get<2>(sorted_edges[0]), 100);
    EXPECT_EQ(std::get<2>(sorted_edges[1]), 150);
    EXPECT_EQ(std::get<2>(sorted_edges[2]), 200);
    EXPECT_EQ(std::get<2>(sorted_edges[3]), 250);

    // Third addition with duplicate timestamps
    auto edges3 = create_edges({
        {50, 60, 150},
        {60, 70, 200},
        {70, 80, 175}
    });
    graph->add_multiple_edges(edges3);

    // Verify order is maintained with duplicates
    sorted_edges = graph->get_edges();
    EXPECT_EQ(sorted_edges.size(), 7);
    EXPECT_EQ(std::get<2>(sorted_edges[0]), 100);
    EXPECT_EQ(std::get<2>(sorted_edges[1]), 150);
    EXPECT_EQ(std::get<2>(sorted_edges[2]), 150);
    EXPECT_EQ(std::get<2>(sorted_edges[3]), 175);
    EXPECT_EQ(std::get<2>(sorted_edges[4]), 200);
    EXPECT_EQ(std::get<2>(sorted_edges[5]), 200);
    EXPECT_EQ(std::get<2>(sorted_edges[6]), 250);

    // Verify node timestamp groups are correct
    // Node 30 has edges at 100 (inbound) and 150 (outbound)
    EXPECT_EQ(graph->count_node_timestamps_greater_than(30, 50), 1);   // Should see 150
    EXPECT_EQ(graph->count_node_timestamps_less_than(30, 200), 1);     // Should see 100
}

// Test time window functionality
TEST_P(TemporalGraphTest, TimeWindowTest) {
    // Create graph with 100 time unit window
    graph = std::make_unique<TemporalGraph>(true, GetParam(), 100);

    // Add edges spanning the time window
    auto edges = create_edges({
        {1, 2, 100},
        {2, 3, 150},
        {3, 4, 249}  // This should cause deletion of first edge
    });

    graph->add_multiple_edges(edges);
    const auto remaining_edges = graph->get_edges();

    EXPECT_EQ(remaining_edges.size(), 2);
    EXPECT_EQ(std::get<2>(remaining_edges[0]), 150);
    EXPECT_EQ(std::get<2>(remaining_edges[1]), 249);
}

// Test edge cases and corner cases
TEST_P(TemporalGraphTest, EdgeAdditionEdgeCasesTest) {
    // Test empty edge list
    graph->add_multiple_edges({});
    EXPECT_TRUE(graph->get_edges().empty());

    // Test single edge
    auto single_edge = create_edges({{1, 2, 100}});
    graph->add_multiple_edges(single_edge);
    EXPECT_EQ(graph->get_edges().size(), 1);

    // Test duplicate timestamps
    const auto dup_time_edges = create_edges({
        {1, 2, 100},
        {2, 3, 100},
        {3, 4, 100}
    });
    graph->add_multiple_edges(dup_time_edges);
    EXPECT_EQ(graph->get_edges().size(), 4);

    // Test max int64 timestamp
    const auto max_time_edge = create_edges({
        {1, 2, INT64_MAX}
    });
    graph->add_multiple_edges(max_time_edge);
    EXPECT_EQ(graph->get_edges().size(), 5);
}

// Test deletion of nodes when all their edges are removed
TEST_P(TemporalGraphTest, NodeDeletionTest) {
    graph = std::make_unique<TemporalGraph>(true, GetParam(), 100);  // 100 time unit window

    // Add initial edges
    auto edges1 = create_edges({
        {1, 2, 100},
        {2, 3, 100},
        {3, 1, 100}
    });
    graph->add_multiple_edges(edges1);
    EXPECT_EQ(graph->get_node_ids().size(), 3);

    // Add edge that causes old edges to be deleted
    auto edges2 = create_edges({
        {4, 5, 250}  // Should cause deletion of all previous edges
    });
    graph->add_multiple_edges(edges2);

    auto remaining_nodes = graph->get_node_ids();
    EXPECT_EQ(remaining_nodes.size(), 2);  // Only nodes 4 and 5 should remain
    EXPECT_TRUE(std::find(remaining_nodes.begin(), remaining_nodes.end(), 4) != remaining_nodes.end());
    EXPECT_TRUE(std::find(remaining_nodes.begin(), remaining_nodes.end(), 5) != remaining_nodes.end());
}

// Test undirected graph behavior
TEST_P(TemporalGraphTest, UndirectedGraphEdgeAdditionTest) {
    graph = std::make_unique<TemporalGraph>(false, GetParam());  // Undirected

    auto edges = create_edges({
        {2, 1, 100},  // Should be stored as (1,2,100)
        {3, 1, 200},  // Should be stored as (1,3,200)
    });

    graph->add_multiple_edges(edges);
    const auto stored_edges = graph->get_edges();

    // Verify edges are stored with lower node ID as source
    EXPECT_EQ(std::get<0>(stored_edges[0]), 1);
    EXPECT_EQ(std::get<1>(stored_edges[0]), 2);
    EXPECT_EQ(std::get<0>(stored_edges[1]), 1);
    EXPECT_EQ(std::get<1>(stored_edges[1]), 3);
}

TEST_P(TemporalGraphTest, CountTimestampsTest) {
   // Set up a graph with carefully chosen timestamps
   auto edges = create_edges({
       {1, 2, 100},  // t0
       {2, 3, 100},  // t0 duplicate
       {1, 3, 200},  // t1
       {2, 4, 300},  // t2
       {3, 4, 300},  // t2 duplicate
       {4, 1, 400}   // t3
   });
   graph->add_multiple_edges(edges);

   // Test count_timestamps_less_than
   EXPECT_EQ(graph->count_timestamps_less_than(50), 0);   // Before first timestamp
   EXPECT_EQ(graph->count_timestamps_less_than(100), 0);  // At first timestamp
   EXPECT_EQ(graph->count_timestamps_less_than(150), 1);  // Between t0 and t1
   EXPECT_EQ(graph->count_timestamps_less_than(200), 1);  // At t1
   EXPECT_EQ(graph->count_timestamps_less_than(300), 2);  // At t2
   EXPECT_EQ(graph->count_timestamps_less_than(400), 3);  // At t3
   EXPECT_EQ(graph->count_timestamps_less_than(500), 4);  // After last timestamp

   // Test count_timestamps_greater_than
   EXPECT_EQ(graph->count_timestamps_greater_than(50), 4);   // Before first timestamp
   EXPECT_EQ(graph->count_timestamps_greater_than(100), 3);  // At first timestamp
   EXPECT_EQ(graph->count_timestamps_greater_than(150), 3);  // Between t0 and t1
   EXPECT_EQ(graph->count_timestamps_greater_than(200), 2);  // At t1
   EXPECT_EQ(graph->count_timestamps_greater_than(300), 1);  // At t2
   EXPECT_EQ(graph->count_timestamps_greater_than(400), 0);  // At t3
   EXPECT_EQ(graph->count_timestamps_greater_than(500), 0);  // After last timestamp

   // Test empty graph
   graph = std::make_unique<TemporalGraph>(true, GetParam());
   EXPECT_EQ(graph->count_timestamps_less_than(100), 0);
   EXPECT_EQ(graph->count_timestamps_greater_than(100), 0);
}

TEST_P(TemporalGraphTest, CountNodeTimestampsDirectedTest) {
    // Set up directed graph with careful node-timestamp patterns
    auto edges = create_edges({
        {1, 2, 100},  // Node 1 out: t0  | Node 2 in: t0
        {1, 3, 100},  // Node 1 out: t0  | Node 3 in: t0
        {2, 1, 200},  // Node 2 out: t1  | Node 1 in: t1
        {1, 2, 300},  // Node 1 out: t2  | Node 2 in: t2
        {3, 1, 300},  // Node 3 out: t2  | Node 1 in: t2
        {1, 4, 400}   // Node 1 out: t3  | Node 4 in: t3
    });
    graph->add_multiple_edges(edges);

    // Test outbound edges for node 1 (count_node_timestamps_greater_than)
    // Node 1's outbound edges are at: 100(x2), 300, 400
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 50), 3);    // Should see 100,300,400
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 100), 2);   // Should see 300,400
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 200), 2);   // Should see 300,400
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 300), 1);   // Should see 400
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 400), 0);   // Nothing after 400
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 500), 0);   // Nothing after 500

    // Test inbound edges for node 1 (count_node_timestamps_less_than)
    // Node 1's inbound edges are at: 200, 300
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 50), 0);     // Nothing before 50
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 200), 0);    // Nothing before 200
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 250), 1);    // Should see 200
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 400), 2);    // Should see 200,300
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 500), 2);    // Should see 200,300

    // Test node 2's timestamps
    // Node 2 outbound: 200
    // Node 2 inbound: 100, 300
    EXPECT_EQ(graph->count_node_timestamps_greater_than(2, 50), 1);   // Should see 200
    EXPECT_EQ(graph->count_node_timestamps_greater_than(2, 200), 0);  // Nothing after 200
    EXPECT_EQ(graph->count_node_timestamps_less_than(2, 400), 2);    // Should see 100,300

    // Test node with no edges
    EXPECT_EQ(graph->count_node_timestamps_greater_than(5, 100), 0);
    EXPECT_EQ(graph->count_node_timestamps_less_than(5, 100), 0);

    // Test invalid node ID
    EXPECT_EQ(graph->count_node_timestamps_greater_than(-1, 100), 0);
    EXPECT_EQ(graph->count_node_timestamps_less_than(-1, 100), 0);
}

TEST_P(TemporalGraphTest, CountNodeTimestampsUndirectedTest) {
   // Create undirected graph
   graph = std::make_unique<TemporalGraph>(false, GetParam());

   // Add edges - note that order will be normalized (smaller ID becomes source)
   auto edges = create_edges({
       {2, 1, 100},  // Will be stored as (1,2,100)
       {3, 1, 100},  // Will be stored as (1,3,100)
       {1, 2, 200},  // Will be stored as (1,2,200)
       {4, 1, 300},  // Will be stored as (1,4,300)
       {1, 3, 300}   // Will be stored as (1,3,300)
   });
   graph->add_multiple_edges(edges);

   // Test timestamps for node 1
   EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 50), 3);   // Before first
   EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 100), 2);  // At first
   EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 200), 1);  // At middle
   EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 300), 0);  // At last
   EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 400), 0);  // After last

   EXPECT_EQ(graph->count_node_timestamps_less_than(1, 50), 0);    // Before first
   EXPECT_EQ(graph->count_node_timestamps_less_than(1, 100), 0);   // At first
   EXPECT_EQ(graph->count_node_timestamps_less_than(1, 150), 1);   // After first
   EXPECT_EQ(graph->count_node_timestamps_less_than(1, 400), 3);   // After last

   // Test edge target node (node 2)
   EXPECT_EQ(graph->count_node_timestamps_greater_than(2, 50), 2);   // Should see both t0 and t1
   EXPECT_EQ(graph->count_node_timestamps_less_than(2, 250), 2);     // Should see both t0 and t1
}

TEST_P(TemporalGraphTest, CountNodeTimestampsDuplicatesTest) {
    // Test handling of duplicate timestamps for a node
    auto edges = create_edges({
        {1, 2, 100},  // t0 outbound from node 1
        {1, 3, 100},  // t0 duplicate outbound from node 1
        {1, 4, 100},  // t0 triplicate outbound from node 1
        {2, 1, 200},  // t1 inbound to node 1
        {3, 1, 200},  // t1 duplicate inbound to node 1
    });
    graph->add_multiple_edges(edges);

    // Test outbound timestamps for node 1
    // Node 1 has outbound edges only at t0 (100)
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 50), 1);    // Should see only t0
    EXPECT_EQ(graph->count_node_timestamps_greater_than(1, 100), 0);   // Nothing after t0

    // Test inbound timestamps for node 1
    // Node 1 has inbound edges only at t1 (200)
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 250), 1);     // Should see only t1
    EXPECT_EQ(graph->count_node_timestamps_less_than(1, 150), 0);     // Nothing before t1

    // Add test for target nodes
    EXPECT_EQ(graph->count_node_timestamps_less_than(2, 150), 1);     // Node 2 receives at t0
    EXPECT_EQ(graph->count_node_timestamps_greater_than(2, 50), 1);   // Node 2 sends at t1
}

TEST_P(TemporalGraphTest, GetEdgeAtTest) {
    // Set up test graph with carefully structured timestamps
    const auto edges = create_edges({
        {10, 20, 100},  // Group 0: timestamp 100
        {30, 40, 100},
        {50, 60, 200},  // Group 1: timestamp 200
        {70, 80, 300},  // Group 2: timestamp 300
        {90, 100, 300},
        {110, 120, 400}  // Group 3: timestamp 400
    });
    graph->add_multiple_edges(edges);

    // Test forward direction (looking for timestamps > given)

    // Test with timestamp = -1 (no constraint)
    auto [src1, tgt1, ts1] = graph->get_edge_at(*first_picker, -1, true);
    EXPECT_EQ(ts1, 100);  // Should select from first group

    auto [src2, tgt2, ts2] = graph->get_edge_at(*last_picker, -1, true);
    EXPECT_EQ(ts2, 400);  // Should select from last group

    // Test with timestamp constraints
    auto [src3, tgt3, ts3] = graph->get_edge_at(*first_picker, 100, true);
    EXPECT_EQ(ts3, 200);  // Should select first group after 100

    auto [src4, tgt4, ts4] = graph->get_edge_at(*first_picker, 300, true);
    EXPECT_EQ(ts4, 400);  // Should select first group after 300

    // Test backward direction (looking for timestamps < given)
    auto [src5, tgt5, ts5] = graph->get_edge_at(*first_picker, 400, false);
    EXPECT_EQ(ts5, 100);  // Should select first group before 400

    auto [src6, tgt6, ts6] = graph->get_edge_at(*last_picker, 250, false);
    EXPECT_EQ(ts6, 200);  // Should select latest group before 250

    // Test edge cases
    // No groups after timestamp
    auto [src7, tgt7, ts7] = graph->get_edge_at(*first_picker, 500, true);
    EXPECT_EQ(ts7, -1);  // Should return -1 when no valid groups

    // No groups before timestamp
    auto [src8, tgt8, ts8] = graph->get_edge_at(*first_picker, 50, false);
    EXPECT_EQ(ts8, -1);

    // Test with empty graph
    graph = std::make_unique<TemporalGraph>(true, GetParam());
    auto [src9, tgt9, ts9] = graph->get_edge_at(*first_picker, 100, true);
    EXPECT_EQ(ts9, -1);
}

TEST_P(TemporalGraphTest, GetEdgeAtDuplicateTimestampsTest) {
    // Setup graph with multiple edges in same timestamp groups
    auto edges = create_edges({
        {10, 20, 100},  // Group 0: three edges at 100
        {30, 40, 100},
        {50, 60, 100},
        {70, 80, 200},  // Group 1: two edges at 200
        {90, 100, 200},
        {110, 120, 300}  // Group 2: single edge at 300
    });
    graph->add_multiple_edges(edges);

    // Test forward selection
    auto [src1, tgt1, ts1] = graph->get_edge_at(*first_picker, 50, true);
    EXPECT_EQ(ts1, 100);
    EXPECT_TRUE((src1 == 10 && tgt1 == 20) ||
                (src1 == 30 && tgt1 == 40) ||
                (src1 == 50 && tgt1 == 60));  // Should be one of the t=100 edges

    // Test backward selection
    auto [src2, tgt2, ts2] = graph->get_edge_at(*first_picker, 250, false);
    EXPECT_EQ(ts2, 100);
    EXPECT_TRUE((src2 == 10 && tgt2 == 20) ||
                (src2 == 30 && tgt2 == 40) ||
                (src2 == 50 && tgt2 == 60));  // Should be one of the t=100 edges
}

TEST_P(TemporalGraphTest, GetEdgeAtBoundaryConditionsTest) {
    // Test exact timestamp boundaries
    auto edges = create_edges({
        {10, 20, 100},
        {30, 40, 200},
        {50, 60, 300}
    });
    graph->add_multiple_edges(edges);

    // Forward direction
    auto [src1, tgt1, ts1] = graph->get_edge_at(*first_picker, 100, true);
    EXPECT_EQ(ts1, 200);  // Should get next timestamp

    auto [src2, tgt2, ts2] = graph->get_edge_at(*first_picker, 300, true);
    EXPECT_EQ(ts2, -1);   // No timestamps after 300

    // Backward direction
    auto [src3, tgt3, ts3] = graph->get_edge_at(*first_picker, 200, false);
    EXPECT_EQ(ts3, 100);  // Should get previous timestamp

    auto [src4, tgt4, ts4] = graph->get_edge_at(*first_picker, 100, false);
    EXPECT_EQ(ts4, -1);   // No timestamps before 100
}

TEST_P(TemporalGraphTest, GetEdgeAtRandomSelectionTest) {
    // Test that we can get different edges from same group
    const auto edges = create_edges({
        {1, 2, 100},
        {3, 4, 100},
        {5, 6, 100}
    });
    graph->add_multiple_edges(edges);

    // Make multiple selections and verify we can get different edges
    std::set<std::pair<int, int>> seen_edges;
    constexpr int NUM_TRIES = 50;

    for (int i = 0; i < NUM_TRIES; i++) {
        auto [src, tgt, ts] = graph->get_edge_at(*first_picker, 50, true);
        EXPECT_EQ(ts, 100);
        seen_edges.insert({src, tgt});
    }

    // We should see more than one edge (due to random selection within group)
    EXPECT_GT(seen_edges.size(), 1);
}

#ifdef HAS_CUDA
INSTANTIATE_TEST_SUITE_P(
    CPUAndGPU,
    TemporalGraphTest,
    ::testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "GPU" : "CPU";
    }
);
#else
INSTANTIATE_TEST_SUITE_P(
    CPUOnly,
    TemporalGraphTest,
    ::testing::Values(false),
    [](const testing::TestParamInfo<bool>& info) {
        return "CPU";
    }
);
#endif
