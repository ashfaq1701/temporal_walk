#include <gtest/gtest.h>

#include "test_utils.h"
#include "../src/core/TemporalWalk.h"

constexpr int TEST_NODE_ID = 45965;
constexpr int LEN_WALK = 20;
constexpr int NUM_WALKS = 100;

class EmptyTemporalWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(NUM_WALKS, LEN_WALK, RandomPickerType::Linear);
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

class FilledTemporalWalkTest : public ::testing::Test {
protected:
    FilledTemporalWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(NUM_WALKS, LEN_WALK, RandomPickerType::Linear);
        temporal_walk->add_multiple_edges(sample_edges);
    }

    std::vector<EdgeInfo> sample_edges;
    std::unique_ptr<TemporalWalk> temporal_walk;
};

// Test the constructor of TemporalWalk to ensure it initializes correctly.
TEST_F(EmptyTemporalWalkTest, ConstructorTest) {
    EXPECT_NO_THROW(temporal_walk = std::make_unique<TemporalWalk>(NUM_WALKS, LEN_WALK, RandomPickerType::Uniform));
    EXPECT_EQ(temporal_walk->get_len_walk(), LEN_WALK);
    EXPECT_EQ(temporal_walk->get_node_count(), 0); // Assuming initial node count is 0
}


// Test adding an edge to the TemporalWalk when it's empty.
TEST_F(EmptyTemporalWalkTest, AddEdgeTest) {
    temporal_walk->add_edge(1, 2, 100);
    EXPECT_EQ(temporal_walk->get_edge_count(), 1);
    EXPECT_EQ(temporal_walk->get_node_count(), 2);
}

// Test to check if a specific node ID is present in the filled TemporalWalk.
TEST_F(FilledTemporalWalkTest, TestNodeFoundTest) {
    const auto nodes = temporal_walk->get_node_ids();
    const auto it = std::find(nodes.begin(), nodes.end(), TEST_NODE_ID);
    EXPECT_NE(it, nodes.end());
}

// Test that the number of random walks generated matches the expected count and checks that no walk exceeds its length.
TEST_F(FilledTemporalWalkTest, WalkCountAndLensTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::Random, TEST_NODE_ID);
    EXPECT_EQ(walks.size(), NUM_WALKS);

    for (const auto& walk : walks) {
        EXPECT_LE(walk.size(), LEN_WALK) << "A walk exceeds the maximum length of " << LEN_WALK;
        EXPECT_GT(walk.size(), 0);
    }
}

// Test that all walks starting from a specific node begin with that node.
TEST_F(FilledTemporalWalkTest, WalkStartTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::Begin, TEST_NODE_ID);
    for (const auto& walk : walks) {
        EXPECT_EQ(walk[0].node, TEST_NODE_ID);
    }
}

// Test that all walks ending at a specific node conclude with that node.
TEST_F(FilledTemporalWalkTest, WalkEndTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::End, TEST_NODE_ID);
    for (const auto& walk : walks) {
        EXPECT_EQ(walk.back().node, TEST_NODE_ID);
    }
}

// Test to verify that the timestamps in each walk are strictly increasing.
TEST_F(FilledTemporalWalkTest, WalkIncreasingTimestampTest) {
    const auto walks = temporal_walk->get_random_walks_with_times(WalkStartAt::Random, TEST_NODE_ID);

    for (const auto& walk : walks) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify random walks for selected nodes and their properties.
TEST_F(FilledTemporalWalkTest, CheckWalksForNodes) {
    constexpr int num_selected_walks = 100;

    const auto nodes = temporal_walk->get_node_ids();
    const auto selected_nodes = std::vector<int>(nodes.begin(), nodes.begin() + num_selected_walks);

    const auto walks_for_nodes = temporal_walk->get_random_walks_for_nodes_with_times(WalkStartAt::Random, selected_nodes);
    EXPECT_EQ(walks_for_nodes.size(), num_selected_walks);

    for (const auto& node : selected_nodes) {
        auto it = walks_for_nodes.find(node);
        EXPECT_NE(it, walks_for_nodes.end()) << "Node " << node << " is not present in walks_for_nodes.";
        EXPECT_EQ(it->second.size(), NUM_WALKS) << "Node " << node << " does not have the expected number of walks.";
    }

    // Test that each walk for each node is strictly increasing in time.
    for (const auto& node : selected_nodes) {
        auto walks = walks_for_nodes.at(node);

        for (const auto& walk : walks) {
            for (size_t i = 1; i < walk.size(); ++i) {
                EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                    << "Timestamps are not strictly increasing in walk: "
                    << i << " with node: " << walk[i].node
                    << ", previous node: " << walk[i - 1].node;
            }
        }
    }
}
