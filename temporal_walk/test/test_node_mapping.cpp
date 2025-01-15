#include <gtest/gtest.h>
#include "../src/data/NodeMapping.h"

class NodeMappingTest : public ::testing::Test {
protected:
    NodeMapping mapping;
    EdgeData edges;

    // Helper to verify bidirectional mapping
    void verify_mapping(int sparse_id, int expected_dense_idx) const {
        EXPECT_EQ(mapping.to_dense(sparse_id), expected_dense_idx);
        if (expected_dense_idx != -1) {
            EXPECT_EQ(mapping.to_sparse(expected_dense_idx), sparse_id);
        }
    }
};

// Test empty state
TEST_F(NodeMappingTest, EmptyStateTest) {
    EXPECT_EQ(mapping.size(), 0);
    EXPECT_EQ(mapping.active_size(), 0);
    EXPECT_TRUE(mapping.get_active_node_ids().empty());
    EXPECT_TRUE(mapping.get_all_sparse_ids().empty());

    // Test invalid mappings in empty state
    EXPECT_EQ(mapping.to_dense(0), -1);
    EXPECT_EQ(mapping.to_dense(-1), -1);
    EXPECT_EQ(mapping.to_sparse(0), -1);
    EXPECT_FALSE(mapping.has_node(0));
}

// Test basic update functionality
TEST_F(NodeMappingTest, BasicUpdateTest) {
    edges.push_back(10, 20, 100);
    edges.push_back(20, 30, 200);
    mapping.update(edges, 0, edges.size());

    // Verify sizes
    EXPECT_EQ(mapping.size(), 3);  // 3 unique nodes
    EXPECT_EQ(mapping.active_size(), 3);  // All nodes active

    // Verify mappings
    verify_mapping(10, 0);  // First node gets dense index 0
    verify_mapping(20, 1);  // Second node gets dense index 1
    verify_mapping(30, 2);  // Third node gets dense index 2

    // Verify node existence
    EXPECT_TRUE(mapping.has_node(10));
    EXPECT_TRUE(mapping.has_node(20));
    EXPECT_TRUE(mapping.has_node(30));
    EXPECT_FALSE(mapping.has_node(15));  // Non-existent node
}

// Test handling of gaps in sparse IDs
TEST_F(NodeMappingTest, SparseGapsTest) {
    edges.push_back(10, 50, 100);  // Gap between 10 and 50
    mapping.update(edges, 0, edges.size());

    // Verify size handling with gaps
    EXPECT_EQ(mapping.size(), 2);  // Only 2 actual nodes
    EXPECT_GE(mapping.sparse_to_dense.size(), 51);  // But space for all up to 50

    // Verify mappings
    verify_mapping(10, 0);
    verify_mapping(50, 1);

    // Verify nodes in gap don't exist
    for (int i = 11; i < 50; i++) {
        EXPECT_FALSE(mapping.has_node(i));
        EXPECT_EQ(mapping.to_dense(i), -1);
    }
}

// Test incremental updates
TEST_F(NodeMappingTest, IncrementalUpdateTest) {
    // First update
    edges.push_back(10, 20, 100);
    mapping.update(edges, 0, 1);

    verify_mapping(10, 0);
    verify_mapping(20, 1);

    // Second update with new nodes
    edges.push_back(30, 40, 200);
    mapping.update(edges, 1, 2);

    verify_mapping(30, 2);
    verify_mapping(40, 3);

    // Third update with existing nodes
    edges.push_back(20, 30, 300);  // Both nodes already exist
    mapping.update(edges, 2, 3);

    EXPECT_EQ(mapping.size(), 4);  // No new nodes added
}

// Test node deletion
TEST_F(NodeMappingTest, NodeDeletionTest) {
    edges.push_back(10, 20, 100);
    edges.push_back(20, 30, 200);
    mapping.update(edges, 0, edges.size());

    // Delete node 20
    mapping.mark_node_deleted(20);

    // Verify counts
    EXPECT_EQ(mapping.size(), 3);        // Total size unchanged
    EXPECT_EQ(mapping.active_size(), 2);  // But one less active node

    // Verify active nodes list
    auto active_nodes = mapping.get_active_node_ids();
    EXPECT_EQ(active_nodes.size(), 2);
    EXPECT_TRUE(std::find(active_nodes.begin(), active_nodes.end(), 10) != active_nodes.end());
    EXPECT_TRUE(std::find(active_nodes.begin(), active_nodes.end(), 30) != active_nodes.end());
    EXPECT_TRUE(std::find(active_nodes.begin(), active_nodes.end(), 20) == active_nodes.end());

    // Mapping should still work for deleted nodes
    EXPECT_NE(mapping.to_dense(20), -1);
}

// Test edge cases and invalid inputs
TEST_F(NodeMappingTest, EdgeCasesTest) {
    // Test with negative IDs
    edges.push_back(-1, -2, 100);
    mapping.update(edges, 0, 1);
    EXPECT_EQ(mapping.to_dense(-1), -1);  // Should not map negative IDs
    EXPECT_EQ(mapping.to_dense(-2), -1);

    // Test with very large sparse ID
    edges.clear();
    edges.push_back(1000000, 1, 100);
    mapping.update(edges, 0, 1);
    verify_mapping(1000000, 0);
    verify_mapping(1, 1);

    // Test invalid dense indices
    EXPECT_EQ(mapping.to_sparse(-1), -1);
    EXPECT_EQ(mapping.to_sparse(1000000), -1);

    // Test marking non-existent node as deleted
    mapping.mark_node_deleted(999);  // Should not crash

    // Test empty range update
    mapping.update(edges, 0, 0);  // Should handle empty range gracefully
}

// Test reservation and clear
TEST_F(NodeMappingTest, ReservationAndClearTest) {
    mapping.reserve(100);

    edges.push_back(10, 20, 100);
    mapping.update(edges, 0, 1);

    mapping.clear();
    EXPECT_EQ(mapping.size(), 0);
    EXPECT_EQ(mapping.active_size(), 0);
    EXPECT_FALSE(mapping.has_node(10));
}
