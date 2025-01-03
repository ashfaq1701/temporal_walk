#include <gtest/gtest.h>
#include <cmath>

#include "test_utils.h"
#include "../src/core/TemporalWalk.h"
#include "../src/random/ExponentialRandomPicker.h"
#include "../src/random/LinearRandomPicker.h"

constexpr int TEST_NODE_ID = 42;
constexpr int MAX_WALK_LEN = 20;
constexpr int64_t MAX_TIME_CAPACITY = 5;

constexpr int RANDOM_START = 0;
constexpr int RANDOM_END = 10000;
constexpr int RANDOM_NUM_SAMPLES = 1000000;

constexpr RandomPickerType exponential_picker_type = RandomPickerType::Exponential;
constexpr RandomPickerType linear_picker_type = RandomPickerType::Linear;

class RandomPickerTest : public ::testing::Test {
protected:

    LinearRandomPicker linear_picker;
    ExponentialRandomPicker exp_picker;

    double compute_average_picks(const bool use_exponential, const bool prioritize_end) {
        double sum = 0;
        for (int i = 0; i < RANDOM_NUM_SAMPLES; i++) {
            const int pick = use_exponential ?
                                 exp_picker.pick_random(RANDOM_START, RANDOM_END, prioritize_end) :
                                 linear_picker.pick_random(RANDOM_START, RANDOM_END, prioritize_end);
            sum += pick;
        }
        return sum / RANDOM_NUM_SAMPLES;
    }
};

class EmptyTemporalWalkTest : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true);
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

class EmptyTemporalWalkTestWithMaxCapacity : public ::testing::Test {
protected:
    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true, MAX_TIME_CAPACITY);
    }

    std::unique_ptr<TemporalWalk> temporal_walk;
};

class FilledDirectedTemporalWalkTest : public ::testing::Test {
protected:
    FilledDirectedTemporalWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(true);
        temporal_walk->add_multiple_edges(sample_edges);
    }

    std::vector<EdgeInfo> sample_edges;
    std::unique_ptr<TemporalWalk> temporal_walk;
};

class FilledUndirectedTemporalWalkTest : public ::testing::Test {
protected:
    FilledUndirectedTemporalWalkTest() {
        sample_edges = read_edges_from_csv("../../../data/sample_data.csv");
    }

    void SetUp() override {
        temporal_walk = std::make_unique<TemporalWalk>(false);
        temporal_walk->add_multiple_edges(sample_edges);
    }

    std::vector<EdgeInfo> sample_edges;
    std::unique_ptr<TemporalWalk> temporal_walk;
};

// Test that prioritize_end=true gives higher average than prioritize_end=false for both pickers
TEST_F(RandomPickerTest, PrioritizeEndGivesHigherAverage) {
    // For Linear Picker
    const double linear_end_prioritized = compute_average_picks(false, true);
    const double linear_start_prioritized = compute_average_picks(false, false);
    EXPECT_GT(linear_end_prioritized, linear_start_prioritized)
        << "Linear picker with prioritize_end=true should give higher average ("
        << linear_end_prioritized << ") than prioritize_end=false ("
        << linear_start_prioritized << ")";

    // For Exponential Picker
    const double exp_end_prioritized = compute_average_picks(true, true);
    const double exp_start_prioritized = compute_average_picks(true, false);
    EXPECT_GT(exp_end_prioritized, exp_start_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than prioritize_end=false ("
        << exp_start_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing end
TEST_F(RandomPickerTest, ExponentialMoreExtremeForEnd) {
    const double linear_end_prioritized = compute_average_picks(false, true);
    const double exp_end_prioritized = compute_average_picks(true, true);

    EXPECT_GT(exp_end_prioritized, linear_end_prioritized)
        << "Exponential picker with prioritize_end=true should give higher average ("
        << exp_end_prioritized << ") than Linear picker ("
        << linear_end_prioritized << ")";
}

// Test that exponential picker is more extreme than linear picker when prioritizing start
TEST_F(RandomPickerTest, ExponentialMoreExtremeForStart) {
    const double linear_start_prioritized = compute_average_picks(false, false);
    const double exp_start_prioritized = compute_average_picks(true, false);

    EXPECT_LT(exp_start_prioritized, linear_start_prioritized)
        << "Exponential picker with prioritize_end=false should give lower average ("
        << exp_start_prioritized << ") than Linear picker ("
        << linear_start_prioritized << ")";
}

// Test that output is always within bounds
TEST_F(RandomPickerTest, BoundsTest) {
    const int start = 5;
    const int end = 10;
    const int num_tests = 1000;

    for (int i = 0; i < num_tests; i++) {
        int linear_result = linear_picker.pick_random(start, end, true);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        linear_result = linear_picker.pick_random(start, end, false);
        EXPECT_GE(linear_result, start);
        EXPECT_LT(linear_result, end);

        int exp_result = exp_picker.pick_random(start, end, true);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);

        exp_result = exp_picker.pick_random(start, end, false);
        EXPECT_GE(exp_result, start);
        EXPECT_LT(exp_result, end);
    }
}

// Test single-element range always returns that element
TEST_F(RandomPickerTest, SingleElementRangeTest) {
    constexpr int start = 5;
    constexpr int end = 6;  // Range of size 1

    // Should always return start for both true and false prioritize_end
    EXPECT_EQ(linear_picker.pick_random(start, end, true), start);
    EXPECT_EQ(linear_picker.pick_random(start, end, false), start);
    EXPECT_EQ(exp_picker.pick_random(start, end, true), start);
    EXPECT_EQ(exp_picker.pick_random(start, end, false), start);
}

// Test probabilities more deterministically for linear random picker and two elements.
TEST_F(RandomPickerTest, TwoElementRangeDistributionTestForLinearRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    const int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = linear_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = linear_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    // For linear picker with size 2:
    // When prioritize_end=true:
    //   weight(0) = 1, weight(1) = 2, total_weight = 3
    //   prob(0) = 1/3, prob(1) = 2/3
    // When prioritize_end=false:
    //   weight(0) = 2, weight(1) = 1, total_weight = 3
    //   prob(0) = 2/3, prob(1) = 1/3

    constexpr double expected_prob_end = 2.0 / 3.0;    // probability of getting 1 when prioritizing end
    constexpr double expected_prob_start = 1.0 / 3.0;  // probability of getting 1 when prioritizing start

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    // Allow for some statistical variation
    constexpr double tolerance = 0.02;  // 2% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

// Test probabilities more deterministically for exponential random picker and two elements.
TEST_F(RandomPickerTest, TwoElementRangeDistributionTestForExponentialRandomPicker) {
    const int start = 0;
    const int end = 2;
    int count_ones_end_prioritized = 0;
    int count_ones_start_prioritized = 0;
    constexpr int num_trials = RANDOM_NUM_SAMPLES;

    // Run trials
    for (int i = 0; i < num_trials; i++) {
        // Test prioritize_end=true
        int result_end = exp_picker.pick_random(start, end, true);
        if (result_end == 1) {
            count_ones_end_prioritized++;
        }

        // Test prioritize_end=false (separate trial)
        int result_start = exp_picker.pick_random(start, end, false);
        if (result_start == 1) {
            count_ones_start_prioritized++;
        }
    }

    // For exponential picker with size 2:
    // When prioritize_end=true:
    //   P(0) = (e-1)/(e^2 - 1)
    //   P(1) = (e-1)e/(e^2 - 1)
    const double e = std::exp(1.0);
    const double e_squared = e * e;
    const double expected_prob_end = (e - 1.0) * e / (e_squared - 1.0);  // probability of getting 1

    // When prioritize_end=false:
    //   P(0) = (e-1)e/(e^2 - 1)
    //   P(1) = (e-1)/(e^2 - 1)
    const double expected_prob_start = (e - 1.0) / (e_squared - 1.0);  // probability of getting 1

    const double actual_prob_end = static_cast<double>(count_ones_end_prioritized) / num_trials;
    const double actual_prob_start = static_cast<double>(count_ones_start_prioritized) / num_trials;

    // Allow for some statistical variation
    constexpr double tolerance = 0.005;  // 0.5% tolerance

    EXPECT_NEAR(actual_prob_end, expected_prob_end, tolerance)
        << "When prioritizing end, probability of picking 1 should be approximately "
        << expected_prob_end << " but got " << actual_prob_end;

    EXPECT_NEAR(actual_prob_start, expected_prob_start, tolerance)
        << "When prioritizing start, probability of picking 1 should be approximately "
        << expected_prob_start << " but got " << actual_prob_start;
}

// Test the constructor of TemporalWalk to ensure it initializes correctly.
TEST_F(EmptyTemporalWalkTest, ConstructorTest) {
    EXPECT_NO_THROW(temporal_walk = std::make_unique<TemporalWalk>(true));
    EXPECT_EQ(temporal_walk->get_node_count(), 0); // Assuming initial node count is 0
}


// Test adding an edge to the TemporalWalk when it's empty.
TEST_F(EmptyTemporalWalkTest, AddEdgeTest) {
    temporal_walk->add_multiple_edges({
        EdgeInfo {1, 2, 100},
        EdgeInfo {2, 3, 101},
        EdgeInfo{7, 8, 102},
        EdgeInfo{1, 7, 103},
        EdgeInfo{3, 2, 103},
        EdgeInfo{10, 11, 104}
    });

    EXPECT_EQ(temporal_walk->get_edge_count(), 6);
    EXPECT_EQ(temporal_walk->get_node_count(), 7);
}

// When later edges are added than the allowed max time capacity, older edges are automatically deleted.
TEST_F(EmptyTemporalWalkTestWithMaxCapacity, WhenMaxTimeCapacityExceedsEdgesAreDeletedAutomatically) {
    temporal_walk->add_multiple_edges({
        EdgeInfo{ 0, 2, 1 },
        EdgeInfo{ 2, 3, 3 },
        EdgeInfo{ 1, 9, 2 },
        EdgeInfo{ 2, 4, 3 },
        EdgeInfo{ 2, 4, 1 },
        EdgeInfo{ 1, 5, 4 }
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 7);
    EXPECT_EQ(temporal_walk->get_edge_count(), 6);

    temporal_walk->add_multiple_edges({
        EdgeInfo{ 5, 6, 4 },
        EdgeInfo{ 2, 5, 4 },
        EdgeInfo{ 4, 3, 5 },
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 8);
    EXPECT_EQ(temporal_walk->get_edge_count(), 9);

    temporal_walk->add_multiple_edges({
        EdgeInfo{ 1, 7, 6 }
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 8);
    EXPECT_EQ(temporal_walk->get_edge_count(), 8);

    temporal_walk->add_multiple_edges({
        EdgeInfo{ 1, 5, 7 },
        EdgeInfo{ 4, 7, 8 }
    });

    EXPECT_EQ(temporal_walk->get_node_count(), 7);
    EXPECT_EQ(temporal_walk->get_edge_count(), 7);
}

// Test to check if a specific node ID is present in the filled TemporalWalk.
TEST_F(FilledDirectedTemporalWalkTest, TestNodeFoundTest) {
    const auto nodes = temporal_walk->get_node_ids();
    const auto it = std::find(nodes.begin(), nodes.end(), TEST_NODE_ID);
    EXPECT_NE(it, nodes.end());
}

// Test that the number of random walks generated matches the expected count and checks that no walk exceeds its length.
// Also test that the system can sample walks of length more than 1.
TEST_F(FilledDirectedTemporalWalkTest, WalkCountAndLensTest) {
    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10);

    int total_walk_lens = 0;

    for (const auto& walk : walks) {
        EXPECT_LE(walk.size(), MAX_WALK_LEN) << "A walk exceeds the maximum length of " << MAX_WALK_LEN;
        EXPECT_GT(walk.size(), 0);

        total_walk_lens += static_cast<int>(walk.size());
    }

    auto average_walk_len = static_cast<float>(total_walk_lens) / static_cast<float>(walks.size());
    EXPECT_GT(average_walk_len, 1) << "System could not sample any walk of length more than 1";
}

// Test to verify that the timestamps in each walk are strictly increasing in directed graphs.
TEST_F(FilledDirectedTemporalWalkTest, WalkIncreasingTimestampTest) {
    const auto walks_forward = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10);

    for (const auto& walk : walks_forward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (const auto& walk : walks_backward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}

// Test to verify that the timestamps in each walk are strictly increasing in undirected graphs.
TEST_F(FilledUndirectedTemporalWalkTest, WalkIncreasingTimestampTest) {
    const auto walks = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10);

    for (const auto& walk : walks) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }

    const auto walks_backward = temporal_walk->get_random_walks_and_times_for_all_nodes(MAX_WALK_LEN, &linear_picker_type, 10, nullptr, WalkDirection::Backward_In_Time);
    for (const auto& walk : walks_backward) {
        for (size_t i = 1; i < walk.size(); ++i) {
            EXPECT_GT(walk[i].timestamp, walk[i - 1].timestamp)
                << "Timestamps are not strictly increasing in walk: "
                << i << " with node: " << walk[i].node
                << ", previous node: " << walk[i - 1].node;
        }
    }
}
