#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "../src/core/TemporalWalk.h"
#include <stdexcept>

namespace py = pybind11;

constexpr int DEFAULT_WALK_FILL_VALUE = -1;

RandomPickerType picker_type_from_string(const std::string& picker_type_str) {
    if (picker_type_str == "Uniform") {
        return RandomPickerType::Uniform;
    } else if (picker_type_str == "Linear") {
        return RandomPickerType::Linear;
    } else if (picker_type_str == "Exponential") {
        return RandomPickerType::Exponential;
    } else {
        throw std::invalid_argument("Invalid picker type: " + picker_type_str);
    }
}

WalkStartAt walk_start_at_from_string(const std::string& walk_start_at_str) {
    if (walk_start_at_str == "Begin") {
        return WalkStartAt::Begin;
    } else if (walk_start_at_str == "End") {
        return WalkStartAt::End;
    } else if (walk_start_at_str == "Random") {
        return WalkStartAt::Random;
    } else {
        throw std::invalid_argument("Invalid walk start at: " + walk_start_at_str);
    }
}

PYBIND11_MODULE(_temporal_walk, m) {
    py::class_<TemporalWalk>(m, "TemporalWalk")
        .def(py::init([](int num_walks, int len_walk, const std::string& picker_type_str, const std::optional<int64_t> max_time_capacity) {
            RandomPickerType picker_type = picker_type_from_string(picker_type_str);
            return std::make_unique<TemporalWalk>(num_walks, len_walk, picker_type, max_time_capacity.value_or(-1));
        }),
        py::arg("num_walks"),
        py::arg("len_walk"),
        py::arg("picker_type"),
        py::arg("max_time_capacity") = py::none())
        .def("add_multiple_edges", [](TemporalWalk& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos) {
            std::vector<EdgeInfo> edges;

            for (const auto& edge_info : edge_infos) {
                int u = std::get<0>(edge_info);
                int i = std::get<1>(edge_info);
                int64_t t = std::get<2>(edge_info);
                edges.emplace_back(EdgeInfo {u, i, t});
            }

            tw.add_multiple_edges(edges);
        },
        R"(
        Adds multiple directed edges to the temporal graph based on the provided vector of tuples.

        Parameters:
        - edge_infos (List[Tuple[int, int, int64_t]]): A list of tuples, each containing (source node, destination node, timestamp).
        )")
        .def("get_random_walks", [](TemporalWalk& tw, const std::string& walk_start_at_str, const int end_node = -1, const int fill_value = DEFAULT_WALK_FILL_VALUE) {
            const WalkStartAt walk_start_at = walk_start_at_from_string(walk_start_at_str);
            const auto walks = tw.get_random_walks(walk_start_at, end_node);
            const int num_walks = static_cast<int>(walks.size());
            const int len_walk = tw.get_len_walk();

            py::array_t<int> py_walks({num_walks, len_walk});
            auto py_walks_mutable = py_walks.mutable_unchecked<2>();

            // Pad walks with -1 for uneven walks
            for (int i = 0; i < num_walks; ++i) {
                const auto& walk = walks[i];
                const int walk_size = static_cast<int>(walk.size());

                std::copy(walk.begin(), walk.end(), py_walks_mutable.mutable_data(i, 0));
                std::fill(py_walks_mutable.mutable_data(i, walk_size), py_walks_mutable.mutable_data(i, len_walk), fill_value);
            }

            return py_walks;
        },
        R"(
        Generates random walks from the temporal graph.

        Parameters:
        - walk_start_at (str): The starting point for the walks ("Begin", "End", "Random").
        - end_node (int, optional): An optional end node ID to start or end the walks. Default is -1 (random).
        - fill_value (int, optional): The value used to pad the walks. Default is 0.

        Returns:
        np.ndarray: A 2D NumPy array containing the generated random walks, padded where necessary.
        )", py::return_value_policy::move)
        .def("get_random_walks_for_nodes", [](TemporalWalk& tw, const std::string& walk_start_at_str, const std::vector<int>& end_nodes, const int fill_value = DEFAULT_WALK_FILL_VALUE) {
            const WalkStartAt walk_start_at = walk_start_at_from_string(walk_start_at_str);
            auto walks_for_nodes = tw.get_random_walks_for_nodes(walk_start_at, end_nodes);
            const int len_walk = tw.get_len_walk();

            py::dict py_walks_dict;  // Create a Python dictionary

            for (int node : end_nodes) {
                const auto& walks = walks_for_nodes[node];
                const int num_walks = static_cast<int>(walks.size());

                py::array_t<int> py_walks({num_walks, len_walk});
                auto py_walks_mutable = py_walks.mutable_unchecked<2>();

                // Pad walks with -1 for uneven walks
                for (int i = 0; i < num_walks; ++i) {
                    const auto& walk = walks[i];
                    const int walk_size = static_cast<int>(walk.size());

                    std::copy(walk.begin(), walk.end(), py_walks_mutable.mutable_data(i, 0));
                    std::fill(py_walks_mutable.mutable_data(i, walk_size), py_walks_mutable.mutable_data(i, len_walk), fill_value);
                }

                py_walks_dict[py::cast(node)] = py_walks;
            }

            return py_walks_dict;
        },
        R"(
        Generates random walks for multiple specified nodes in the temporal graph.

        Parameters:
        - walk_start_at (str): The starting point for the walks ("Begin", "End", "Random").
        - end_nodes (List[int]): A list of node IDs for which the walks will be generated.
        - fill_value (int, optional): The value used to pad the walks. Default is 0.

        Returns:
        Dict[int, np.ndarray]: A dictionary mapping node IDs to their corresponding random walks.
        )", py::return_value_policy::move)
            .def("get_random_walks_with_times", [](TemporalWalk& tw, const std::string& walk_start_at_str, const int end_node = -1) {
            const WalkStartAt walk_start_at = walk_start_at_from_string(walk_start_at_str);
            const auto walks_with_times = tw.get_random_walks_with_times(walk_start_at, end_node);

            // Convert the result to a Python list of lists of tuples
            std::vector<std::vector<std::tuple<int, int64_t>>> py_walks;
            for (const auto& walk : walks_with_times) {
                std::vector<std::tuple<int, int64_t>> py_walk;
                for (const auto& [node, timestamp] : walk) {
                    py_walk.emplace_back(node, timestamp); // NOLINT(*-inefficient-vector-operation)
                }
                py_walks.emplace_back(std::move(py_walk));
            }

            return py_walks;
        },
        R"(
        Generates random walks with timestamps from the temporal graph.

        Parameters:
        - walk_start_at (str): The starting point for the walks ("Begin", "End", "Random").
        - end_node (int, optional): An optional end node ID to start or end the walks. Default is -1 (random).

        Returns:
        List[List[Tuple[int, int64_t]]]: A list of random walks, each walk is a list of tuples containing (node_id, timestamp).
        )", py::return_value_policy::move)

        .def("get_random_walks_for_nodes_with_times", [](TemporalWalk& tw, const std::string& walk_start_at_str, const std::vector<int>& end_nodes) {
            const WalkStartAt walk_start_at = walk_start_at_from_string(walk_start_at_str);
            auto walks_for_nodes_with_times = tw.get_random_walks_for_nodes_with_times(walk_start_at, end_nodes);

            // Convert the result to a Python dictionary of lists of lists of tuples
            py::dict py_walks_dict;
            for (const auto& [node_id, walks] : walks_for_nodes_with_times) {
                std::vector<std::vector<std::tuple<int, int64_t>>> py_walks;
                for (const auto& walk : walks) {
                    std::vector<std::tuple<int, int64_t>> py_walk;
                    for (const auto& [node, timestamp] : walk) {
                        py_walk.emplace_back(node, timestamp); // NOLINT(*-inefficient-vector-operation)
                    }
                    py_walks.emplace_back(std::move(py_walk));
                }
                py_walks_dict[py::cast(node_id)] = py_walks;
            }

            return py_walks_dict;
        },
        R"(
        Generates random walks with timestamps for multiple specified nodes in the temporal graph.

        Parameters:
        - walk_start_at (str): The starting point for the walks ("Begin", "End", "Random").
        - end_nodes (List[int]): A list of node IDs for which the walks will be generated.

        Returns:
        Dict[int, List[List[Tuple[int, int64_t]]]]: A dictionary mapping node IDs to their corresponding random walks, each walk is a list of tuples containing (node_id, timestamp).
        )", py::return_value_policy::move)
        .def("get_node_count", &TemporalWalk::get_node_count,
             R"(
             Returns the total number of nodes present in the temporal graph.

             Returns:
             int: The total number of nodes.
             )")
        .def("get_edge_count", &TemporalWalk::get_edge_count,
             R"(
             Returns the total number of directed edges in the temporal graph.

             Returns:
             int: The total number of directed edges.
             )")
        .def("get_node_ids", [](const TemporalWalk& tw) {
            const auto& node_ids = tw.get_node_ids();
            py::array_t<int> py_node_ids(static_cast<long>(node_ids.size()));

            auto py_node_ids_mutable = py_node_ids.mutable_unchecked<1>();
            for (size_t i = 0; i < node_ids.size(); ++i) {
                py_node_ids_mutable(i) = node_ids[i];
            }

            return py_node_ids;
        },
        R"(
        Returns a NumPy array containing the IDs of all nodes in the temporal graph.

        Returns:
        np.ndarray: A NumPy array with all node IDs.
        )");
}
