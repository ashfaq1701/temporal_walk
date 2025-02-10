#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include <stdexcept>
#include "../src/core/structs.h"
#include "../src/random/UniformRandomPicker.cuh"
#include "../src/random/LinearRandomPicker.cuh"
#include "../src/random/ExponentialIndexRandomPicker.cuh"
#include "../src/random/WeightBasedRandomPicker.cuh"
#include "../src/core/TemporalWalk.cuh"


namespace py = pybind11;

RandomPickerType picker_type_from_string(const std::string& picker_type_str)
{
    if (picker_type_str == "Uniform")
    {
        return RandomPickerType::Uniform;
    }
    else if (picker_type_str == "Linear")
    {
        return RandomPickerType::Linear;
    }
    else if (picker_type_str == "ExponentialIndex")
    {
        return RandomPickerType::ExponentialIndex;
    }
    else if (picker_type_str == "ExponentialWeight")
    {
        return RandomPickerType::ExponentialWeight;
    }
    else
    {
        throw std::invalid_argument("Invalid picker type: " + picker_type_str);
    }
}

WalkDirection walk_direction_from_string(const std::string& walk_direction_str)
{
    if (walk_direction_str == "Forward_In_Time")
    {
        return WalkDirection::Forward_In_Time;
    }
    else if (walk_direction_str == "Backward_In_Time")
    {
        return WalkDirection::Backward_In_Time;
    }
    else
    {
        throw std::invalid_argument("Invalid walk direction: " + walk_direction_str);
    }
}

PYBIND11_MODULE(_temporal_walk, m)
{
    py::class_<TemporalWalk>(m, "TemporalWalk")
        .def(py::init([](const bool is_directed, const std::optional<bool> use_gpu, const std::optional<int64_t> max_time_capacity, std::optional<bool> enable_weight_computation, std::optional<double> timescale_bound)
             {
                 return std::make_unique<TemporalWalk>(
                     is_directed,
                     use_gpu.value_or(false),
                     max_time_capacity.value_or(-1),
                     enable_weight_computation.value_or(false),
                     timescale_bound.value_or(DEFAULT_TIMESCALE_BOUND));
             }),
             py::arg("is_directed"),
             py::arg("use_gpu") = false,
             py::arg("max_time_capacity") = py::none(),
             py::arg("enable_weight_computation") = py::none(),
             py::arg("timescale_bound") = py::none())
        .def("add_multiple_edges", [](TemporalWalk& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos)
             {
                 tw.add_multiple_edges(edge_infos);
             },
             R"(
            Adds multiple directed edges to the temporal graph based on the provided vector of tuples.

            Parameters:
            - edge_infos (List[Tuple[int, int, int64_t]]): A list of tuples, each containing (source node, destination node, timestamp).
            )"
        )
        .def("get_random_walks_for_all_nodes", [](TemporalWalk& tw,
                                    const int max_walk_len,
                                    const std::string& walk_bias,
                                    const int num_walks_per_node,
                                    const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                    const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 const WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 return tw.get_random_walks_for_all_nodes(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);
             },
             R"(
             Generates temporal random walks for all the nodes in the graph.

             Parameters:
             max_walk_len (int): Maximum length of each random walk
             walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex" or "ExponentialWeight")
             num_walks_per_node (int): Number of walks per node
             initial_edge_bias (str, optional): Type of bias for selecting initial edge
             walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")

             Returns:
             List[List[int]]: List of walks, each containing a sequence of node IDs
             )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks_and_times_for_all_nodes", [](TemporalWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const int num_walks_per_node,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 const WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 const auto walks_with_times = tw.get_random_walks_and_times_for_all_nodes(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);

                std::vector<std::vector<std::tuple<int, int64_t>>> result;
                result.reserve(walks_with_times.size());

                for (const auto& walk : walks_with_times) {
                    std::vector<std::tuple<int, int64_t>> converted_walk;
                    converted_walk.reserve(walk.size());

                    for (const auto& node_time : walk) {
                        converted_walk.emplace_back(node_time.node, node_time.timestamp);
                    }

                    result.push_back(std::move(converted_walk));
                }

                return result;
             },
             R"(
            Generates temporal random walks with timestamps for all the nodes in the graph.

            Parameters:
            max_walk_len (int): Maximum length of each random walk
            walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
            num_walks_per_node (int): Number of walks per node
            initial_edge_bias (str, optional): Type of bias for selecting initial edge
            walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")

            Returns:
            List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks", [](TemporalWalk& tw,
                                    const int max_walk_len,
                                    const std::string& walk_bias,
                                    const int num_walks_per_node,
                                    const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                    const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 return tw.get_random_walks(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);
             },
             R"(
             Generates temporal random walks.

             Parameters:
             max_walk_len (int): Maximum length of each random walk
             walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
             num_walks_per_node (int): Number of walks per node
             initial_edge_bias (str, optional): Type of bias for selecting initial edge
             walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")

             Returns:
             List[List[int]]: List of walks, each containing a sequence of node IDs
             )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks_and_times", [](TemporalWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const int num_walks_per_node,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time")
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 auto walks_with_times = tw.get_random_walks_and_times(
                     max_walk_len,
                     &walk_bias_enum,
                     num_walks_per_node,
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum);

                std::vector<std::vector<std::tuple<int, int64_t>>> result;
                result.reserve(walks_with_times.size());

                for (const auto& walk : walks_with_times) {
                    std::vector<std::tuple<int, int64_t>> converted_walk;
                    converted_walk.reserve(walk.size());

                    for (const auto& node_time : walk) {
                        converted_walk.emplace_back(node_time.node, node_time.timestamp);
                    }

                    result.push_back(std::move(converted_walk));
                }

                return result;
             },
             R"(
            Generates temporal random walks with timestamps.

            Parameters:
            max_walk_len (int): Maximum length of each random walk
            walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
            num_walks_per_node (int): Number of walks per node
            initial_edge_bias (str, optional): Type of bias for selecting initial edge
            walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")

            Returns:
            List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_walks_per_node"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time")

        .def("get_random_walks_with_specific_number_of_contexts", [](TemporalWalk& tw,
                                    const int max_walk_len,
                                    const std::string& walk_bias,
                                    const std::optional<long> num_cw = std::nullopt,
                                    const std::optional<int> num_walks_per_node = std::nullopt,
                                    const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                    const std::string& walk_direction = "Forward_In_Time",
                                    const std::optional<int> context_window_len = std::nullopt,
                                    const float p_walk_success_threshold = DEFAULT_SUCCESS_THRESHOLD)
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 return tw.get_random_walks_with_specific_number_of_contexts(
                     max_walk_len,
                     &walk_bias_enum,
                     num_cw.value_or(-1),
                     num_walks_per_node.value_or(-1),
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum,
                     context_window_len.value_or(-1),
                     p_walk_success_threshold);
             },
             R"(
             Generates temporal random walks with specified number of contexts. Here number of walks can vary based on their actual lengths.

             Parameters:
             max_walk_len (int): Maximum length of each random walk
             walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
             num_cw (int, optional): Number of context windows to generate
             num_walks_per_node (int, optional): Number of walks per node (used if num_cw not specified)
             initial_edge_bias (str, optional): Type of bias for selecting initial edge
             walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
             context_window_len (int, optional): Size of context window
             p_walk_success_threshold (float): Minimum proportion of successful walks (default: 0.01)

             Returns:
             List[List[int]]: List of walks, each containing a sequence of node IDs
             )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_cw") = py::none(),
             py::arg("num_walks_per_node") = py::none(),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time",
             py::arg("context_window_len") = py::none(),
             py::arg("p_walk_success_threshold") = DEFAULT_SUCCESS_THRESHOLD)

        .def("get_random_walks_and_times_with_specific_number_of_contexts", [](TemporalWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const std::optional<long> num_cw = std::nullopt,
                                               const std::optional<int> num_walks_per_node = std::nullopt,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time",
                                               const std::optional<int> context_window_len = std::nullopt,
                                               const float p_walk_success_threshold = DEFAULT_SUCCESS_THRESHOLD)
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);

                 auto walks_with_times = tw.get_random_walks_and_times_with_specific_number_of_contexts(
                     max_walk_len,
                     &walk_bias_enum,
                     num_cw.value_or(-1),
                     num_walks_per_node.value_or(-1),
                     initial_edge_bias_enum_ptr,
                     walk_direction_enum,
                     context_window_len.value_or(-1),
                     p_walk_success_threshold);

                std::vector<std::vector<std::tuple<int, int64_t>>> result;
                result.reserve(walks_with_times.size());

                for (const auto& walk : walks_with_times) {
                    std::vector<std::tuple<int, int64_t>> converted_walk;
                    converted_walk.reserve(walk.size());

                    for (const auto& node_time : walk) {
                        converted_walk.emplace_back(node_time.node, node_time.timestamp);
                    }

                    result.push_back(std::move(converted_walk));
                }

                return result;
             },
             R"(
            Generates temporal random walks with timestamps with specified number of contexts. Here number of walks can vary based on their actual lengths.

            Parameters:
            max_walk_len (int): Maximum length of each random walk
            walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "ExponentialIndex", "ExponentialWeight")
            num_cw (int, optional): Number of context windows to generate
            num_walks_per_node (int, optional): Number of walks per node (used if num_cw not specified)
            initial_edge_bias (str, optional): Type of bias for selecting initial edge
            walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
            context_window_len (int, optional): Size of context window
            p_walk_success_threshold (float): Minimum proportion of successful walks (default: 0.01)

            Returns:
            List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("num_cw") = py::none(),
             py::arg("num_walks_per_node") = py::none(),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time",
             py::arg("context_window_len") = py::none(),
             py::arg("p_walk_success_threshold") = DEFAULT_SUCCESS_THRESHOLD)

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
        .def("get_node_ids", [](const TemporalWalk& tw)
             {
                 const auto& node_ids = tw.get_node_ids();
                 py::array_t<int> py_node_ids(static_cast<long>(node_ids.size()));

                 auto py_node_ids_mutable = py_node_ids.mutable_unchecked<1>();
                 for (size_t i = 0; i < node_ids.size(); ++i)
                 {
                     py_node_ids_mutable(i) = node_ids[i];
                 }

                 return py_node_ids;
             },
             R"(
            Returns a NumPy array containing the IDs of all nodes in the temporal graph.

            Returns:
            np.ndarray: A NumPy array with all node IDs.
            )"
        )
        .def("clear", &TemporalWalk::clear,
             R"(
            Clears and reinitiates the underlying graph.
            )"
        )
        .def("add_edges_from_networkx", [](TemporalWalk& tw, const py::object& nx_graph)
             {
                 const py::object edges = nx_graph.attr("edges")(py::arg("data") = true);

                 std::vector<std::tuple<int, int, int64_t>> edge_infos;
                 for (const auto& edge : edges)
                 {
                     auto edge_tuple = edge.cast<py::tuple>();
                     const int source = py::cast<int>(edge_tuple[0]);
                     const int target = py::cast<int>(edge_tuple[1]);
                     const auto attrs = edge_tuple[2].cast<py::dict>();
                     const int64_t timestamp = py::cast<int64_t>(attrs["timestamp"]);

                     edge_infos.emplace_back(source, target, timestamp);
                 }

                 tw.add_multiple_edges(edge_infos);
             },
             R"(
            Adds edges from a networkx graph to the current TemporalWalk object.

            Parameters:
            - nx_graph (networkx.Graph): The networkx graph to load edges from.
            )"
        )
        .def("to_networkx", [](TemporalWalk& tw)
             {
                 const auto edges = tw.get_edges();

                 const py::module nx = py::module::import("networkx");
                 const py::object GraphClass = tw.get_is_directed() ? nx.attr("DiGraph") : nx.attr("Graph");
                 py::object nx_graph = GraphClass();

                 for (const auto& [src, dest, ts] : edges)
                 {
                     py::dict kwargs;
                     kwargs["timestamp"] = ts;

                     nx_graph.attr("add_edge")(src, dest, **kwargs);
                 }

                 return nx_graph;
             },
             R"(
            Exports the TemporalWalk object to a networkX graph.

            Returns:
            networkx.Graph: The exported networkx graph.
            )"
        );

    py::class_<LinearRandomPicker>(m, "LinearRandomPicker")
        .def(py::init<bool>(), "Initialize a LinearRandomPicker instance.", py::arg("use_gpu") = false)
        .def("pick_random", &LinearRandomPicker::pick_random,
            "Pick a random index with linear probabilities.",
            py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<ExponentialIndexRandomPicker>(m, "ExponentialIndexRandomPicker")
        .def(py::init<bool>(), "Initialize a ExponentialIndexRandomPicker instance.", py::arg("use_gpu") = false)
        .def("pick_random", &ExponentialIndexRandomPicker::pick_random,
            "Pick a random index with exponential probabilities with index sampling.",
            py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<UniformRandomPicker>(m, "UniformRandomPicker")
        .def(py::init<bool>(), "Initialize a UniformRandomPicker instance.", py::arg("use_gpu") = false)
        .def("pick_random", &UniformRandomPicker::pick_random,
            "Pick a random index with uniform probabilities.",
            py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<WeightBasedRandomPicker>(m, "WeightBasedRandomPicker")
        .def(py::init([]() {
            return std::make_unique<WeightBasedRandomPicker>(false);
        }), "Initialize a WeightBasedRandomPicker instance.")
        .def("pick_random", &WeightBasedRandomPicker::pick_random,
            "Pick a random index based on cumulative weights",
            py::arg("cumulative_weights"), py::arg("group_start"), py::arg("group_end"));
}
