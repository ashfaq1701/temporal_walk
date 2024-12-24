#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <optional>
#include "../src/core/TemporalWalk.h"
#include "../src/random/LinearRandomPicker.h"
#include "../src/random/ExponentialRandomPicker.h"
#include "../src/random/UniformRandomPicker.h"
#include <stdexcept>


namespace py = pybind11;

constexpr int DEFAULT_WALK_FILL_VALUE = 0;

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
    else if (picker_type_str == "Exponential")
    {
        return RandomPickerType::Exponential;
    }
    else
    {
        throw std::invalid_argument("Invalid picker type: " + picker_type_str);
    }
}

WalkInitEdgeTimeBias walk_init_edge_time_bias_from_string(const std::string& walk_init_edge_time_bias_str)
{
    if (walk_init_edge_time_bias_str == "Bias_Earliest_Time")
    {
        return WalkInitEdgeTimeBias::Bias_Earliest_Time;
    }
    else if (walk_init_edge_time_bias_str == "Bias_Latest_Time")
    {
        return WalkInitEdgeTimeBias::Bias_Latest_Time;
    }
    else
    {
        throw std::invalid_argument("Invalid walk init edge time bias: " + walk_init_edge_time_bias_str);
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
        .def(py::init([](const std::optional<int64_t> max_time_capacity)
             {
                 return std::make_unique<TemporalWalk>(max_time_capacity.value_or(-1));
             }),
             py::arg("max_time_capacity") = py::none())
        .def("add_multiple_edges", [](TemporalWalk& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos)
             {
                 std::vector<EdgeInfo> edges;

                 for (const auto& edge_info : edge_infos)
                 {
                     int u = std::get<0>(edge_info);
                     int i = std::get<1>(edge_info);
                     int64_t t = std::get<2>(edge_info);
                     edges.emplace_back(EdgeInfo{u, i, t});
                 }

                 tw.add_multiple_edges(edges);
             },
             R"(
            Adds multiple directed edges to the temporal graph based on the provided vector of tuples.

            Parameters:
            - edge_infos (List[Tuple[int, int, int64_t]]): A list of tuples, each containing (source node, destination node, timestamp).
            )"
        )
        .def("get_random_walks", [](TemporalWalk& tw,
                                    const int max_walk_len,
                                    const std::string& walk_bias,
                                    const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                    const std::optional<long> num_cw = std::nullopt,
                                    const std::optional<int> num_walks_per_node = std::nullopt,
                                    const std::string& walk_direction = "Forward_In_Time",
                                    const std::string& walk_init_edge_time_bias = "Bias_Earliest_Time",
                                    const std::optional<int> context_window_len = std::nullopt,
                                    const float p_walk_success_threshold = 0.95)
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);
                 WalkInitEdgeTimeBias walk_init_edge_time_bias_enum = walk_init_edge_time_bias_from_string(
                     walk_init_edge_time_bias);

                 return tw.get_random_walks(
                     max_walk_len,
                     &walk_bias_enum,
                     initial_edge_bias_enum_ptr,
                     num_cw.value_or(-1),
                     num_walks_per_node.value_or(-1),
                     walk_direction_enum,
                     walk_init_edge_time_bias_enum,
                     context_window_len.value_or(-1),
                     p_walk_success_threshold);
             },
             R"(
             Generates temporal random walks.

             Parameters:
             max_walk_len (int): Maximum length of each random walk
             walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "Exponential")
             initial_edge_bias (str, optional): Type of bias for selecting initial edge
             num_cw (int, optional): Number of context windows to generate
             num_walks_per_node (int, optional): Number of walks per node (used if num_cw not specified)
             walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
             walk_init_edge_time_bias (str): Time bias for initial edge ("Bias_Earliest_Time" or "Bias_Latest_Time")
             context_window_len (int, optional): Size of context window
             p_walk_success_threshold (float): Minimum proportion of successful walks (default: 0.95)

             Returns:
             List[List[int]]: List of walks, each containing a sequence of node IDs
             )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("num_cw") = py::none(),
             py::arg("num_walks_per_node") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time",
             py::arg("walk_init_edge_time_bias") = "Bias_Earliest_Time",
             py::arg("context_window_len") = py::none(),
             py::arg("p_walk_success_threshold") = 0.95)

        .def("get_random_walks_with_times", [](TemporalWalk& tw,
                                               const int max_walk_len,
                                               const std::string& walk_bias,
                                               const std::optional<std::string>& initial_edge_bias = std::nullopt,
                                               const std::optional<long> num_cw = std::nullopt,
                                               const std::optional<int> num_walks_per_node = std::nullopt,
                                               const std::string& walk_direction = "Forward_In_Time",
                                               const std::string& walk_init_edge_time_bias = "Bias_Earliest_Time",
                                               const std::optional<int> context_window_len = std::nullopt,
                                               const float p_walk_success_threshold = 0.95)
             {
                 const RandomPickerType walk_bias_enum = picker_type_from_string(walk_bias);
                 const RandomPickerType* initial_edge_bias_enum_ptr = nullptr;
                 if (initial_edge_bias.has_value())
                 {
                     static const RandomPickerType edge_bias_enum = picker_type_from_string(*initial_edge_bias);
                     initial_edge_bias_enum_ptr = &edge_bias_enum;
                 }

                 WalkDirection walk_direction_enum = walk_direction_from_string(walk_direction);
                 WalkInitEdgeTimeBias walk_init_edge_time_bias_enum = walk_init_edge_time_bias_from_string(
                     walk_init_edge_time_bias);

                 return tw.get_random_walks_with_times(
                     max_walk_len,
                     &walk_bias_enum,
                     initial_edge_bias_enum_ptr,
                     num_cw.value_or(-1),
                     num_walks_per_node.value_or(-1),
                     walk_direction_enum,
                     walk_init_edge_time_bias_enum,
                     context_window_len.value_or(-1),
                     p_walk_success_threshold);
             },
             R"(
            Generates temporal random walks with timestamps.

            Parameters:
            max_walk_len (int): Maximum length of each random walk
            walk_bias (str): Type of bias for selecting next node ("Uniform", "Linear", "Exponential")
            initial_edge_bias (str, optional): Type of bias for selecting initial edge
            num_cw (int, optional): Number of context windows to generate
            num_walks_per_node (int, optional): Number of walks per node (used if num_cw not specified)
            walk_direction (str): Direction of walk ("Forward_In_Time" or "Backward_In_Time")
            walk_init_edge_time_bias (str): Time bias for initial edge ("Bias_Earliest_Time" or "Bias_Latest_Time")
            context_window_len (int, optional): Size of context window
            p_walk_success_threshold (float): Minimum proportion of successful walks (default: 0.95)

            Returns:
            List[List[Tuple[int, int64_t]]]: List of walks, each containing (node_id, timestamp) pairs
            )",
             py::arg("max_walk_len"),
             py::arg("walk_bias"),
             py::arg("initial_edge_bias") = py::none(),
             py::arg("num_cw") = py::none(),
             py::arg("num_walks_per_node") = py::none(),
             py::arg("walk_direction") = "Forward_In_Time",
             py::arg("walk_init_edge_time_bias") = "Bias_Earliest_Time",
             py::arg("context_window_len") = py::none(),
             py::arg("p_walk_success_threshold") = 0.95)

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

                 std::vector<EdgeInfo> edge_infos;
                 for (const auto& edge : edges)
                 {
                     auto edge_tuple = edge.cast<py::tuple>();
                     const int source = py::cast<int>(edge_tuple[0]);
                     const int target = py::cast<int>(edge_tuple[1]);
                     const auto attrs = edge_tuple[2].cast<py::dict>();
                     const int64_t timestamp = py::cast<int64_t>(attrs["timestamp"]);

                     edge_infos.emplace_back(EdgeInfo{source, target, timestamp});
                 }

                 tw.add_multiple_edges(edge_infos);
             },
             R"(
            Adds edges from a networkx graph to the current TemporalWalk object.

            Parameters:
            - nx_graph (networkx.DiGraph): The networkx graph to load edges from.
            )"
        )
        .def("to_networkx", [](const TemporalWalk& tw)
             {
                 const auto edges = tw.get_edges();

                 const py::module nx = py::module::import("networkx");
                 const py::object DiGraph = nx.attr("DiGraph");
                 py::object nx_graph = DiGraph();

                 for (const auto& edge : edges)
                 {
                     py::dict kwargs;
                     kwargs["timestamp"] = edge.t;

                     nx_graph.attr("add_edge")(edge.u, edge.i, **kwargs);
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
        .def(py::init<>(), "Initialize a LinearRandomPicker instance.")
        .def("pick_random", &LinearRandomPicker::pick_random,
             "Pick a random index with linear probabilities.",
             py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<ExponentialRandomPicker>(m, "ExponentialRandomPicker")
        .def(py::init<>(), "Initialize a ExponentialRandomPicker instance.")
        .def("pick_random", &ExponentialRandomPicker::pick_random,
             "Pick a random index with exponential probabilities.",
             py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);

    py::class_<UniformRandomPicker>(m, "UniformRandomPicker")
        .def(py::init<>(), "Initialize a UniformRandomPicker instance.")
        .def("pick_random", &UniformRandomPicker::pick_random,
             "Pick a random index with uniform probabilities.",
             py::arg("start"), py::arg("end"), py::arg("prioritize_end") = true);
}
