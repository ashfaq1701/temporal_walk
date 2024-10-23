#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/TemporalWalk.h"
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
        .def(py::init([](int num_walks, int len_walk, const std::string& picker_type_str) {
            RandomPickerType picker_type = picker_type_from_string(picker_type_str);
            return std::make_unique<TemporalWalk>(num_walks, len_walk, picker_type);
        }), py::arg("num_walks"), py::arg("len_walk"), py::arg("picker_type"))

        .def("add_edge", &TemporalWalk::add_edge)
        .def("add_multiple_edges", [](TemporalWalk& tw, const std::vector<std::tuple<int, int, int64_t>>& edge_infos) {
            std::vector<EdgeInfo> edges;

            for (const auto& edge_info : edge_infos) {
                int u = std::get<0>(edge_info);
                int i = std::get<1>(edge_info);
                int64_t t = std::get<2>(edge_info);
                edges.emplace_back(EdgeInfo {u, i, t});
            }

            tw.add_multiple_edges(edges);
        })

        .def("get_random_walks", [](TemporalWalk& tw, const std::string& walk_start_at_str, const int end_node=-1, const int fill_value=DEFAULT_WALK_FILL_VALUE) {
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
        })

        .def("get_random_walks_for_nodes", [](TemporalWalk& tw, const std::string& walk_start_at_str, const std::vector<int>& end_nodes, const int fill_value=DEFAULT_WALK_FILL_VALUE) {
            const WalkStartAt walk_start_at = walk_start_at_from_string(walk_start_at_str);

            auto walks_for_nodes = tw.get_random_walks_for_nodes(walk_start_at, end_nodes);
            const int len_walk = tw.get_len_walk();  // Assuming len_walk is retrievable

            py::list py_walks_list;

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

                py_walks_list.append(py_walks);
            }

            return py_walks_list;
        })
    .def("get_node_count", &TemporalWalk::get_node_count)
    .def("get_edge_count", &TemporalWalk::get_edge_count)
    .def("get_node_ids", [](const TemporalWalk& tw) {
        const auto& node_ids = tw.get_node_ids();
        py::array_t<int> py_node_ids(static_cast<long>(node_ids.size()));

        auto py_node_ids_mutable = py_node_ids.mutable_unchecked<1>();
        for (size_t i = 0; i < node_ids.size(); ++i) {
            py_node_ids_mutable(i) = node_ids[i];
        }

        return py_node_ids;
    });
}
