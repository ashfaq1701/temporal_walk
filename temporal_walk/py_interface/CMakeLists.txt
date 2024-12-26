pybind11_add_module(
        _temporal_walk
        _temporal_walk.cpp
        ${SRC_FILES}
)

target_link_libraries(_temporal_walk PRIVATE pybind11::module ${Python3_LIBRARIES} Boost::math)
