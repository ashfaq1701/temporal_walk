# Set required CUDA flags for the module
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
    )
endif()

# Create the Python module
pybind11_add_module(
        _temporal_walk
        _temporal_walk.cpp
        ${SRC_FILES}
)

# Set common compile properties
target_compile_features(_temporal_walk PRIVATE cxx_std_17)
set_target_properties(_temporal_walk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

# Set CUDA-specific properties
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    # Set CUDA source properties
    set_source_files_properties(
            ${CUDA_SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )

    # Add CUDA compile options including extended lambda
    target_compile_options(_temporal_walk PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
            >
    )

    # Link CUDA libraries
    target_link_libraries(_temporal_walk PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
    )

    # CUDA compilation properties
    set_target_properties(_temporal_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 89
    )
endif()

# Common libraries and includes
target_link_libraries(_temporal_walk PRIVATE
        pybind11::module
        ${Python3_LIBRARIES}
        Boost::math
)

# Include directories
target_include_directories(_temporal_walk PRIVATE
        ${CMAKE_SOURCE_DIR}/temporal_walk/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
