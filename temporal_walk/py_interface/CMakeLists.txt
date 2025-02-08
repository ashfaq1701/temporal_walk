if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
    )

    set_source_files_properties(
            ${SRC_FILES}
            PROPERTIES
            LANGUAGE CUDA
    )
endif()

pybind11_add_module(_temporal_walk
        _temporal_walk.cpp
        ${SRC_FILES}
)

target_compile_features(_temporal_walk PRIVATE cxx_std_17)
set_target_properties(_temporal_walk PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_compile_options(_temporal_walk PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
            -Xcompiler=-fPIC
            >
    )

    target_link_libraries(_temporal_walk PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
    )

    set_target_properties(_temporal_walk PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 89
    )

    target_include_directories(_temporal_walk PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_walk/src
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
else ()
    target_include_directories(_temporal_walk PRIVATE
            ${CMAKE_SOURCE_DIR}/temporal_walk/src
    )
endif()

target_link_libraries(_temporal_walk PRIVATE
        pybind11::module
        ${Python3_LIBRARIES}
        Boost::math
)
