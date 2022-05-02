cmake_minimum_required(VERSION 3.2)
project(tf_knn_op)
set(CMAKE_CXX_STANDARD 11)

find_package(Tensorflow REQUIRED)
find_package(PkgConfig REQUIRED)

add_library(k_nearest_neighbor_op
    SHARED
    k_nearest_neighbor_op.cc
    k_nearest_neighbor_kernel.cc
    k_nearest_neighbor.cc)

target_include_directories(k_nearest_neighbor_op
    PRIVATE
    ${Tensorflow_INCLUDE_DIR}
    )

target_link_libraries(k_nearest_neighbor_op
    ${Tensorflow_LIB}
    nanoflann::nanoflann
    )
target_compile_options(k_nearest_neighbor_op
    PRIVATE
    ${Tensorflow_ABI_FLAG}
    PRIVATE
    -ffast-math
    )