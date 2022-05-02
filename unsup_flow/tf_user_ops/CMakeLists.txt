## CMake build file for the tf-user-ops repository.
##

# allow for project VERSION option
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

project(tfops VERSION 0.2 LANGUAGES CXX)
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run "
  "CMake from there. You may need to remove CMakeCache.txt. ")
endif()


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CXX_CLANG_COMPILE_OPTIONS
  "-march=native"
  "-Weverything"
  "-Wno-c++98-compat"
  "-Wno-c++98-c++11-c++14-compat"
  )
set(CXX_GCC_COMPILE_OPTIONS
  "-march=native"
  "-Wall"
  "-Wno-unknown-pragmas"
  "-D_GLIBCXX_USE_CXX11_ABI=0"
  )

# -------- FIND DEPENDENCIES WITH CMAKE CONVENTIONS --------
# -------- --------- REQUIRED DEPENDENCIES -------- --------

# TENSORFLOW
set(TENSORFLOW_LOCATION "" CACHE FILEPATH "User-specified install location of tensorflow (python site-packages)")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
find_package(Tensorflow REQUIRED)

# 3rd party modules like nanoflann
add_subdirectory(3rd_party)

# ---------------- tfops LIBRARY TARGETS -------------------
add_subdirectory(src/points_to_voxel)
add_subdirectory(src/tf_knn_op)
