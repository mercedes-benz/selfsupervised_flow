# -------------- OBJECTS3D_COMMON LIBRARY TARGET ---------------
# library target name
set(LIB_SUFFIX points-to-voxel)
set(TFOPS_LIB ${PROJECT_NAME}_${LIB_SUFFIX})

set(${TFOPS_LIB}_SOURCES
  points_to_voxel_op.cc
  )
# to be installed
set(${TFOPS_LIB}_PUBLIC_HEADERS
  )

add_library(${TFOPS_LIB} SHARED ${${TFOPS_LIB}_SOURCES})
# Export as title case common
set_target_properties(${TFOPS_LIB} PROPERTIES EXPORT_NAME ${TFOPS_LIB})
set_target_properties(${TFOPS_LIB} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${TFOPS_LIB} PROPERTIES SOVERSION ${PROJECT_VERSION})
set_target_properties(${TFOPS_LIB} PROPERTIES PUBLIC_HEADER "${${TFOPS_LIB}_PUBLIC_HEADERS}")

target_include_directories(${TFOPS_LIB} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_include_directories(${TFOPS_LIB} SYSTEM PUBLIC
  ${Tensorflow_INCLUDE_DIR}
  )

target_link_libraries(${TFOPS_LIB}
  ${Tensorflow_LIB}
  )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(${TFOPS_LIB} PRIVATE ${CXX_CLANG_COMPILE_OPTIONS})
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(${TFOPS_LIB} PRIVATE ${CXX_GCC_COMPILE_OPTIONS})
endif()
