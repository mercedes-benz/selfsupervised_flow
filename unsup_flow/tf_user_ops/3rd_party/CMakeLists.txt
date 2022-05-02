# Define nanoflann lib (header-only)
add_library(nanoflann INTERFACE)

target_include_directories(nanoflann
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nanoflann/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)

add_library(nanoflann::nanoflann ALIAS nanoflann)
