###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
#
# Kripke CMake Build for Ascent (Standalone)
#
###############################################################################

cmake_minimum_required(VERSION 3.14)

project(Kripke LANGUAGES CXX)

# Ensure we have CMake's MPI targets.
find_package(MPI REQUIRED COMPONENTS CXX)

# Find Ascent, provide a hint if needed
find_package(Ascent REQUIRED
             NO_DEFAULT_PATH
             PATHS ${CMAKE_SOURCE_DIR}/../../../)

# Collect source files
set(KRIPKE_SOURCES
    Kripke/Directions.cpp
    Kripke/Grid.cpp
    Kripke/Kernel.cpp
    Kripke/Layout.cpp
    Kripke/Subdomain.cpp
    Kripke/Sweep_Solver.cpp
    Kripke/ParallelComm.cpp
    Kripke/Timing.cpp

    Kripke/Kernel/Kernel_3d_GDZ.cpp
    Kripke/Kernel/Kernel_3d_DGZ.cpp
    Kripke/Kernel/Kernel_3d_ZDG.cpp
    Kripke/Kernel/Kernel_3d_DZG.cpp
    Kripke/Kernel/Kernel_3d_ZGD.cpp
    Kripke/Kernel/Kernel_3d_GZD.cpp

    Kripke/ParallelComm/BlockJacobiComm.cpp
    Kripke/ParallelComm/SweepComm.cpp
    tools/kripke.cpp
    tools/testKernels.cpp
)

# Add include directories
include_directories(.)
include_directories("tools")

# Ensure MPI include directories are added
include_directories(${MPI_CXX_INCLUDE_DIRS})

# Add the executable
add_executable(kripke_par_exec ${KRIPKE_SOURCES})

# Link to required libraries (explicitly link MPI and Ascent)
target_link_libraries(kripke_par_exec ascent::ascent_mpi ${MPI_CXX_LIBRARIES})

# Set compile flags (no need for OpenMP)
target_compile_definitions(kripke_par_exec PRIVATE "-D PARALLEL")

# Copy example scripts and yaml files from the source directory to the compiled executable location
configure_file(${CMAKE_SOURCE_DIR}/ascent_actions.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ascent_actions.yaml
               COPYONLY)

# Ensure the YAML files are in the same directory as the executable
add_custom_command(TARGET kripke_par_exec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_actions.yaml
    $<TARGET_FILE_DIR:kripke_par_exec>
)
