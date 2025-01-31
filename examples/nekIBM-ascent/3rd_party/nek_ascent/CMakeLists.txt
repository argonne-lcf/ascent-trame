cmake_minimum_required(VERSION 3.9)
project(NekAscent)

find_package(MPI REQUIRED)
find_package(Ascent REQUIRED)

if($ENV{UNDERSCORE})
    add_definitions(-DUNDERSCORE)
endif()
if($ENV{UPCASE})
    add_definitions(-DUPCASE)
endif()

add_definitions()
add_library(nekascent STATIC Bridge.cxx)
target_link_libraries(nekascent ascent::ascent_mpi)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_BUILD_TYPE Release)

install(TARGETS nekascent DESTINATION lib)
install(FILES Bridge.h DESTINATION include)
