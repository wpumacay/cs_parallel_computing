cmake_minimum_required(VERSION 3.0)

enable_language(CXX)
set (CMAKE_CXX_STANDARD 11)

find_package(OpenMP)

if ( OPENMP_FOUND )
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

project( quicksort )

add_executable( quicksort quicksort.cpp )

target_link_libraries( quicksort pthread )