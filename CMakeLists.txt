cmake_minimum_required(VERSION 3.20)
project(eigen-nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)


find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(eigen-nn-test eigen-nn-test.cpp)
target_include_directories(eigen-nn-test PRIVATE
    ./
)
target_link_libraries(eigen-nn-test PRIVATE
    Eigen3::Eigen
)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(ext/nanobind)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

nanobind_add_module(eignn binding.cpp)
target_include_directories(eignn PRIVATE
    ./
)
target_link_libraries(eignn PRIVATE
    Eigen3::Eigen
)


add_dependencies(eigen-nn-test eignn)
