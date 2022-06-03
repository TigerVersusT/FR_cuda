cmake_minimum_required(VERSION 3.5)

project(franctionalRetinex LANGUAGES CXX CUDA)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_library(retinexCuda STATIC
    ./kernels/franctionalRetinex.cu)
target_include_directories(retinexCuda PUBLIC 
    ./)

add_executable(franctionalRetinex ./franctionalRetinexCuda.cpp)
target_include_directories(franctionalRetinex PUBLIC 
    ./kernels/)

target_link_libraries(franctionalRetinex PRIVATE
    ${OpenCV_LIBS} 
    retinexCuda
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(franctionalRetinex PUBLIC OpenMP::OpenMP_CXX)
endif()