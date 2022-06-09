cmake_minimum_required(VERSION 3.5)

include(FetchContent)
FetchContent_Declare(OpenCV
    URL  http://artifactory.smoa.cloud/artifactory/opencv/opencv-4.0.1-linux.tar.gz
    #URL_HASH  MD5=aaaa841f8d28af8a587c73d27a668298  
    #SOURCE_DIR  
    )
FetchContent_MakeAvailable(OpenCV)
set(OpenCV_PATH ${opencv_SOURCE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

project(franctionalRetinex LANGUAGES CXX CUDA)

add_library(retinexCuda STATIC
    ./kernels/franctionalRetinex.cu)
target_include_directories(retinexCuda PUBLIC 
    ./)

add_executable(franctionalRetinex ./franctionalRetinexCuda.cpp)
target_include_directories(franctionalRetinex PUBLIC 
    ./kernels/
    )

find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)
set(OpenCV_DIR ${OpenCV_PATH}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

target_link_libraries(franctionalRetinex PRIVATE
    retinexCuda
    ${OpenCV_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(franctionalRetinex PUBLIC OpenMP::OpenMP_CXX)
endif()

add_custom_command(TARGET franctionalRetinex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${opencv_SOURCE_DIR}/lib
        $<TARGET_FILE_DIR:franctionalRetinex>)