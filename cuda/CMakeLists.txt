cmake_minimum_required(VERSION 3.5)

project(franctionalRetinexCuda LANGUAGES CXX CUDA)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_library(utills STATIC
    ./kernels/utills.cu)

add_library(fstDiffCuda STATIC
    ./kernels/fstDiffCuda.cu)
target_include_directories(fstDiffCuda 
    PUBLIC
        ./
)
target_link_libraries(fstDiffCuda
    PRIVATE
        utills
)

add_library(cal8DerivCuda STATIC
    ./kernels/cal8DerivCuda.cu)
target_include_directories(cal8DerivCuda
    PUBLIC
        ./
)
target_link_libraries(cal8DerivCuda
    PRIVATE
        utills    
)

add_library(arrayPowerCuda STATIC
    ./kernels/arrayPower.cu
)
target_include_directories(arrayPowerCuda
    PUBLIC
        ./
)
target_link_libraries(arrayPowerCuda
    PRIVATE
        utills
)

add_library(secondTermCuda STATIC
    ./kernels/secondTerm.cu
)
target_include_directories(secondTermCuda
    PUBLIC
        ./
)
target_link_libraries(secondTermCuda
    PRIVATE
        utills
)

add_library(tempCuda STATIC
    ./kernels/tempCuda.cu
)
target_include_directories(tempCuda
    PUBLIC
        ./
)
target_link_libraries(tempCuda
    PRIVATE
        utills
)

add_executable(franctionalRetinexCuda ./franctionalRetinexCuda.cpp)
target_link_libraries(franctionalRetinexCuda PRIVATE
    ${OpenCV_LIBS}  
    fstDiffCuda
    cal8DerivCuda
    arrayPowerCuda
    secondTermCuda
    tempCuda
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(franctionalRetinexCuda PUBLIC OpenMP::OpenMP_CXX)
endif()