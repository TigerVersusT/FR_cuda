cmake_minimum_required(VERSION 3.5)

project (franctionalRetinex)

find_package(OpenCV REQUIRED)

add_executable(franctionalRetinex franctionalRetinex.cpp)

target_include_directories(franctionalRetinex
    PRIVATE 
        /home/ubuntu/ncnn/build/install/include/ncnn/
)

target_link_libraries(franctionalRetinex
    ${OpenCV_LIBS}
    /home/ubuntu/ncnn/build/install/lib/libncnn.a
    gomp
    pthread
    )