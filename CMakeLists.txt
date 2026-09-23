cmake_minimum_required(VERSION 3.18)
project(jetson_yolo_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(yolo_postprocess STATIC
    src/yolo_cuda/postprocess.cu
)
target_include_directories(yolo_postprocess PUBLIC include)
set_target_properties(yolo_postprocess PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(yolo_postprocess_demo
    src/yolo_cuda/postprocess_demo.cu
)
target_link_libraries(yolo_postprocess_demo PRIVATE yolo_postprocess)
target_include_directories(yolo_postprocess_demo PRIVATE include)
