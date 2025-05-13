cmake_minimum_required(VERSION 3.18)
project(ParallelJacobian LANGUAGES CXX CUDA)

include_directories(
    "${CMAKE_SOURCE_DIR}/Include"
    "C:/Program Files/NVIDIA cuDSS/v0.4/include"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
)

link_directories(
    "C:/Program Files/NVIDIA cuDSS/v0.4/lib/12"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"
)

file(GLOB SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/Source/*.cpp"
    "${CMAKE_SOURCE_DIR}/Source/gpufunc.cu"
)

add_executable(${CMAKE_PROJECT_NAME} "${CMAKE_SOURCE_DIR}/Source/kernel.cu")

add_library(my_lib ${SOURCE_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE my_lib cudss cublas)

set(CMAKE_BUILD_TYPE Release)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "native"
)

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "C:/Program Files/NVIDIA cuDSS/v0.4/bin/12/cudss64_0.dll"
    $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>
)