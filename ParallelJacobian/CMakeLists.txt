cmake_minimum_required(VERSION 3.18)
project(ParallelJacobian LANGUAGES CXX CUDA)

include_directories("C:/Program Files/NVIDIA cuDSS/v0.4/include")

include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")

link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64")
link_directories("C:/Program Files/NVIDIA cuDSS/v0.4/lib/12")

set(SOURCE_FILES
    CuDssSolver.cpp
    DataInitializer.cpp
    EditionalTools.cpp
    FileOperations.cpp
    NewtonSolver.cpp
)

add_library(my_lib ${SOURCE_FILES} gpufunc.cu)

add_executable(ParallelJacobian kernel.cu)

target_link_libraries(ParallelJacobian PRIVATE my_lib cudss cublas)

set(CMAKE_BUILD_TYPE Debug)

set_target_properties(ParallelJacobian PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

add_custom_command(TARGET ParallelJacobian POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/cudss64_0.dll"  # Шлях до cudss64_0.dll
    "${CMAKE_SOURCE_DIR}/run.ps1"        # Шлях до run.ps1
    "${CMAKE_SOURCE_DIR}/draw.py"        # Шлях до draw.py
    $<TARGET_FILE_DIR:ParallelJacobian>  # Директорія виконуваного файлу
)
