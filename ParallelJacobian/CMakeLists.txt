cmake_minimum_required(VERSION 3.18)
project(ParallelJacobian LANGUAGES CXX CUDA)

if(WIN32)
    file(GLOB CUDA_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
    list(SORT CUDA_DIRS)
    list(GET CUDA_DIRS 0 CUDA_PATH)

    string(REGEX MATCH "v([0-9]+\\.[0-9]+)" _cuda_ver_match "${CUDA_PATH}")
    string(REPLACE "v" "" CUDA_VERSION "${_cuda_ver_match}")
    message(STATUS "Detected CUDA path: ${CUDA_PATH}")
    message(STATUS "Detected CUDA version: ${CUDA_VERSION}")

    file(GLOB CUDSS_DIRS "C:/Program Files/NVIDIA cuDSS/v*")
    list(SORT CUDSS_DIRS)
    list(GET CUDSS_DIRS 0 CUDSS_PATH)

    string(REGEX MATCH "v([0-9]+\\.[0-9]+)" _cudss_ver_match "${CUDSS_PATH}")
    string(REPLACE "v" "" CUDSS_VERSION "${_cudss_ver_match}")
    message(STATUS "Detected cuDSS path: ${CUDSS_PATH}")
    message(STATUS "Detected cuDSS version: ${CUDSS_VERSION}")

endif()

if(UNIX)
    set(CUDA_PATH "/usr/local/cuda")
    set(CUDSS_PATH "/usr/local/cudss")
endif()

include_directories(
    "${CMAKE_SOURCE_DIR}/Include"
    "${CUDSS_PATH}/include"
    "${CUDA_PATH}/include"
)

if(WIN32)
    link_directories(
        "${CUDSS_PATH}/lib/12"
        "${CUDA_PATH}/lib/x64"
    )
endif()

if(UNIX)
    link_directories(
        "${CUDSS_PATH}/lib/12"
        "${CUDA_PATH}/lib64"
    )
endif()

file(GLOB SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/Source/*.cpp"
    "${CMAKE_SOURCE_DIR}/Source/*.cu"
)

add_executable(${PROJECT_NAME} "${CMAKE_SOURCE_DIR}/Source/kernel.cu")

add_library(my_lib ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE my_lib cudss cublas)

set(CMAKE_BUILD_TYPE Release)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CUDSS_PATH}/bin/12/cudss64_0.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()