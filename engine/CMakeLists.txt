cmake_minimum_required(VERSION 3.20)

# Prefer clang++-18 by default if available (can still override via -DCMAKE_CXX_COMPILER)
if(NOT CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/clang++-18")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18" CACHE FILEPATH "" FORCE)
endif()

# Ensure CMake's compiler check uses libc++ with Clang to avoid -lstdc++
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-stdlib=libc++ -lc++abi")

project(chess_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Download single-header chess library during configure
set(CHESS_URL "https://raw.githubusercontent.com/Disservin/chess-library/master/include/chess.hpp")
set(THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third_party/chess-library")
set(CHESS_HEADER "${THIRD_PARTY_DIR}/chess.hpp")

file(MAKE_DIRECTORY "${THIRD_PARTY_DIR}")
if(NOT EXISTS "${CHESS_HEADER}")
    message(STATUS "Downloading chess.hpp from ${CHESS_URL}")
    file(DOWNLOAD "${CHESS_URL}" "${CHESS_HEADER}" SHOW_PROGRESS TLS_VERIFY ON)
endif()

add_executable(chess_sample src/main.cpp)
target_include_directories(chess_sample PRIVATE "${THIRD_PARTY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(chess_sample PRIVATE _LIBCPP_ENABLE_EXPERIMENTAL)

# Link pthreads for threading support (required for std::jthread and friends)
find_package(Threads REQUIRED)
target_link_libraries(chess_sample PRIVATE Threads::Threads)

# Build and link against libc++ when using Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(chess_sample PRIVATE -stdlib=libc++)
    target_link_options(chess_sample PRIVATE -stdlib=libc++ -lc++abi)
    # Use lld if available for faster linking (optional)
    find_program(LLD_BIN NAMES ld.lld-18 ld.lld lld)
    if(LLD_BIN)
        target_link_options(chess_sample PRIVATE -fuse-ld=lld)
    endif()
endif()


# -----------------------------
# PoC: parallelism/coroutines demo
# -----------------------------
add_executable(poc_parallel
    src/poc/main_poc.cpp
)
target_include_directories(poc_parallel PRIVATE "${THIRD_PARTY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(poc_parallel PRIVATE _LIBCPP_ENABLE_EXPERIMENTAL)
target_link_libraries(poc_parallel PRIVATE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(poc_parallel PRIVATE -stdlib=libc++)
    target_link_options(poc_parallel PRIVATE -stdlib=libc++ -lc++abi)
    find_program(LLD_BIN NAMES ld.lld-18 ld.lld lld)
    if(LLD_BIN)
        target_link_options(poc_parallel PRIVATE -fuse-ld=lld)
    endif()
endif()


# ---------------------------------
# TensorRT + CUDA for poc_parallel
# ---------------------------------
find_package(CUDAToolkit)

set(TENSORRT_HINT_DIRS
    /usr
    /usr/local
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/lib64
    /opt/tensorrt
    /usr/local/TensorRT
)

find_library(TENSORRT_NVINFER nvinfer HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_library(TENSORRT_NVPARSERS nvparsers HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_path(TENSORRT_INCLUDE NvInfer.h HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)

if (CUDAToolkit_FOUND AND TENSORRT_NVINFER AND TENSORRT_INCLUDE)
    message(STATUS "Found CUDA and TensorRT; enabling TensorRT in targets")
    # poc_parallel
    target_compile_definitions(poc_parallel PRIVATE USE_TENSORRT)
    target_include_directories(poc_parallel PRIVATE ${TENSORRT_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(poc_parallel PRIVATE ${TENSORRT_NVINFER})
    if (CUDAToolkit_FOUND)
        target_link_libraries(poc_parallel PRIVATE CUDA::cudart)
    endif()
    if (TENSORRT_NVPARSERS)
        target_link_libraries(poc_parallel PRIVATE ${TENSORRT_NVPARSERS})
    endif()

    # chess_sample also uses NNEvaluator; link TRT/CUDA as well
    target_compile_definitions(chess_sample PRIVATE USE_TENSORRT)
    target_include_directories(chess_sample PRIVATE ${TENSORRT_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(chess_sample PRIVATE ${TENSORRT_NVINFER})
    if (CUDAToolkit_FOUND)
        target_link_libraries(chess_sample PRIVATE CUDA::cudart)
    endif()
else()
    message(WARNING "CUDA/TensorRT not found; poc_parallel will build but TensorRT runtime will not be linked.")
endif()


