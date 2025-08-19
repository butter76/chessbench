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

# Threads
find_package(Threads REQUIRED)

# Download single-header chess library during configure
set(CHESS_URL "https://raw.githubusercontent.com/Disservin/chess-library/master/include/chess.hpp")
set(THIRD_PARTY_DIR "${CMAKE_BINARY_DIR}/third_party/chess-library")
set(CHESS_HEADER "${THIRD_PARTY_DIR}/chess.hpp")

file(MAKE_DIRECTORY "${THIRD_PARTY_DIR}")
if(NOT EXISTS "${CHESS_HEADER}")
    message(STATUS "Downloading chess.hpp from ${CHESS_URL}")
    file(DOWNLOAD "${CHESS_URL}" "${CHESS_HEADER}" SHOW_PROGRESS TLS_VERIFY ON)
endif()

add_executable(chess_sample
    src/main.cpp
    src/syzygy_handler.cpp
    src/time/fixed_time.cpp
)
target_include_directories(chess_sample PRIVATE "${THIRD_PARTY_DIR}")
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


# Proof-of-concept: coroutine work-stealing pool + mock evaluator
add_executable(coroutines_poc
    src/poc/demo.cpp
    src/poc/async_pool.cpp
    src/poc/nn_evaluator.cpp
)
target_link_libraries(coroutines_poc PRIVATE Threads::Threads)


