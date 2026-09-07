cmake_minimum_required(VERSION 3.20)
project(BareMetal-GPT
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Bibliothèque principale ----------
add_library(baremetalgpt
    src/Tensor.cpp
    src/matmul.cpp
)

target_include_directories(baremetalgpt
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Warnings strictes — zéro tolérance
target_compile_options(baremetalgpt
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Werror>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
)

# ---------- Tests ----------
option(BAREMETALGPT_BUILD_TESTS "Build unit tests" ON)

if(BAREMETALGPT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
