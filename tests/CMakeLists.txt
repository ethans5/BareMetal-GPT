# ---------- Google Test via FetchContent ----------
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

# Empêche GTest d'installer ses propres targets dans notre install
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# ---------- Test executable ----------
add_executable(test_tensor
    test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        baremetalgpt
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_tensor)

# ---------- MatMul tests ----------
add_executable(test_matmul
    test_matmul.cpp
)

target_link_libraries(test_matmul
    PRIVATE
        baremetalgpt
        GTest::gtest_main
)

gtest_discover_tests(test_matmul)

# ---------- WeightLoader tests ----------
add_executable(test_weight_loader
    test_weight_loader.cpp
)

target_link_libraries(test_weight_loader
    PRIVATE
        baremetalgpt
        GTest::gtest_main
)

# Inject absolute paths at compile time so tests locate weight files
# regardless of the working directory used by ctest.
target_compile_definitions(test_weight_loader
    PRIVATE
        WEIGHTS_DIR=${CMAKE_SOURCE_DIR}/weights
)

gtest_discover_tests(test_weight_loader)

# ---------- LayerNorm tests ----------
add_executable(test_layer_norm
    test_layer_norm.cpp
)

target_link_libraries(test_layer_norm
    PRIVATE
        baremetalgpt
        GTest::gtest_main
)

# Inject absolute paths at compile time so tests locate weight + dump files
# regardless of the working directory used by ctest.
target_compile_definitions(test_layer_norm
    PRIVATE
        WEIGHTS_DIR=${CMAKE_SOURCE_DIR}/weights
        DUMPS_DIR=${CMAKE_SOURCE_DIR}/reference_dumps/micro
)

gtest_discover_tests(test_layer_norm)
