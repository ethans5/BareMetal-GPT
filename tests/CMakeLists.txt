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
