add_executable(parity_test parity_test.cpp)
target_link_libraries(parity_test PRIVATE px4rl::fleet px4rl::cuda CUDA::cudart_static GTest::gtest_main)
add_test(NAME parity_test COMMAND parity_test)
