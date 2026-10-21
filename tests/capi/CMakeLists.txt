add_executable(c_api_test c_api_test.cpp)
target_link_libraries(c_api_test PRIVATE px4rl::cuda CUDA::cudart_static GTest::gtest_main)
add_test(NAME c_api_test COMMAND c_api_test)
