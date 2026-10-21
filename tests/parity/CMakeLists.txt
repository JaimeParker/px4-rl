add_executable(parity_test parity_test.cpp)
target_link_libraries(parity_test PRIVATE px4rl::fleet px4rl::cuda CUDA::cudart_static GTest::gtest_main)
add_test(NAME parity_test COMMAND parity_test)
# See tests/fleet/CMakeLists.txt: a small OpenMP team avoids busy-wait stalls on loaded machines.
set_tests_properties(parity_test PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=4")
