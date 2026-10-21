include(GoogleTest)

add_executable(fleet_cpu_test fleet_cpu_test.cpp)
target_link_libraries(fleet_cpu_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME fleet_cpu_test COMMAND fleet_cpu_test)

add_executable(fleet_cpu_threading_test fleet_cpu_threading_test.cpp)
target_link_libraries(fleet_cpu_threading_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME fleet_cpu_threading_test COMMAND fleet_cpu_threading_test)

# OpenMP's default thread count (all cores) with busy-waiting barriers stalls badly on shared or loaded
# machines; the results do not depend on the thread count, so the tests use a small fixed team.
set_tests_properties(fleet_cpu_test PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=4")
set_tests_properties(fleet_cpu_threading_test PROPERTIES ENVIRONMENT "OMP_WAIT_POLICY=passive")
