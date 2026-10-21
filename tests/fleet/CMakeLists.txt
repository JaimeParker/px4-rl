include(GoogleTest)

add_executable(fleet_cpu_test fleet_cpu_test.cpp)
target_link_libraries(fleet_cpu_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME fleet_cpu_test COMMAND fleet_cpu_test)
