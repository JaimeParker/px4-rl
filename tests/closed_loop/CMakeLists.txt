include(GoogleTest)

add_executable(closed_loop_test closed_loop_test.cpp)
target_link_libraries(closed_loop_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME closed_loop_test COMMAND closed_loop_test)
