include(GoogleTest)

add_executable(closed_loop_test closed_loop_test.cpp)
target_link_libraries(closed_loop_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME closed_loop_test COMMAND closed_loop_test)

add_executable(iris_closed_loop_test iris_closed_loop_test.cpp)
target_link_libraries(iris_closed_loop_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME iris_closed_loop_test COMMAND iris_closed_loop_test)
