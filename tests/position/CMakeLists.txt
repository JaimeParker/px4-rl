include(GoogleTest)

add_executable(position_module_test position_module_test.cpp)
target_link_libraries(position_module_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME position_module_test COMMAND position_module_test)

add_executable(hover_thrust_module_test hover_thrust_module_test.cpp)
target_link_libraries(hover_thrust_module_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME hover_thrust_module_test COMMAND hover_thrust_module_test)
