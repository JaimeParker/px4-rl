include(FetchContent)
FetchContent_Declare(googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.14.0
	GIT_SHALLOW TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(px4rl_add_gtest name source)
	add_executable(${name} ${source})
	target_link_libraries(${name} PRIVATE px4rl::px4 GTest::gtest_main)
	gtest_discover_tests(${name})
endfunction()

px4rl_add_gtest(attitude_control_gtest AttitudeControlTest.cpp)
px4rl_add_gtest(rate_control_gtest RateControlTest.cpp)
px4rl_add_gtest(control_allocator_gtest ControlAllocatorTest.cpp)
px4rl_add_gtest(actuator_output_gtest ActuatorOutputTest.cpp)
px4rl_add_gtest(l2_equivalence_gtest L2_EquivalenceTest.cpp)
px4rl_add_gtest(l3_replay_gtest L3_ReplayTest.cpp)
target_compile_definitions(l3_replay_gtest PRIVATE
	L3_TRACE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/l3_trace.csv"
)

# Upstream PX4 v1.16.2 tests, byte-identical (see px4/PROVENANCE.md).
px4rl_add_gtest(position_control_gtest PositionControlTest.cpp)
px4rl_add_gtest(control_math_gtest ControlMathTest.cpp)
# ControlMathTest.cpp takes the M_*_F float constants from px4_platform_common/defines.h, which the
# px4-rl shim lacks; px4_defines.h provides them.
target_compile_options(control_math_gtest PRIVATE -include ${PROJECT_SOURCE_DIR}/px4/lib/px4_defines.h)
px4rl_add_gtest(zero_order_hover_thrust_ekf_gtest zero_order_hover_thrust_ekf_test.cpp)

add_executable(attitude_control_simple_test attitude_control_simple_test.cpp)
target_link_libraries(attitude_control_simple_test PRIVATE px4rl::px4)
add_test(NAME attitude_control_simple_test COMMAND attitude_control_simple_test)
