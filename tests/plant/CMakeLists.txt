add_executable(plant_test plant_test.cpp)
target_link_libraries(plant_test PRIVATE px4rl::fleet GTest::gtest_main)
add_test(NAME plant_test COMMAND plant_test)

if(PX4RL_BUILD_CUDA)
	# Only the headers are used, compiled with PX4RL_HD for device code. px4rl::fleet is deliberately
	# not linked: it would propagate the empty PX4RL_HD definition of the CPU build.
	add_executable(plant_cuda_test plant_cuda_test.cu)
	target_link_libraries(plant_cuda_test PRIVATE px4rl_px4_headers CUDA::cudart_static)
	target_include_directories(plant_cuda_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
	target_compile_definitions(plant_cuda_test PRIVATE "PX4RL_HD=__host__ __device__")
	target_compile_options(plant_cuda_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	add_test(NAME plant_cuda_test COMMAND plant_cuda_test)
endif()
