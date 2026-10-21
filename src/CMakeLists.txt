add_library(px4rl_fleet STATIC
	cpu/cpu_fleet.cpp
	cpu/c_params.cpp
	cpu/c_hover_thrust.cpp
	airframes/iris.cpp
)
add_library(px4rl::fleet ALIAS px4rl_fleet)
set_target_properties(px4rl_fleet PROPERTIES
	EXPORT_NAME fleet
	POSITION_INDEPENDENT_CODE ON
)
target_include_directories(px4rl_fleet PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)
target_link_libraries(px4rl_fleet PUBLIC px4rl::px4)
target_compile_options(px4rl_fleet PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
if(PX4RL_OPENMP_ENABLED)
	target_link_libraries(px4rl_fleet PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS px4rl_fleet
	EXPORT px4rlTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(PX4RL_BUILD_CUDA)
	add_library(px4rl_cuda SHARED cuda/px4rl_cuda.cu)
	add_library(px4rl::cuda ALIAS px4rl_cuda)
	set_target_properties(px4rl_cuda PROPERTIES
		EXPORT_NAME cuda
		CUDA_RUNTIME_LIBRARY Static
		CXX_VISIBILITY_PRESET hidden
		CUDA_VISIBILITY_PRESET hidden
		VISIBILITY_INLINES_HIDDEN ON
	)
	# The px4 sources are compiled into the .cu (unity include), so only their headers are linked.
	target_link_libraries(px4rl_cuda PRIVATE px4rl_px4_headers)
	target_include_directories(px4rl_cuda PUBLIC
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)
	target_compile_definitions(px4rl_cuda PRIVATE "PX4RL_HD=__host__ __device__")
	target_compile_options(px4rl_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
	target_link_options(px4rl_cuda PRIVATE LINKER:--exclude-libs,ALL)

	find_program(PX4RL_NM nm REQUIRED)
	set(px4rl_check_exports
		${CMAKE_COMMAND} -DNM=${PX4RL_NM} -DLIBRARY=$<TARGET_FILE:px4rl_cuda>
		-P ${PROJECT_SOURCE_DIR}/cmake/CheckCudaExports.cmake
	)
	add_custom_command(TARGET px4rl_cuda POST_BUILD COMMAND ${px4rl_check_exports} VERBATIM)
	if(PX4RL_BUILD_TESTS)
		add_test(NAME cuda_exports COMMAND ${px4rl_check_exports})
	endif()

	install(TARGETS px4rl_cuda
		EXPORT px4rlTargets
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endif()
