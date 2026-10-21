# Prints `nm -D --defined-only` for libpx4rl_cuda.so and fails unless it exports exactly the
# px4rl_* C API. Invoked as: cmake -DNM=<nm> -DLIBRARY=<.so> -P CheckCudaExports.cmake
cmake_minimum_required(VERSION 3.18)
foreach(var NM LIBRARY)
	if(NOT ${var})
		message(FATAL_ERROR "${var} is required")
	endif()
endforeach()

execute_process(COMMAND ${NM} -D --defined-only ${LIBRARY}
	OUTPUT_VARIABLE nm_output RESULT_VARIABLE nm_result)
if(NOT nm_result EQUAL 0)
	message(FATAL_ERROR "nm failed (${nm_result}) on ${LIBRARY}")
endif()
message(STATUS "nm -D --defined-only ${LIBRARY}:\n${nm_output}")

set(expected
	px4rl_params_default px4rl_params_iris px4rl_last_error
	px4rl_plant_params_iris px4rl_rigid_body_params_iris px4rl_hover_thrust
	px4rl_params_size px4rl_plant_params_size px4rl_rigid_body_params_size
	px4rl_cuda_fleet_create px4rl_cuda_fleet_destroy px4rl_cuda_fleet_size px4rl_cuda_fleet_reset
	px4rl_cuda_fleet_reset_plant
	px4rl_cuda_position_step px4rl_cuda_attitude_step px4rl_cuda_rate_step px4rl_cuda_rotor_step
)

string(REPLACE "\n" ";" lines "${nm_output}")
set(exported "")
set(unexpected "")
foreach(line IN LISTS lines)
	if(line MATCHES "^[0-9a-fA-F]+ +[A-Za-z] +([^ ]+)$")
		set(name ${CMAKE_MATCH_1})
		if(name MATCHES "^px4rl_")
			list(APPEND exported ${name})
		else()
			list(APPEND unexpected ${name})
		endif()
	endif()
endforeach()

if(unexpected)
	message(FATAL_ERROR "libpx4rl_cuda exports non-px4rl_ symbols: ${unexpected}")
endif()
foreach(name IN LISTS expected)
	if(NOT name IN_LIST exported)
		message(FATAL_ERROR "libpx4rl_cuda does not export ${name}")
	endif()
endforeach()
foreach(name IN LISTS exported)
	if(NOT name IN_LIST expected)
		message(FATAL_ERROR "libpx4rl_cuda exports unexpected ${name}")
	endif()
endforeach()
message(STATUS "libpx4rl_cuda exports only the px4rl_* C API")
