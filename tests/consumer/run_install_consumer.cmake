# Installs the built px4rl to a temporary prefix, then configures, builds and
# runs tests/consumer against it. Invoked by the `install_consumer` ctest.
foreach(var PX4RL_BUILD_DIR PX4RL_CONSUMER_SOURCE_DIR PX4RL_WORK_DIR)
	if(NOT ${var})
		message(FATAL_ERROR "${var} is required")
	endif()
endforeach()

function(run_step)
	execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
	if(NOT result EQUAL 0)
		message(FATAL_ERROR "step failed (${result}): ${ARGN}")
	endif()
endfunction()

set(prefix ${PX4RL_WORK_DIR}/prefix)
set(consumer_build ${PX4RL_WORK_DIR}/build)
file(REMOVE_RECURSE ${PX4RL_WORK_DIR})

run_step(${CMAKE_COMMAND} --install ${PX4RL_BUILD_DIR} --prefix ${prefix})

set(configure_args -S ${PX4RL_CONSUMER_SOURCE_DIR} -B ${consumer_build} -DCMAKE_PREFIX_PATH=${prefix})
if(CMAKE_CXX_COMPILER)
	list(APPEND configure_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
endif()
if(CMAKE_BUILD_TYPE)
	list(APPEND configure_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()
run_step(${CMAKE_COMMAND} ${configure_args})
run_step(${CMAKE_COMMAND} --build ${consumer_build})
run_step(${consumer_build}/px4rl_consumer)
