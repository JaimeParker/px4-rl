cmake_minimum_required(VERSION 3.18)
project(px4rl VERSION 0.1.0 LANGUAGES CXX)

option(PX4RL_BUILD_TESTS "Build the C++ tests" ON)
option(PX4RL_BUILD_CUDA "Build the CUDA library and its tests" OFF)
option(PX4RL_BUILD_PYTHON "Build the Python package files (pybind11 module, generated ctypes structs); pip install . sets it" OFF)
option(PX4RL_OPENMP "Parallelise CpuFleet over vehicles with OpenMP when available" ON)
option(PX4RL_STRICT_FP "Disable FMA contraction (--fmad=false, -ffp-contract=off) for tight CPU/GPU comparison" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PX4RL_BUILD_CUDA)
	# Must be set before enable_language(CUDA). The last entry has no -real suffix, so it also embeds PTX.
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89-real;90")
	endif()
	set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
endif()

if(PX4RL_OPENMP)
	find_package(OpenMP COMPONENTS CXX)
endif()
if(PX4RL_OPENMP AND OpenMP_CXX_FOUND)
	set(PX4RL_OPENMP_ENABLED ON)
else()
	set(PX4RL_OPENMP_ENABLED OFF)
endif()

if(PX4RL_STRICT_FP)
	add_compile_options(
		"$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>"
		"$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -ffp-contract=off>"
	)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(PX4RL_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/px4rl)

if(PX4RL_BUILD_TESTS)
	enable_testing()
endif()

add_subdirectory(px4)
add_subdirectory(src)
if(PX4RL_BUILD_PYTHON)
	add_subdirectory(src/python)
endif()

install(DIRECTORY include/px4rl
	DESTINATION include
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

configure_package_config_file(
	cmake/px4rlConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/px4rlConfig.cmake
	INSTALL_DESTINATION ${PX4RL_CMAKE_INSTALL_DIR}
)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/px4rlConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/px4rlConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/px4rlConfigVersion.cmake
	DESTINATION ${PX4RL_CMAKE_INSTALL_DIR}
)

if(PX4RL_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests/cpp)
	add_subdirectory(tests/fleet)
	add_subdirectory(tests/plant)
	add_subdirectory(tests/position)
	add_subdirectory(tests/closed_loop)
	if(PX4RL_BUILD_CUDA)
		add_subdirectory(tests/parity)
		add_subdirectory(tests/capi)
		add_subdirectory(bench)
	endif()

	add_test(NAME install_consumer
		COMMAND ${CMAKE_COMMAND}
			-DPX4RL_BUILD_DIR=${CMAKE_BINARY_DIR}
			-DPX4RL_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer
			-DPX4RL_WORK_DIR=${CMAKE_BINARY_DIR}/install_consumer_work
			-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
			-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
			-P ${CMAKE_CURRENT_SOURCE_DIR}/tests/consumer/run_install_consumer.cmake
	)
endif()
