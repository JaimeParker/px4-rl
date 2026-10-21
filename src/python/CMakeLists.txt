# Python package: the pybind11 module px4rl._cpu, the generated ctypes structs and, with CUDA, the shared library.
# The pure-Python sources are in python/px4rl; scikit-build-core (pyproject.toml) packages both together.
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

set(PX4RL_PYTHON_INSTALL_DIR px4rl CACHE STRING
	"Install directory of the Python package's compiled and generated files, relative to the install prefix")
# Only the rules below belong in a wheel, not the headers and static libraries that the C++ install rules cover.
set(PX4RL_PYTHON_COMPONENT python)

pybind11_add_module(px4rl_cpu_module MODULE bindings.cpp)
set_target_properties(px4rl_cpu_module PROPERTIES OUTPUT_NAME _cpu)
target_link_libraries(px4rl_cpu_module PRIVATE px4rl::fleet)
# Keep the px4rl_* C symbols of the static libraries out of the module's dynamic symbol table, so that it can
# share a process with libpx4rl_cuda.so whatever the dlopen flags are.
target_link_options(px4rl_cpu_module PRIVATE LINKER:--exclude-libs,ALL)
install(TARGETS px4rl_cpu_module
	LIBRARY DESTINATION ${PX4RL_PYTHON_INSTALL_DIR}
	COMPONENT ${PX4RL_PYTHON_COMPONENT}
)

# _structs.py comes from the headers on every build, so the ctypes structs cannot drift from the C ones.
set(PX4RL_GENERATED_STRUCTS ${CMAKE_CURRENT_BINARY_DIR}/_structs.py)
add_custom_command(
	OUTPUT ${PX4RL_GENERATED_STRUCTS}
	COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/gen_ctypes.py
		--output ${PX4RL_GENERATED_STRUCTS}
		${PROJECT_SOURCE_DIR}/include/px4rl/c_api.h
		${PROJECT_SOURCE_DIR}/include/px4rl/plant.h
	DEPENDS
		${PROJECT_SOURCE_DIR}/tools/gen_ctypes.py
		${PROJECT_SOURCE_DIR}/include/px4rl/c_api.h
		${PROJECT_SOURCE_DIR}/include/px4rl/plant.h
	VERBATIM
)
add_custom_target(px4rl_structs ALL DEPENDS ${PX4RL_GENERATED_STRUCTS})

configure_file(_version.py.in ${CMAKE_CURRENT_BINARY_DIR}/_version.py @ONLY)

install(FILES ${PX4RL_GENERATED_STRUCTS} ${CMAKE_CURRENT_BINARY_DIR}/_version.py
	DESTINATION ${PX4RL_PYTHON_INSTALL_DIR}
	COMPONENT ${PX4RL_PYTHON_COMPONENT}
)

if(PX4RL_BUILD_CUDA)
	install(FILES $<TARGET_FILE:px4rl_cuda>
		DESTINATION ${PX4RL_PYTHON_INSTALL_DIR}
		COMPONENT ${PX4RL_PYTHON_COMPONENT}
	)
endif()
