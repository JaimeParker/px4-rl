# Include directories are the layout the vendored sources expect, e.g.
# <matrix/matrix/math.hpp>, <mathlib/math/Functions.hpp>, "RateControl.hpp".
set(PX4RL_PX4_INCLUDE_SUBDIRS
	lib
	attitude_control
	rate_control
	control_allocator
	actuator_output
	position_control
	hover_thrust_estimator
)
# Include directories only: the files below are installed through the entries above.
# `.` resolves <lib/mathlib/mathlib.h>, lib/matrix resolves <matrix/math.hpp>, as written upstream.
set(PX4RL_PX4_INCLUDE_ONLY_SUBDIRS
	.
	lib/matrix
)

# Headers only: PX4RL_HD is deliberately not defined here, so the CUDA target can define it differently.
add_library(px4rl_px4_headers INTERFACE)
foreach(subdir IN LISTS PX4RL_PX4_INCLUDE_SUBDIRS PX4RL_PX4_INCLUDE_ONLY_SUBDIRS)
	target_include_directories(px4rl_px4_headers INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${subdir}>
		$<INSTALL_INTERFACE:include/px4rl/px4/${subdir}>
	)
endforeach()
set_target_properties(px4rl_px4_headers PROPERTIES EXPORT_NAME px4_headers)
add_library(px4rl::px4_headers ALIAS px4rl_px4_headers)

add_library(px4rl_px4 STATIC
	attitude_control/AttitudeControl.cpp
	attitude_control/AttitudeControllerWrapper.cpp
	rate_control/RateControl.cpp
	rate_control/RateControllerWrapper.cpp
	control_allocator/ActuatorEffectiveness.cpp
	control_allocator/ControlAllocation.cpp
	control_allocator/ControlAllocationPseudoInverse.cpp
	control_allocator/ControlAllocationSequentialDesaturation.cpp
	control_allocator/ControlAllocatorWrapper.cpp
	actuator_output/ActuatorOutputWrapper.cpp
	position_control/PositionControl.cpp
	position_control/ControlMath.cpp
	hover_thrust_estimator/zero_order_hover_thrust_ekf.cpp
	lib/hysteresis/hysteresis.cpp
)
add_library(px4rl::px4 ALIAS px4rl_px4)
set_target_properties(px4rl_px4 PROPERTIES
	EXPORT_NAME px4
	POSITION_INDEPENDENT_CODE ON
)
target_compile_features(px4rl_px4 PUBLIC cxx_std_17)
target_link_libraries(px4rl_px4 PUBLIC px4rl_px4_headers)
target_compile_definitions(px4rl_px4 PUBLIC PX4RL_HD=)
# -Wno-unused-parameter matches PX4's own flags (cmake/px4_add_common_flags.cmake).
target_compile_options(px4rl_px4 PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)

install(TARGETS px4rl_px4_headers px4rl_px4
	EXPORT px4rlTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
foreach(subdir IN LISTS PX4RL_PX4_INCLUDE_SUBDIRS)
	install(DIRECTORY ${subdir}/
		DESTINATION include/px4rl/px4/${subdir}
		FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
	)
endforeach()
install(EXPORT px4rlTargets
	NAMESPACE px4rl::
	DESTINATION ${PX4RL_CMAKE_INSTALL_DIR}
)
