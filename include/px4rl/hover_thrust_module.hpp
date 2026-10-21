// px4rl::HoverThrustModule: PX4's MulticopterHoverThrustEstimator (v1.16.2) around the vendored
// ZeroOrderHoverThrustEkf.
//
// Reproduces modules/mc_hover_thrust_estimator/MulticopterHoverThrustEstimator.cpp. The uORB topics,
// land detector and vehicle status are replaced by caller flags, and time is a caller-provided
// microsecond counter. Every block below names the PX4 lines it mirrors.
#pragma once

#include <float.h>
#include <math.h>
#include <stdint.h>

#include <px4_defines.h>

#include <hysteresis/hysteresis.h>
#include <mathlib/mathlib.h>
#include <zero_order_hover_thrust_ekf.hpp>

#include "px4rl/params.hpp"

namespace px4rl {

class HoverThrustModule {
public:
	/// Mirrors the constructor (:47-53): hysteresis time, updateParams(), reset().
	PX4RL_HD void configure(const px4rl_params &p)
	{
		_valid_hysteresis.set_hysteresis_time_from(false, kValidHysteresisUs);
		_ht_err_init = p.hte_ht_err_init;
		updateParams(p);
		reset();
	}

	/// Mirrors reset() (:70-75).
	PX4RL_HD void reset()
	{
		_hover_thrust_ekf.setHoverThrust(_mpc_thr_hover);
		_hover_thrust_ekf.setHoverThrustStdDev(_ht_err_init);
		_hover_thrust_ekf.resetAccelNoise();
	}

	/// Mirrors updateParams() (:77-95).
	PX4RL_HD void updateParams(const px4rl_params &p)
	{
		const float ht_err_init_prev = _ht_err_init;
		_ht_err_init = p.hte_ht_err_init;
		_mpc_thr_hover = p.mpc_thr_hover;
		_hte_vxy_thr = p.hte_vxy_thr;
		_hte_vz_thr = p.hte_vz_thr;

		_hover_thrust_ekf.setProcessNoiseStdDev(p.hte_ht_noise);

		if (fabsf(_ht_err_init - ht_err_init_prev) > FLT_EPSILON) {
			_hover_thrust_ekf.setHoverThrustStdDev(_ht_err_init);
		}

		_hover_thrust_ekf.setAccelInnovGate(p.hte_acc_gate);

		_hover_thrust_ekf.setMinHoverThrust(math::constrain(_mpc_thr_hover - p.hte_thr_range, 0.f, 0.8f));
		_hover_thrust_ekf.setMaxHoverThrust(math::constrain(_mpc_thr_hover + p.hte_thr_range, 0.2f, 0.9f));
	}

	/// One Run() cycle (:96-215).
	/// - `armed`, `in_air`: caller flags (PX4 derives them from vehicle_status and the land detector).
	/// - `dt_s`: time since the previous call.
	/// - `acceleration_z`: vehicle_local_position.az, vertical acceleration in NED (positive down).
	/// - `velocity`: NED velocity, used for the measurement noise scaling.
	/// - `q`: attitude, body FRD to NED.
	/// - `thrust_sp_body`, `unallocated_thrust`: normalized thrust setpoint and the allocator's unallocated
	///   part, both body FRD.
	/// - `time_us`: caller's microsecond counter.
	PX4RL_HD void update(bool armed, bool in_air, float dt_s, float acceleration_z, const matrix::Vector3f &velocity,
			     const matrix::Quatf &q, const matrix::Vector3f &thrust_sp_body,
			     const matrix::Vector3f &unallocated_thrust, uint64_t time_us)
	{
		// :160
		if (armed && in_air && (dt_s > 0.001f) && (dt_s < 1.f) && PX4_ISFINITE(acceleration_z)) {

			_hover_thrust_ekf.predict(dt_s);

			// :177-180
			const matrix::Vector3f thrust_body_allocated = thrust_sp_body - unallocated_thrust;
			const matrix::Vector3f thrust_allocated = q.rotateVector(thrust_body_allocated);

			if (PX4_ISFINITE(thrust_allocated(2))) {
				// Guard against fast up and down motions biasing the estimator due to large drag and prop wash effects
				const float meas_noise_coeff_z = fmaxf((fabsf(velocity(2)) - _hte_vz_thr) + 1.f, 1.f);
				const float meas_noise_coeff_xy = fmaxf((matrix::Vector2f(velocity(0), velocity(1)).norm() - _hte_vxy_thr)
									+ 1.f, 1.f);

				_hover_thrust_ekf.setMeasurementNoiseScale(fmaxf(meas_noise_coeff_xy, meas_noise_coeff_z));
				_hover_thrust_ekf.fuseAccZ(-acceleration_z, -thrust_allocated(2));

				bool valid = (_hover_thrust_ekf.getHoverThrustEstimateVar() < 0.001f);

				// The test ratio does not need to pass all the time to have a valid estimate
				if (!_valid) {
					valid = valid && (_hover_thrust_ekf.getInnovationTestRatio() < 1.f);
				}

				_valid_hysteresis.set_state_and_update(valid, time_us);
				_valid = _valid_hysteresis.get_state();
			}

		} else {
			// :208-222
			_valid_hysteresis.set_state_and_update(false, time_us);

			if (!armed) {
				reset();
			}

			_valid = false;
		}
	}

	PX4RL_HD float hoverThrust() const { return _hover_thrust_ekf.getHoverThrustEstimate(); }
	PX4RL_HD bool valid() const { return _valid; }

private:
	static constexpr uint64_t kValidHysteresisUs = 2000000; // 2_s

	ZeroOrderHoverThrustEkf _hover_thrust_ekf{};
	systemlib::Hysteresis _valid_hysteresis{false};
	bool _valid{false};

	float _ht_err_init{0.f};
	float _mpc_thr_hover{0.5f};
	float _hte_vxy_thr{0.f};
	float _hte_vz_thr{0.f};
};

} // namespace px4rl
