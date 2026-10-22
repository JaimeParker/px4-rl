#!/usr/bin/env python3
"""
Extract inner-loop controller trace data from a PX4 ULog file.

Reads every relevant topic for the inner-loop control pipeline
(attitude → rate → allocator → actuator output) and merges them into
a single CSV aligned on the vehicle_angular_velocity timestamps
(which trigger PX4's rate controller loop at ~250 Hz).

The columns match the extracted code's I/O structs so the C++ replay
test (tests/cpp/L3_ReplayTest.cpp) can read the CSV directly. This is how
tests/cpp/data/l3_trace.csv was produced.

Requires pyulog (pip install pyulog) and numpy.

Usage:
    python tools/extract_ulog_trace.py <input.ulg> <output.csv>
"""

import sys
import csv
import math
import numpy as np
from pyulog import ULog


def find_nearest(arr: np.ndarray, val: int) -> int:
    """Binary search for the index of the nearest timestamp."""
    idx = int(np.searchsorted(arr, val))
    if idx >= len(arr):
        return len(arr) - 1
    if idx > 0 and abs(float(arr[idx - 1]) - float(val)) < abs(float(arr[idx]) - float(val)):
        return idx - 1
    return idx


def get_dataset(ulog: ULog, name: str):
    """Get a dataset by topic name, or None if not found."""
    for d in ulog.data_list:
        if d.name == name:
            return d
    return None


def extract_trace(ulg_path: str, csv_path: str) -> None:
    print(f"Reading {ulg_path} ...")
    ulog = ULog(ulg_path)

    # ---------- load topics ----------
    att = get_dataset(ulog, "vehicle_attitude")
    att_sp = get_dataset(ulog, "vehicle_attitude_setpoint")
    rates_sp = get_dataset(ulog, "vehicle_rates_setpoint")
    ang_vel = get_dataset(ulog, "vehicle_angular_velocity")
    torque_sp = get_dataset(ulog, "vehicle_torque_setpoint")
    thrust_sp = get_dataset(ulog, "vehicle_thrust_setpoint")
    motors = get_dataset(ulog, "actuator_motors")
    act_out = get_dataset(ulog, "actuator_outputs")

    required = {
        "vehicle_attitude": att,
        "vehicle_attitude_setpoint": att_sp,
        "vehicle_rates_setpoint": rates_sp,
        "vehicle_angular_velocity": ang_vel,
        "actuator_motors": motors,
        "actuator_outputs": act_out,
    }
    for name, ds in required.items():
        if ds is None:
            print(f"ERROR: required topic '{name}' not found in ULog")
            sys.exit(1)

    # Optional topics — warn but continue
    if torque_sp is None:
        print("WARNING: vehicle_torque_setpoint not found — torque columns will be NaN")
    if thrust_sp is None:
        print("WARNING: vehicle_thrust_setpoint not found — (not critical, thrust from att_sp)")

    # ---------- diagnostics ----------
    for name, ds in [("vehicle_attitude", att),
                     ("vehicle_attitude_setpoint", att_sp),
                     ("vehicle_rates_setpoint", rates_sp),
                     ("vehicle_angular_velocity", ang_vel),
                     ("vehicle_torque_setpoint", torque_sp),
                     ("vehicle_thrust_setpoint", thrust_sp),
                     ("actuator_motors", motors),
                     ("actuator_outputs", act_out)]:
        if ds is not None:
            ts = ds.data["timestamp"]
            dt = np.diff(ts) * 1e-6
            print(f"  {name:40s}  samples={len(ts):6d}  "
                  f"dt: min={dt.min():.4f}s  max={dt.max():.4f}s  mean={dt.mean():.4f}s")

    # ---------- reference clock: vehicle_angular_velocity ----------
    ref_ts = ang_vel.data["timestamp"]

    # Check whether angular accel (xyz_derivative) is present
    has_ang_accel = "xyz_derivative[0]" in ang_vel.data

    if not has_ang_accel:
        print("WARNING: vehicle_angular_velocity.xyz_derivative not logged — "
              "angular_accel columns will be 0")

    # Pre-fetch timestamp arrays for binary search
    att_ts = att.data["timestamp"]
    att_sp_ts = att_sp.data["timestamp"]
    rates_sp_ts = rates_sp.data["timestamp"]
    motors_ts = motors.data["timestamp"]
    act_out_ts = act_out.data["timestamp"]
    torque_sp_ts = torque_sp.data["timestamp"] if torque_sp else None
    thrust_sp_ts = thrust_sp.data["timestamp"] if thrust_sp else None

    # ---------- write CSV ----------
    header = [
        "timestamp_us",
        # Attitude state (4 floats)
        "q0", "q1", "q2", "q3",
        # Attitude setpoint (4 + 1 + 1 floats)
        "q_d0", "q_d1", "q_d2", "q_d3",
        "thrust_body_z",
        "yaw_sp_move_rate",
        # Rate setpoint from attitude controller (3 floats)
        "rollspeed_sp", "pitchspeed_sp", "yawspeed_sp",
        # Actual body rates from gyro (3 floats)
        "rollspeed", "pitchspeed", "yawspeed",
        # Angular acceleration (3 floats)
        "angular_accel_x", "angular_accel_y", "angular_accel_z",
        # Torque setpoint from rate controller (3 floats)
        "torque_sp_x", "torque_sp_y", "torque_sp_z",
        # Normalized motor outputs [0,1] (4 floats)
        "motor0", "motor1", "motor2", "motor3",
        # Final PWM / actuator outputs (4 floats)
        "pwm0", "pwm1", "pwm2", "pwm3",
    ]

    nan_count = 0
    skipped_early = 0
    rows_written = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, ts in enumerate(ref_ts):
            # Skip samples before attitude setpoint is published
            if att_sp_ts[0] > ts:
                skipped_early += 1
                continue

            row = [int(ts)]

            # --- Attitude state ---
            idx = find_nearest(att_ts, ts)
            for k in range(4):
                row.append(float(att.data[f"q[{k}]"][idx]))

            # --- Attitude setpoint ---
            idx = find_nearest(att_sp_ts, ts)
            for k in range(4):
                row.append(float(att_sp.data[f"q_d[{k}]"][idx]))
            row.append(float(att_sp.data["thrust_body[2]"][idx]))
            # yaw_sp_move_rate
            if "yaw_sp_move_rate" in att_sp.data:
                row.append(float(att_sp.data["yaw_sp_move_rate"][idx]))
            else:
                row.append(0.0)

            # --- Rate setpoint (attitude controller output) ---
            idx = find_nearest(rates_sp_ts, ts)
            row.append(float(rates_sp.data["roll"][idx]))
            row.append(float(rates_sp.data["pitch"][idx]))
            row.append(float(rates_sp.data["yaw"][idx]))

            # --- Actual body rates ---
            row.append(float(ang_vel.data["xyz[0]"][i]))
            row.append(float(ang_vel.data["xyz[1]"][i]))
            row.append(float(ang_vel.data["xyz[2]"][i]))

            # --- Angular acceleration ---
            if has_ang_accel:
                row.append(float(ang_vel.data["xyz_derivative[0]"][i]))
                row.append(float(ang_vel.data["xyz_derivative[1]"][i]))
                row.append(float(ang_vel.data["xyz_derivative[2]"][i]))
            else:
                row.extend([0.0, 0.0, 0.0])

            # --- Torque setpoint (rate controller output) ---
            if torque_sp is not None:
                idx = find_nearest(torque_sp_ts, ts)
                row.append(float(torque_sp.data["xyz[0]"][idx]))
                row.append(float(torque_sp.data["xyz[1]"][idx]))
                row.append(float(torque_sp.data["xyz[2]"][idx]))
            else:
                row.extend([float("nan")] * 3)

            # --- Motor outputs ---
            idx = find_nearest(motors_ts, ts)
            for m in range(4):
                val = float(motors.data[f"control[{m}]"][idx])
                row.append(val)

            # --- Actuator outputs (PWM) ---
            idx = find_nearest(act_out_ts, ts)
            for m in range(4):
                row.append(float(act_out.data[f"output[{m}]"][idx]))

            # NaN check
            if any(math.isnan(v) for v in row[1:] if isinstance(v, float)):
                nan_count += 1

            writer.writerow(row)
            rows_written += 1

    # --- summary ---
    print(f"\nWrote {rows_written} samples to {csv_path}")
    if skipped_early > 0:
        print(f"  Skipped {skipped_early} early samples (before attitude setpoint published)")
    if nan_count > 0:
        print(f"  WARNING: {nan_count} rows contain NaN values")
    else:
        print("  No NaN values detected")

    # Verify monotonic timestamps
    loaded = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=0)
    if np.all(np.diff(loaded) > 0):
        print("  Timestamps: monotonically increasing ✓")
    else:
        print("  WARNING: timestamps are NOT monotonically increasing")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.ulg> <output.csv>")
        sys.exit(1)
    extract_trace(sys.argv[1], sys.argv[2])
