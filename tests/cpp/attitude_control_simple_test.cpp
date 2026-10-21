/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file attitude_control_simple_test.cpp
 *
 * Simple test for attitude controller extraction
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "AttitudeControllerWrapper.hpp"

using namespace px4_stack;
using namespace matrix;

void printVector3(const char* name, const Vector3f& v) {
    std::cout << name << ": ["
              << std::fixed << std::setprecision(4)
              << v(0) << ", " << v(1) << ", " << v(2) << "]" << std::endl;
}

void printQuat(const char* name, const Quatf& q) {
    std::cout << name << ": [w="
              << std::fixed << std::setprecision(4)
              << q(0) << ", x=" << q(1) << ", y=" << q(2) << ", z=" << q(3) << "]" << std::endl;
}

int main() {
    std::cout << "=== PX4 Attitude Controller Simple Test ===" << std::endl << std::endl;

    // Create controller with default gains
    AttitudeControllerWrapper controller;

    std::cout << "Controller initialized with default gains:" << std::endl;
    auto config = controller.getGains();
    std::cout << "  Roll P: " << config.roll_p << std::endl;
    std::cout << "  Pitch P: " << config.pitch_p << std::endl;
    std::cout << "  Yaw P: " << config.yaw_p << std::endl;
    std::cout << "  Yaw Weight: " << config.yaw_weight << std::endl;
    std::cout << std::endl;

    // Test 1: Zero error (current attitude equals desired attitude)
    std::cout << "Test 1: Zero attitude error" << std::endl;
    std::cout << "-------------------------" << std::endl;

    AttitudeState state1;
    state1.q = Quatf(1.0f, 0.0f, 0.0f, 0.0f);  // Identity quaternion (no rotation)
    state1.timestamp = 0;

    AttitudeSetpoint setpoint1;
    setpoint1.q_d = Quatf(1.0f, 0.0f, 0.0f, 0.0f);  // Same as current
    setpoint1.yaw_sp_move_rate = 0.0f;
    setpoint1.thrust_body = Vector3f(0.0f, 0.0f, -0.5f);  // 50% thrust down

    printQuat("Current attitude", state1.q);
    printQuat("Desired attitude", setpoint1.q_d);

    RateSetpoint output1 = controller.update(state1, setpoint1);

    printVector3("Rate setpoint", output1.rates);
    printVector3("Thrust", output1.thrust_body);
    std::cout << "Expected: rates ≈ [0, 0, 0]" << std::endl;
    std::cout << std::endl;

    // Test 2: 15 degree roll error
    std::cout << "Test 2: 15 degree roll error" << std::endl;
    std::cout << "-------------------------" << std::endl;

    AttitudeState state2;
    state2.q = Quatf(1.0f, 0.0f, 0.0f, 0.0f);  // Level

    float roll_angle = 15.0f * M_PI / 180.0f;  // 15 degrees in radians
    AttitudeSetpoint setpoint2;
    // Quaternion for roll rotation: q = [cos(θ/2), sin(θ/2), 0, 0]
    setpoint2.q_d = Quatf(cosf(roll_angle/2.0f), sinf(roll_angle/2.0f), 0.0f, 0.0f);
    setpoint2.yaw_sp_move_rate = 0.0f;
    setpoint2.thrust_body = Vector3f(0.0f, 0.0f, -0.5f);

    printQuat("Current attitude", state2.q);
    printQuat("Desired attitude", setpoint2.q_d);

    RateSetpoint output2 = controller.update(state2, setpoint2);

    printVector3("Rate setpoint", output2.rates);
    printVector3("Thrust", output2.thrust_body);
    std::cout << "Expected: positive roll rate (output2.rates[0] > 0)" << std::endl;
    std::cout << std::endl;

    // Test 3: Yaw rate feedforward
    std::cout << "Test 3: Yaw rate feedforward" << std::endl;
    std::cout << "-------------------------" << std::endl;

    AttitudeState state3;
    state3.q = Quatf(1.0f, 0.0f, 0.0f, 0.0f);  // Level

    AttitudeSetpoint setpoint3;
    setpoint3.q_d = Quatf(1.0f, 0.0f, 0.0f, 0.0f);  // No attitude error
    setpoint3.yaw_sp_move_rate = 0.5f;  // 0.5 rad/s yaw rate command
    setpoint3.thrust_body = Vector3f(0.0f, 0.0f, -0.5f);

    printQuat("Current attitude", state3.q);
    printQuat("Desired attitude", setpoint3.q_d);
    std::cout << "Yaw rate feedforward: " << setpoint3.yaw_sp_move_rate << " rad/s" << std::endl;

    RateSetpoint output3 = controller.update(state3, setpoint3);

    printVector3("Rate setpoint", output3.rates);
    printVector3("Thrust", output3.thrust_body);
    std::cout << "Expected: yaw rate ≈ 0.5 rad/s (output3.rates[2] ≈ 0.5)" << std::endl;
    std::cout << std::endl;

    // Test 4: Custom gains
    std::cout << "Test 4: Custom gains" << std::endl;
    std::cout << "-------------------------" << std::endl;

    AttitudeControlConfig custom_config;
    custom_config.roll_p = 10.0f;
    custom_config.pitch_p = 10.0f;
    custom_config.yaw_p = 5.0f;
    custom_config.yaw_weight = 0.5f;

    controller.setGains(custom_config);

    std::cout << "Set custom gains:" << std::endl;
    std::cout << "  Roll P: " << custom_config.roll_p << std::endl;
    std::cout << "  Pitch P: " << custom_config.pitch_p << std::endl;
    std::cout << "  Yaw P: " << custom_config.yaw_p << std::endl;

    // Same 15 degree roll error as Test 2
    RateSetpoint output4 = controller.update(state2, setpoint2);

    printVector3("Rate setpoint with higher gain", output4.rates);
    std::cout << "Expected: higher roll rate than Test 2 (≈ 10/6.5 times)" << std::endl;
    std::cout << std::endl;

    std::cout << "=== All tests completed ===" << std::endl;

    return 0;
}
