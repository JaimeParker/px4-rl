# PX4 Iris / x500 quadrotor parameters: raw values and sources

Raw values copied verbatim from the sources. Nothing is converted, computed, or reconciled.

## Conventions and provenance

| Item | Value |
|---|---|
| PX4 checkout | PX4-Autopilot (https://github.com/PX4/PX4-Autopilot), tag `v1.16.2`, HEAD `54f0455ffcd755534539a7cf33a09a20bf71d29d` |
| Gazebo Classic model submodule | `Tools/simulation/gazebo-classic/sitl_gazebo-classic` at `70683dc759cbd3ebccfa78429b564f8b1fb5d149` (submodule not checked out) |
| Gazebo Classic Iris SDF used | `https://raw.githubusercontent.com/PX4/PX4-SITL_gazebo-classic/70683dc759cbd3ebccfa78429b564f8b1fb5d149/models/iris/iris.sdf.jinja` |
| `models/iris/iris.sdf` | HTTP 404 at that commit (not used) |
| gz submodule | `Tools/simulation/gz` at `e05f4312d3f28aa621157610584a4870406cb6d3` (submodule not checked out) |
| x500 SDFs used | `https://raw.githubusercontent.com/PX4/PX4-gazebo-models/e05f4312d3f28aa621157610584a4870406cb6d3/models/x500/model.sdf` and `.../models/x500_base/model.sdf` |

Source-location notation: `iris.sdf.jinja:N` and `x500_base_model.sdf:N` are line numbers in the downloaded files. `x500_model.sdf:N` is the wrapper file. Paths like `src/...` and `ROMFS/...` are relative to the PX4 checkout root.

---

## 1. Gazebo Classic Iris physical model (`iris.sdf.jinja`)

### 1a. Base link inertia

| Name | Value | Source |
|---|---|---|
| base_link inertial pose | `0 0 0 0 0 0` | `iris.sdf.jinja:7` |
| base_link mass | `1.5` | `iris.sdf.jinja:8` (`<mass>1.5</mass>`) |
| ixx | `0.029125` | `iris.sdf.jinja:10` |
| ixy | `0` | `iris.sdf.jinja:11` |
| ixz | `0` | `iris.sdf.jinja:12` |
| iyy | `0.029125` | `iris.sdf.jinja:13` |
| iyz | `0` | `iris.sdf.jinja:14` |
| izz | `0.055225` | `iris.sdf.jinja:15` |

Raw XML:
```xml
<inertial>
  <pose>0 0 0 0 0 0</pose>
  <mass>1.5</mass>
  <inertia>
    <ixx>0.029125</ixx>
    <ixy>0</ixy>
    <ixz>0</ixz>
    <iyy>0.029125</iyy>
    <iyz>0</iyz>
    <izz>0.055225</izz>
  </inertia>
</inertial>
```

### 1b. Rotor link poses, mass and inertia

| Name | Value | Source |
|---|---|---|
| rotor_0 pose | `0.13 -0.22 0.023 0 0 0` | `iris.sdf.jinja:89` (`<pose>0.13 -0.22 0.023 0 0 0</pose>`) |
| rotor_1 pose | `-0.13 0.2 0.023 0 0 0` | `iris.sdf.jinja:154` |
| rotor_2 pose | `0.13 0.22 0.023 0 0 0` | `iris.sdf.jinja:219` |
| rotor_3 pose | `-0.13 -0.2 0.023 0 0 0` | `iris.sdf.jinja:284` |
| rotor_0..3 mass | `0.005` (each) | `iris.sdf.jinja:92`, `:157`, `:222`, `:287` |
| rotor_0..3 ixx | `9.75e-07` (each) | `iris.sdf.jinja:94`, `:159`, `:224`, `:289` |
| rotor_0..3 iyy | `0.000273104` (each) | `iris.sdf.jinja:97`, `:162`, `:227`, `:292` |
| rotor_0..3 izz | `0.000274004` (each) | `iris.sdf.jinja:99`, `:164`, `:229`, `:294` |

Rotor joints: `rotor_0_joint` (`:137`), `rotor_1_joint` (`:202`), `rotor_2_joint` (`:267`), `rotor_3_joint` (`:332`); all `type='revolute'`, axis `0 0 1`, limit `-1e+16` to `1e+16`.

### 1c. Motor plugin parameters (`libgazebo_motor_model.so`)

Each plugin block is one rotor. Values are listed in order rotor_0 / rotor_1 / rotor_2 / rotor_3.

| Name | Value (rotor_0 / rotor_1 / rotor_2 / rotor_3) | Source (lines for rotor_0 / rotor_1 / rotor_2 / rotor_3) |
|---|---|---|
| plugin `name` | `front_right_motor_model` / `back_left_motor_model` / `front_left_motor_model` / `back_right_motor_model` | `:353` / `:370` / `:387` / `:404` |
| jointName | `rotor_0_joint` / `rotor_1_joint` / `rotor_2_joint` / `rotor_3_joint` | `:355` / `:372` / `:389` / `:406` |
| linkName | `rotor_0` / `rotor_1` / `rotor_2` / `rotor_3` | `:356` / `:373` / `:390` / `:407` |
| turningDirection | `ccw` / `ccw` / `cw` / `cw` | `:357` / `:374` / `:391` / `:408` |
| timeConstantUp | `0.0125` (all) | `:358` / `:375` / `:392` / `:409` |
| timeConstantDown | `0.025` (all) | `:359` / `:376` / `:393` / `:410` |
| maxRotVelocity | `1100` (all) | `:360` / `:377` / `:394` / `:411` |
| motorConstant | `5.84e-06` (all) | `:361` / `:378` / `:395` / `:412` |
| momentConstant | `0.06` (all) | `:362` / `:379` / `:396` / `:413` |
| commandSubTopic | `/gazebo/command/motor_speed` (all) | `:363` / `:380` / `:397` / `:414` |
| motorNumber | `0` / `1` / `2` / `3` | `:364` / `:381` / `:398` / `:415` |
| rotorDragCoefficient | `0.000175` (all) | `:365` / `:382` / `:399` / `:416` |
| rollingMomentCoefficient | `1e-06` (all) | `:366` / `:383` / `:400` / `:417` |
| motorSpeedPubTopic | `/motor_speed/0` / `/motor_speed/1` / `/motor_speed/2` / `/motor_speed/3` | `:367` / `:384` / `:401` / `:418` |
| rotorVelocitySlowdownSim | `10` (all) | `:368` / `:385` / `:402` / `:419` |

Raw XML (rotor_0 block, verbatim):
```xml
    <plugin name='front_right_motor_model' filename='libgazebo_motor_model.so'>
      <robotNamespace/>
      <jointName>rotor_0_joint</jointName>
      <linkName>rotor_0</linkName>
      <turningDirection>ccw</turningDirection>
      <timeConstantUp>0.0125</timeConstantUp>
      <timeConstantDown>0.025</timeConstantDown>
      <maxRotVelocity>1100</maxRotVelocity>
      <motorConstant>5.84e-06</motorConstant>
      <momentConstant>0.06</momentConstant>
      <commandSubTopic>/gazebo/command/motor_speed</commandSubTopic>
      <motorNumber>0</motorNumber>
      <rotorDragCoefficient>0.000175</rotorDragCoefficient>
      <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
      <motorSpeedPubTopic>/motor_speed/0</motorSpeedPubTopic>
      <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
    </plugin>
```

Base multirotor plugin (not a motor): `<plugin name='rosbag' filename='libgazebo_multirotor_base_plugin.so'>` at `iris.sdf.jinja:348-352`, with `<rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>` at `:351`.

### 1d. Body drag

| Name | Value | Source |
|---|---|---|
| base_link `velocity_decay` | `<velocity_decay/>` (empty element, no parameters) | `iris.sdf.jinja:53` |
| rotor link `velocity_decay` | `<velocity_decay/>` (empty, no parameters) | `iris.sdf.jinja:135`, `:200`, `:265`, `:330` |
| Linear/angular damping, `<drag>` or similar body-drag tags | NOT FOUND | (searched whole file) |
| Aerodynamic drag term | `rotorDragCoefficient` only (see 1c) | `iris.sdf.jinja:365` etc. |

---

## 2. Gazebo (gz) x500 model, for comparison only

Sources: `x500_base_model.sdf` (model `x500_base`) and `x500_model.sdf` (wrapper that includes `x500_base` and adds four motor plugins).

### 2a. Base link and rotor links

| Name | Value | Source |
|---|---|---|
| x500_base model pose | `0 0 .24 0 0 0` | `x500_base_model.sdf:4` |
| base_link mass | `2.0` | `x500_base_model.sdf:9` |
| base_link ixx | `0.02166666666666667` | `x500_base_model.sdf:11` |
| base_link ixy | `0` | `x500_base_model.sdf:12` |
| base_link ixz | `0` | `x500_base_model.sdf:13` |
| base_link iyy | `0.02166666666666667` | `x500_base_model.sdf:14` |
| base_link iyz | `0` | `x500_base_model.sdf:15` |
| base_link izz | `0.04000000000000001` | `x500_base_model.sdf:16` |
| base_link `velocity_decay` | `<velocity_decay/>` (empty) | `x500_base_model.sdf:20` |
| rotor_0 link pose | `0.174 -0.174 0.06 0 0 0` | `x500_base_model.sdf:316` |
| rotor_1 link pose | `-0.174 0.174 0.06 0 0 0` | `x500_base_model.sdf:388` |
| rotor_2 link pose | `0.174 0.174 0.06 0 0 0` | `x500_base_model.sdf:460` |
| rotor_3 link pose | `-0.174 -0.174 0.06 0 0 0` | `x500_base_model.sdf:532` |
| rotor_0..3 mass | `0.016076923076923075` (each) | `x500_base_model.sdf:318`, `:390`, `:462`, `:534` |
| rotor_0..3 ixx | `3.8464910483993325e-07` (each) | `x500_base_model.sdf:320`, `:392`, `:464`, `:536` |
| rotor_0..3 iyy | `2.6115851691700804e-05` (each) | `x500_base_model.sdf:321`, `:393`, `:465`, `:537` |
| rotor_0..3 izz | `2.649858234714004e-05` (each) | `x500_base_model.sdf:322`, `:394`, `:466`, `:538` |
| rotor link `velocity_decay` | `<velocity_decay/>` (empty) | `x500_base_model.sdf:315`, `:387`, `:459`, `:531` |
| Body drag / damping tags | NOT FOUND beyond empty `velocity_decay` | (searched whole file) |

### 2b. Motor plugins (`gz-sim-multicopter-motor-model-system`)

Plugin blocks in `x500_model.sdf`: rotor_0 `:7-22`, rotor_1 `:23-38`, rotor_2 `:39-54`, rotor_3 `:55-70`.

| Name | Value (rotor_0 / rotor_1 / rotor_2 / rotor_3) | Source (lines for rotor_0 / rotor_1 / rotor_2 / rotor_3) |
|---|---|---|
| jointName | `rotor_0_joint` / `rotor_1_joint` / `rotor_2_joint` / `rotor_3_joint` | `:8` / `:24` / `:40` / `:56` |
| linkName | `rotor_0` / `rotor_1` / `rotor_2` / `rotor_3` | `:9` / `:25` / `:41` / `:57` |
| turningDirection | `ccw` / `ccw` / `cw` / `cw` | `:10` / `:26` / `:42` / `:58` |
| timeConstantUp | `0.0125` (all) | `:11` / `:27` / `:43` / `:59` |
| timeConstantDown | `0.025` (all) | `:12` / `:28` / `:44` / `:60` |
| maxRotVelocity | `1000.0` (all) | `:13` / `:29` / `:45` / `:61` |
| motorConstant | `8.54858e-06` (all) | `:14` / `:30` / `:46` / `:62` |
| momentConstant | `0.016` (all) | `:15` / `:31` / `:47` / `:63` |
| commandSubTopic | `command/motor_speed` (all) | `:16` / `:32` / `:48` / `:64` |
| motorNumber | `0` / `1` / `2` / `3` | `:17` / `:33` / `:49` / `:65` |
| rotorDragCoefficient | `8.06428e-05` (all) | `:18` / `:34` / `:50` / `:66` |
| rollingMomentCoefficient | `1e-06` (all) | `:19` / `:35` / `:51` / `:67` |
| rotorVelocitySlowdownSim | `10` (all) | `:20` / `:36` / `:52` / `:68` |
| motorType | `velocity` (all) | `:21` / `:37` / `:53` / `:69` |

---

## 3. PX4 SITL airframe config for Iris and sourced files

### 3a. Airframe `ROMFS/px4fmu_common/init.d-posix/airframes/10015_gazebo-classic_iris` (full contents, verbatim)

```sh
#!/bin/sh
#
# @name 3DR Iris Quadrotor SITL
#
# @type Quadrotor Wide
#
# @maintainer Julian Oes <julian@oes.ch>
#

. ${R}etc/init.d/rc.mc_defaults


param set-default CA_AIRFRAME 0

param set-default CA_ROTOR_COUNT 4
param set-default CA_ROTOR0_PX 0.1515
param set-default CA_ROTOR0_PY 0.245
param set-default CA_ROTOR0_KM 0.05
param set-default CA_ROTOR1_PX -0.1515
param set-default CA_ROTOR1_PY -0.1875
param set-default CA_ROTOR1_KM 0.05
param set-default CA_ROTOR2_PX 0.1515
param set-default CA_ROTOR2_PY -0.245
param set-default CA_ROTOR2_KM -0.05
param set-default CA_ROTOR3_PX -0.1515
param set-default CA_ROTOR3_PY 0.1875
param set-default CA_ROTOR3_KM -0.05

param set-default PWM_MAIN_FUNC1 101
param set-default PWM_MAIN_FUNC2 102
param set-default PWM_MAIN_FUNC3 103
param set-default PWM_MAIN_FUNC4 104
```

Line 10 sources `${R}etc/init.d/rc.mc_defaults`. No `10015_gazebo-classic_iris.post` file exists in the checkout.

### 3b. `ROMFS/px4fmu_common/init.d/rc.mc_defaults` (full contents, verbatim)

```sh
#!/bin/sh
#
# Multicopter default parameters.
#
# NOTE: Script variables are declared/initialized/unset in the rcS script.
#

set VEHICLE_TYPE mc

# MAV_TYPE_QUADROTOR 2
param set-default MAV_TYPE 2

if param compare IMU_GYRO_RATEMAX 400
then
	param set-default IMU_GYRO_RATEMAX 800
fi

param set-default NAV_ACC_RAD 2

param set-default RTL_RETURN_ALT 30
param set-default RTL_DESCEND_ALT 10

param set-default GPS_UBX_DYNMODEL 6

# lower RNG_FOG since MC are expected to fly closer over obstacles
param set-default EKF2_RNG_FOG 1.0
```

### 3c. Every `param set` / `param set-default` / `param compare` line that sets MC_/MPC_/CA_/IMU_GYRO/THR_MDL/HTE_ params (file:line, verbatim)

| Name | Value | Source |
|---|---|---|
| CA_AIRFRAME | `param set-default CA_AIRFRAME 0` | `airframes/10015_gazebo-classic_iris:13` |
| CA_ROTOR_COUNT | `param set-default CA_ROTOR_COUNT 4` | `airframes/10015_gazebo-classic_iris:15` |
| CA_ROTOR0_PX | `param set-default CA_ROTOR0_PX 0.1515` | `airframes/10015_gazebo-classic_iris:16` |
| CA_ROTOR0_PY | `param set-default CA_ROTOR0_PY 0.245` | `airframes/10015_gazebo-classic_iris:17` |
| CA_ROTOR0_KM | `param set-default CA_ROTOR0_KM 0.05` | `airframes/10015_gazebo-classic_iris:18` |
| CA_ROTOR1_PX | `param set-default CA_ROTOR1_PX -0.1515` | `airframes/10015_gazebo-classic_iris:19` |
| CA_ROTOR1_PY | `param set-default CA_ROTOR1_PY -0.1875` | `airframes/10015_gazebo-classic_iris:20` |
| CA_ROTOR1_KM | `param set-default CA_ROTOR1_KM 0.05` | `airframes/10015_gazebo-classic_iris:21` |
| CA_ROTOR2_PX | `param set-default CA_ROTOR2_PX 0.1515` | `airframes/10015_gazebo-classic_iris:22` |
| CA_ROTOR2_PY | `param set-default CA_ROTOR2_PY -0.245` | `airframes/10015_gazebo-classic_iris:23` |
| CA_ROTOR2_KM | `param set-default CA_ROTOR2_KM -0.05` | `airframes/10015_gazebo-classic_iris:24` |
| CA_ROTOR3_PX | `param set-default CA_ROTOR3_PX -0.1515` | `airframes/10015_gazebo-classic_iris:25` |
| CA_ROTOR3_PY | `param set-default CA_ROTOR3_PY 0.1875` | `airframes/10015_gazebo-classic_iris:26` |
| CA_ROTOR3_KM | `param set-default CA_ROTOR3_KM -0.05` | `airframes/10015_gazebo-classic_iris:27` |
| MAV_TYPE | `param set-default MAV_TYPE 2` | `init.d/rc.mc_defaults:11` |
| IMU_GYRO_RATEMAX (condition) | `if param compare IMU_GYRO_RATEMAX 400` | `init.d/rc.mc_defaults:13` |
| IMU_GYRO_RATEMAX (set) | `param set-default IMU_GYRO_RATEMAX 800` | `init.d/rc.mc_defaults:15` (runs only if the value is still 400) |
| NAV_ACC_RAD | `param set-default NAV_ACC_RAD 2` | `init.d/rc.mc_defaults:18` |
| RTL_RETURN_ALT | `param set-default RTL_RETURN_ALT 30` | `init.d/rc.mc_defaults:20` |
| RTL_DESCEND_ALT | `param set-default RTL_DESCEND_ALT 10` | `init.d/rc.mc_defaults:21` |
| GPS_UBX_DYNMODEL | `param set-default GPS_UBX_DYNMODEL 6` | `init.d/rc.mc_defaults:23` |
| EKF2_RNG_FOG | `param set-default EKF2_RNG_FOG 1.0` | `init.d/rc.mc_defaults:26` |
| MC_AT_EN | `param set-default -s MC_AT_EN 1` | `init.d-posix/rcS:178` |

Other MC/MPC/CA sets found in the posix tree are for non-Iris airframes or VTOL/rover defaults (`init.d/rc.vtol_defaults`, `rc.rover_*_defaults`, `rc.heli_defaults`) and are not applied to 10015. `init.d-posix/rcS` also sources the selected airframe (`rcS:217-234`) and `init.d/rc.vehicle_setup` (`rcS:290`); neither sets MC_/MPC_/CA_ params. Non-MC/CA sets in `rcS` (for example `rcS:163-194`) are not listed.

---

## 4. Default values of PX4 parameters at v1.16.2

Values are copied as written in the source (`PARAM_DEFINE_*` literal, or YAML `default:`). Source paths are relative to the PX4 checkout root.

| Name | Value | Source |
|---|---|---|
| MC_ROLL_P | `6.5f` | `src/modules/mc_att_control/mc_att_control_params.c:53` |
| MC_PITCH_P | `6.5f` | `src/modules/mc_att_control/mc_att_control_params.c:66` |
| MC_YAW_P | `2.8f` | `src/modules/mc_att_control/mc_att_control_params.c:79` |
| MC_YAW_WEIGHT | `0.4f` | `src/modules/mc_att_control/mc_att_control_params.c:97` |
| MC_ROLLRATE_MAX | `220.0f` | `src/modules/mc_att_control/mc_att_control_params.c:116` |
| MC_PITCHRATE_MAX | `220.0f` | `src/modules/mc_att_control/mc_att_control_params.c:135` |
| MC_YAWRATE_MAX | `200.0f` | `src/modules/mc_att_control/mc_att_control_params.c:147` |
| MC_ROLLRATE_P | `0.15f` | `src/modules/mc_rate_control/mc_rate_control_params.c:51` |
| MC_ROLLRATE_I | `0.2f` | `src/modules/mc_rate_control/mc_rate_control_params.c:63` |
| MC_ROLLRATE_D | `0.003f` | `src/modules/mc_rate_control/mc_rate_control_params.c:88` |
| MC_ROLLRATE_FF | `0.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:99` |
| MC_ROLLRATE_K | `1.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:119` |
| MC_RR_INT_LIM | `0.30f` | `src/modules/mc_rate_control/mc_rate_control_params.c:75` |
| MC_PITCHRATE_P | `0.15f` | `src/modules/mc_rate_control/mc_rate_control_params.c:132` |
| MC_PITCHRATE_I | `0.2f` | `src/modules/mc_rate_control/mc_rate_control_params.c:144` |
| MC_PITCHRATE_D | `0.003f` | `src/modules/mc_rate_control/mc_rate_control_params.c:168` |
| MC_PITCHRATE_FF | `0.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:179` |
| MC_PITCHRATE_K | `1.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:199` |
| MC_PR_INT_LIM | `0.30f` | `src/modules/mc_rate_control/mc_rate_control_params.c:156` |
| MC_YAWRATE_P | `0.2f` | `src/modules/mc_rate_control/mc_rate_control_params.c:212` |
| MC_YAWRATE_I | `0.1f` | `src/modules/mc_rate_control/mc_rate_control_params.c:224` |
| MC_YAWRATE_D | `0.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:248` |
| MC_YAWRATE_FF | `0.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:260` |
| MC_YAWRATE_K | `1.0f` | `src/modules/mc_rate_control/mc_rate_control_params.c:280` |
| MC_YR_INT_LIM | `0.30f` | `src/modules/mc_rate_control/mc_rate_control_params.c:236` |
| MC_YAW_TQ_CUTOFF | `2.f` | `src/modules/mc_rate_control/mc_rate_control_params.c:308` |
| MC_AIRMODE | `0` (`PARAM_DEFINE_INT32`) | `src/lib/mixer_module/params.c:18` |
| MPC_XY_P | `0.95f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:58` |
| MPC_Z_P | `1.f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:45` |
| MPC_XY_VEL_P_ACC | `1.8f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:84` |
| MPC_XY_VEL_I_ACC | `0.4f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:111` |
| MPC_XY_VEL_D_ACC | `0.2f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:137` |
| MPC_Z_VEL_P_ACC | `4.f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:71` |
| MPC_Z_VEL_I_ACC | `2.f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:97` |
| MPC_Z_VEL_D_ACC | `0.f` | `src/modules/mc_pos_control/multicopter_position_control_gain_params.c:124` |
| MPC_XY_VEL_MAX | `12.f` | `src/modules/mc_pos_control/multicopter_position_control_limits_params.c:47` |
| MPC_Z_VEL_MAX_UP | `3.f` | `src/modules/mc_pos_control/multicopter_position_control_limits_params.c:64` |
| MPC_Z_VEL_MAX_DN | `1.5f` | `src/modules/mc_pos_control/multicopter_position_control_limits_params.c:81` |
| MPC_TILTMAX_AIR | `45.f` | `src/modules/mc_pos_control/multicopter_position_control_limits_params.c:96` |
| MPC_THR_MIN | `0.12f` | `src/modules/mc_pos_control/multicopter_position_control_limits_params.c:126` |
| MPC_THR_MAX | `1.f` | `src/modules/mc_pos_control/multicopter_position_control_limits_params.c:140` |
| MPC_THR_HOVER | `0.5f` | `src/modules/mc_pos_control/multicopter_position_control_params.c:49` |
| MPC_THR_XY_MARG | `0.3f` | `src/modules/mc_pos_control/multicopter_position_control_params.c:75` |
| MPC_USE_HTE | `1` (`PARAM_DEFINE_INT32`) | `src/modules/mc_pos_control/multicopter_position_control_params.c:60` |
| MPC_VEL_LP | `0.0f` | `src/modules/mc_pos_control/multicopter_position_control_params.c:89` |
| MPC_VELD_LP | `5.0f` | `src/modules/mc_pos_control/multicopter_position_control_params.c:132` |
| MPC_VEL_NF_FRQ | `0.0f` | `src/modules/mc_pos_control/multicopter_position_control_params.c:104` |
| MPC_VEL_NF_BW | `5.0f` | `src/modules/mc_pos_control/multicopter_position_control_params.c:118` |
| HTE_HT_NOISE | `0.0036` | `src/modules/mc_hover_thrust_estimator/hover_thrust_estimator_params.c:55` |
| HTE_HT_ERR_INIT | `0.1` | `src/modules/mc_hover_thrust_estimator/hover_thrust_estimator_params.c:83` |
| HTE_ACC_GATE | `3.0` | `src/modules/mc_hover_thrust_estimator/hover_thrust_estimator_params.c:69` |
| HTE_THR_RANGE | `0.2` | `src/modules/mc_hover_thrust_estimator/hover_thrust_estimator_params.c:132` |
| HTE_VXY_THR | `10.0` | `src/modules/mc_hover_thrust_estimator/hover_thrust_estimator_params.c:99` |
| HTE_VZ_THR | `2.0` | `src/modules/mc_hover_thrust_estimator/hover_thrust_estimator_params.c:115` |
| THR_MDL_FAC | `0.0f` | `src/lib/mixer_module/motor_params.c:57` |
| CA_ROTOR_COUNT | `0` (YAML `default:`; airframe sets `4`) | `src/modules/control_allocator/module.yaml:127` |
| CA_ROTOR${i}_PX | `0.0` (YAML template default) | `src/modules/control_allocator/module.yaml:138` |
| CA_ROTOR${i}_PY | `0.0` (YAML template default) | `src/modules/control_allocator/module.yaml:149` |
| CA_ROTOR${i}_PZ | `0.0` (YAML template default) | `src/modules/control_allocator/module.yaml:160` |
| CA_ROTOR${i}_AX | `0.0` (YAML template default) | `src/modules/control_allocator/module.yaml:172` |
| CA_ROTOR${i}_AY | `0.0` (YAML template default) | `src/modules/control_allocator/module.yaml:183` |
| CA_ROTOR${i}_AZ | `-1.0` (YAML template default) | `src/modules/control_allocator/module.yaml:194` |
| CA_ROTOR${i}_CT | `6.5` (YAML template default) | `src/modules/control_allocator/module.yaml:209` |
| CA_ROTOR${i}_KM | `0.05` (YAML template default) | `src/modules/control_allocator/module.yaml:225` |
| IMU_GYRO_RATEMAX | `400` (`PARAM_DEFINE_INT32`; rc.mc_defaults raises to 800 at line 15 when still 400) | `src/modules/sensors/vehicle_angular_velocity/imu_gyro_parameters.c:150` |
| IMU_GYRO_CUTOFF | `40.0f` | `src/modules/sensors/vehicle_angular_velocity/imu_gyro_parameters.c:128` |
| IMU_DGYRO_CUTOFF | `30.0f` | `src/modules/sensors/vehicle_angular_velocity/imu_gyro_parameters.c:172` |

Notes:
- `CA_ROTOR${i}_*` are generated per instance from `module.yaml` (`num_instances: *max_num_mc_motors`, which is `12`, `module.yaml:1`). Only the template default is listed.
- For `CA_ROTOR${i}_TILT` and others not in the list, see the same YAML block (`module.yaml:226` onward).

---

## 5. Where rotor geometry is set for the gazebo-classic Iris

| Name | Value | Source |
|---|---|---|
| CA_AIRFRAME (multirotor) | `0` | `ROMFS/px4fmu_common/init.d-posix/airframes/10015_gazebo-classic_iris:13` |
| CA_ROTOR_COUNT | `4` | `airframes/10015_gazebo-classic_iris:15` |
| CA_ROTOR0_PX / PY / KM | `0.1515` / `0.245` / `0.05` | `airframes/10015_gazebo-classic_iris:16-18` |
| CA_ROTOR1_PX / PY / KM | `-0.1515` / `-0.1875` / `0.05` | `airframes/10015_gazebo-classic_iris:19-21` |
| CA_ROTOR2_PX / PY / KM | `0.1515` / `-0.245` / `-0.05` | `airframes/10015_gazebo-classic_iris:22-24` |
| CA_ROTOR3_PX / PY / KM | `-0.1515` / `0.1875` / `-0.05` | `airframes/10015_gazebo-classic_iris:25-27` |
| CA_ROTOR*_PZ, AX, AY, AZ, CT | NOT set by the airframe; YAML defaults apply (section 4) | `src/modules/control_allocator/module.yaml:160, :172, :183, :194, :209` |

The airframe sets only PX, PY and KM per rotor. The SDF-side geometry (section 1b) is a separate source and is not used by PX4 to set these values.

### Raw side-by-side (no conversion, no reconciliation)

| Quantity | Gazebo Classic SDF (`iris.sdf.jinja`) | PX4 airframe (`10015`) |
|---|---|---|
| rotor_0 x / y | `0.13` / `-0.22` (`:89`) | `CA_ROTOR0_PX` `0.1515` / `CA_ROTOR0_PY` `0.245` (`:16-17`) |
| rotor_1 x / y | `-0.13` / `0.2` (`:154`) | `CA_ROTOR1_PX` `-0.1515` / `CA_ROTOR1_PY` `-0.1875` (`:19-20`) |
| rotor_2 x / y | `0.13` / `0.22` (`:219`) | `CA_ROTOR2_PX` `0.1515` / `CA_ROTOR2_PY` `-0.245` (`:22-23`) |
| rotor_3 x / y | `-0.13` / `-0.2` (`:284`) | `CA_ROTOR3_PX` `-0.1515` / `CA_ROTOR3_PY` `0.1875` (`:25-26`) |
| Yaw moment coefficient | `momentConstant` `0.06` (`:362`) | `CA_ROTOR0_KM` `0.05` (`:18`) |
| Thrust coefficient | not present in the SDF | `CA_ROTOR*_CT` default `6.5` (`module.yaml:209`), not set by the airframe |

Frame conventions of the two sources were not checked, and no conversion was done.
