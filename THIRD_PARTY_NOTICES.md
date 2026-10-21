# Third-party notices

px4-rl is licensed under the BSD 3-Clause License (see `LICENSE`). It contains
or derives from the following third-party works, which keep their own licenses.

## PX4-Autopilot (BSD 3-Clause)

- Upstream: https://github.com/PX4/PX4-Autopilot, v1.16.x.
- Location: `px4/` (except where `px4/PROVENANCE.md` lists a file as written for px4-rl).
- Copyright (c) PX4 Development Team. The license text is kept in each file header.
- Changes are listed in `px4/PROVENANCE.md`.

## PX4-SITL_gazebo-classic / rotors_simulator (Apache License 2.0)

- Upstream: https://github.com/PX4/PX4-SITL_gazebo-classic at commit `70683dc759cbd3ebccfa78429b564f8b1fb5d149` (the version pinned by PX4 v1.16.2), which derives from ETH Zurich ASL's rotors_simulator.
- Location: `include/px4rl/gazebo_motor_model.hpp`. It is a port of `GazeboMotorModel::UpdateForcesAndMoments`, `FirstOrderFilter` and the mavlink-interface input mapping to batched host/device C++.
- Copyright 2015 Fadri Furrer, Michael Burri, Mina Kamel, Janosch Nikolic, Markus Achtelik (ASL, ETH Zurich); Copyright 2015-2021 PX4 Development Team (`gazebo_mavlink_interface.cpp`).
- Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0. The file header states that it was modified from the original.

Iris parameter values come from PX4's `iris.sdf.jinja` and airframe files at the same versions. Sources are tabulated in `docs/iris-params-sources.md`.

## GoogleTest (BSD 3-Clause)

GoogleTest is fetched at build time for the test suite only (FetchContent, v1.14.0). It is not redistributed.
