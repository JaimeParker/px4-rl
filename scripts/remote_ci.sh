#!/usr/bin/env bash
# Build and test this working tree inside a remote CUDA container.
#
# The tree (minus reference/, build dirs and .git) is streamed with tar into a
# fresh per-invocation directory in the container, configured with CMake,
# built, and tested with ctest. The directory is removed afterwards unless
# PX4RL_KEEP=1. FetchContent downloads are cached across runs.
#
# Required env:  PX4RL_SSH        ssh alias of the host (e.g. gpu-host)
#                PX4RL_CONTAINER  docker container with nvcc + cmake + g++
# Optional env:  PX4RL_GPU        CUDA_VISIBLE_DEVICES value (default: unset, all GPUs visible)
#                PX4RL_PYTHON     python bin dir prepended to PATH (for pybind / Python tests)
#                PX4RL_KEEP=1     keep the remote directory (path is printed)
#                PX4RL_CTEST_ARGS extra ctest args (e.g. "-R parity")
# Usage: scripts/remote_ci.sh [extra cmake configure args...]
set -euo pipefail
: "${PX4RL_SSH:?set PX4RL_SSH}" "${PX4RL_CONTAINER:?set PX4RL_CONTAINER}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPU_EXPORT=${PX4RL_GPU:+export CUDA_VISIBLE_DEVICES=${PX4RL_GPU} &&}
REMOTE_DIR=/tmp/px4rl_ci_$(date +%s)_$$
DEPS_DIR=/tmp/px4rl_deps
CMAKE_ARGS=$(printf ' %q' "$@")
PY_PATH=${PX4RL_PYTHON:+export PATH=${PX4RL_PYTHON}:\$PATH &&}

echo "[remote_ci] $PX4RL_SSH:$PX4RL_CONTAINER:$REMOTE_DIR (GPU ${PX4RL_GPU:-all})"
tar c --exclude=./reference --exclude=./build --exclude='./build-*' --exclude=./.git \
    --exclude='./experiments/*/out' -C "$ROOT" . |
    ssh "$PX4RL_SSH" "docker exec -i $PX4RL_CONTAINER bash -c 'mkdir -p $REMOTE_DIR && tar x -C $REMOTE_DIR'"

set +e
ssh "$PX4RL_SSH" "docker exec $PX4RL_CONTAINER bash -lc '
  set -e
  $PY_PATH
  cd $REMOTE_DIR
  $GPU_EXPORT true
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR=$DEPS_DIR $CMAKE_ARGS
  cmake --build build -j16
  ctest --test-dir build --output-on-failure ${PX4RL_CTEST_ARGS:-}
'"
status=$?
set -e
if [ "${PX4RL_KEEP:-0}" = "1" ]; then
    echo "[remote_ci] kept $REMOTE_DIR"
else
    ssh "$PX4RL_SSH" "docker exec $PX4RL_CONTAINER rm -rf $REMOTE_DIR" || true
fi
echo "[remote_ci] exit=$status"
exit $status
