#!/usr/bin/env bash
# Build the Python package from this working tree inside a remote container and run the Python tests.
#
# The tree (minus reference/, build dirs and .git) is streamed with tar into a fresh per-invocation directory in
# the container. There a throwaway virtual environment is created from PX4RL_PYTHON, the build dependencies
# (pybind11, scikit-build-core) are installed into it, the package is built with `pip install .`, and
# `pytest tests/python` runs against the installed package. The given interpreter's environment is never
# modified, and the whole directory, venv included, is removed afterwards unless PX4RL_KEEP=1.
#
# The throwaway venv is created with --system-site-packages. If PX4RL_PYTHON belongs to a virtual environment
# itself, that flag exposes the base interpreter's packages, not that environment's, so the environment's
# site-packages (torch, numpy, pytest, ...) are added to the new venv through a .pth file.
#
# Required env:  PX4RL_SSH        ssh alias of the host (e.g. gpu-host)
#                PX4RL_CONTAINER  docker container with cmake, g++ (and nvcc for CUDA)
#                PX4RL_PYTHON     path of the Python interpreter in the container
# Optional env:  PX4RL_GPU          CUDA_VISIBLE_DEVICES value (default: unset, all GPUs visible)
#                PX4RL_BUILD_CUDA=1 build libpx4rl_cuda.so and require the CUDA tests to run, not to skip
#                PX4RL_CMAKE_DEFINES extra "-C cmake.define.NAME=VALUE" style pip arguments, e.g.
#                                    "-C cmake.define.PX4RL_STRICT_FP=ON"
#                PX4RL_PIP_ARGS    extra pip arguments for installing the build dependencies (e.g. an index URL)
#                PX4RL_PYTEST_ARGS extra pytest args (e.g. "-k fleet")
#                PX4RL_KEEP=1      keep the remote directory (path is printed)
# Usage: scripts/remote_pytest.sh
set -euo pipefail
: "${PX4RL_SSH:?set PX4RL_SSH}" "${PX4RL_CONTAINER:?set PX4RL_CONTAINER}" "${PX4RL_PYTHON:?set PX4RL_PYTHON}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_DIR=/tmp/px4rl_py_$(date +%s)_$$
BUILD_CUDA=${PX4RL_BUILD_CUDA:-0}

echo "[remote_pytest] $PX4RL_SSH:$PX4RL_CONTAINER:$REMOTE_DIR (python $PX4RL_PYTHON, CUDA $BUILD_CUDA, GPU ${PX4RL_GPU:-all})"
tar c --exclude=./reference --exclude=./build --exclude='./build-*' --exclude=./.git \
    --exclude='./experiments/*/out' --exclude=__pycache__ --exclude='*.egg-info' -C "$ROOT" . |
    ssh "$PX4RL_SSH" "docker exec -i $PX4RL_CONTAINER bash -c 'mkdir -p $REMOTE_DIR && tar x -C $REMOTE_DIR'"

CUDA_PIP_ARGS=""
CUDA_EXPORTS=""
if [ "$BUILD_CUDA" = "1" ]; then
    CUDA_PIP_ARGS="-C cmake.define.PX4RL_BUILD_CUDA=ON"
    CUDA_EXPORTS="export PX4RL_REQUIRE_CUDA=1"
fi

set +e
ssh "$PX4RL_SSH" "docker exec -i $PX4RL_CONTAINER bash -l -s" <<REMOTE
set -euo pipefail
cd $REMOTE_DIR
${PX4RL_GPU:+export CUDA_VISIBLE_DEVICES=${PX4RL_GPU}}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export CMAKE_BUILD_PARALLEL_LEVEL=16
$CUDA_EXPORTS

$(printf '%q' "$PX4RL_PYTHON") -m venv --system-site-packages $REMOTE_DIR/venv
VPY=$REMOTE_DIR/venv/bin/python
BASE_SITE=\$($(printf '%q' "$PX4RL_PYTHON") -c 'import site; print(site.getsitepackages()[0])')
VENV_SITE=\$(\$VPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "import site; site.addsitedir('\$BASE_SITE')" > "\$VENV_SITE/px4rl_base_environment.pth"

\$VPY -m pip install --quiet pybind11 scikit-build-core numpy pytest ${PX4RL_PIP_ARGS:-}
\$VPY -m pip install -v --no-build-isolation --no-deps $CUDA_PIP_ARGS ${PX4RL_CMAKE_DEFINES:-} .

\$VPY -c 'import px4rl; print("px4rl", px4rl.__version__, "from", px4rl.__file__)'
if [ "$BUILD_CUDA" = "1" ]; then
    \$VPY -c 'import torch; assert torch.cuda.is_available(), "torch sees no CUDA device"; import px4rl.cuda; print("torch", torch.__version__, torch.cuda.get_device_name(0))'
fi
\$VPY -m pytest tests/python -v -rs ${PX4RL_PYTEST_ARGS:-}
REMOTE
status=$?
set -e
if [ "${PX4RL_KEEP:-0}" = "1" ]; then
    echo "[remote_pytest] kept $REMOTE_DIR"
else
    ssh "$PX4RL_SSH" "docker exec $PX4RL_CONTAINER rm -rf $REMOTE_DIR" || true
fi
echo "[remote_pytest] exit=$status"
exit $status
