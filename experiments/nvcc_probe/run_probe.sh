#!/usr/bin/env bash
# Build and run the nvcc probe against the repository's px4/ sources, in two
# floating-point flag sets. Needs nvcc and a CUDA GPU.
# Usage: experiments/nvcc_probe/run_probe.sh <sm_arch, e.g. 89> [N] [steps]
# Outputs go to experiments/nvcc_probe/out/ (git-ignored).
set -u
ARCH=${1:?sm arch, e.g. 89}
N=${2:-4096}
STEPS=${3:-600}
HERE=$(cd "$(dirname "$0")" && pwd)
PX4=$HERE/../../px4
OUT=$HERE/out
INC="-I$PX4/lib -I$PX4/attitude_control -I$PX4/rate_control -I$PX4/control_allocator -I$PX4/actuator_output"
COMMON=(-std=c++17 -O3 -arch=sm_${ARCH} --expt-relaxed-constexpr
        "-DPX4RL_HD=__host__ __device__" -Xcudafe --display_error_number -Xptxas -v)
mkdir -p "$OUT"
nvcc --version | tail -2 > "$OUT/toolchain.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$OUT/toolchain.txt"

echo "== build strict (device --fmad=false, host -ffp-contract=off)"
nvcc "${COMMON[@]}" --fmad=false -Xcompiler -ffp-contract=off $INC "$HERE/probe.cu" -o "$OUT/probe_strict" \
    > "$OUT/build_strict.txt" 2>&1
echo "exit=$?" >> "$OUT/build_strict.txt"
echo "== build default"
nvcc "${COMMON[@]}" $INC "$HERE/probe.cu" -o "$OUT/probe_default" > "$OUT/build_default.txt" 2>&1
echo "exit=$?" >> "$OUT/build_default.txt"

for v in strict default; do
    if [ -x "$OUT/probe_$v" ]; then
        echo "== run $v"
        "$OUT/probe_$v" "$N" "$STEPS" > "$OUT/run_$v.txt" 2>&1
        echo "exit=$?" >> "$OUT/run_$v.txt"
    fi
done
tail -n 3 "$OUT"/build_*.txt
