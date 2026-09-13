#!/usr/bin/env bash
# Compile every kernel that depends on the shared tile library to sm_90a PTX.
#
# There is no H100 here, so an sm90 kernel cannot be run. It CAN be compiled:
# nvcc accepts sm_90a on this box. That makes a refactor of the shared tile
# headers verifiable the only way that matters without the target hardware --
# byte-identical PTX before and after. Any difference is the refactor being
# wrong, not "probably fine".
#
# Usage:
#   lab/sm120/tile_ptx_gate.sh baseline   # capture into artifacts/tile-ptx/baseline
#   lab/sm120/tile_ptx_gate.sh check      # recapture and diff against baseline
set -uo pipefail

MODE="${1:-check}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NVCC="${CUDA_HOME:-/usr}/bin/nvcc"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc)" || { echo "no nvcc; set CUDA_HOME"; exit 2; }

OUT="$ROOT/artifacts/tile-ptx/$MODE"
BASE="$ROOT/artifacts/tile-ptx/baseline"
mkdir -p "$OUT"

# Every .cu that reaches tile/sm90, directly or through a local header.
KERNELS=(
  "hardware/nvidia/h100/lingbot_vla/backends/cuda/kernels/skinny_gemm.cu"
  "hardware/nvidia/h100/lingbot_vla/backends/cuda/kernels/split_attention.cu"
  "hardware/nvidia/h100/gemma_expert/backends/cuda/kernels/attn_taskloop.cu"
  "hardware/nvidia/h100/gemma_expert/backends/cuda/kernels/ffn_taskloop.cu"
  "hardware/nvidia/h100/gemma_backbone/backends/cuda/kernels/enc_attn.cu"
  "hardware/nvidia/h100/gemma_backbone/backends/cuda/kernels/gated_ffn.cu"
  "hardware/nvidia/h100/siglip/backends/cuda/kernels/siglip_attn.cu"
)

fail=0
built=0
for rel in "${KERNELS[@]}"; do
  src="$ROOT/src/flash_vla/$rel"
  name="$(basename "${rel%.cu}")"
  # The kernels include their own directory's headers plus the vendor tile root.
  if ! "$NVCC" -O3 -std=c++17 -arch=sm_90a --ptx --expt-relaxed-constexpr \
        -I"$ROOT/third_party/cutlass/include" \
        -I"$ROOT/src/flash_vla/hardware/nvidia/cuda" \
        -I"$(dirname "$src")" \
        -o "$OUT/$name.ptx" "$src" 2> "$OUT/$name.log"; then
    printf '  %-20s BUILD FAILED (see %s)\n' "$name" "$OUT/$name.log"
    fail=$((fail + 1))
    continue
  fi
  built=$((built + 1))
  if [ "$MODE" = "baseline" ]; then
    printf '  %-20s %8s lines\n' "$name" "$(wc -l < "$OUT/$name.ptx")"
  else
    if [ ! -f "$BASE/$name.ptx" ]; then
      printf '  %-20s NO BASELINE\n' "$name"
      fail=$((fail + 1))
    elif cmp -s "$BASE/$name.ptx" "$OUT/$name.ptx"; then
      printf '  %-20s identical   %8s lines\n' "$name" "$(wc -l < "$OUT/$name.ptx")"
    else
      printf '  %-20s *** PTX DIFFERS ***  %s\n' "$name" \
        "$(diff <(cat "$BASE/$name.ptx") <(cat "$OUT/$name.ptx") | grep -c '^[<>]') changed lines"
      fail=$((fail + 1))
    fi
  fi
done

echo
if [ "$MODE" = "baseline" ]; then
  echo "baseline: $built kernels captured into $OUT"
else
  echo "gate: $built built, $fail problems"
fi
exit $((fail > 0))
