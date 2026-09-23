#!/usr/bin/env bash
# Refresh the vendored, read-only quantization kernel references.
#
# Each directory beside this script is a copy of a shallow, blob-filtered
# sparse checkout of one upstream: the paths that hold its FP8/NVFP4 GEMM,
# quantize, scale-layout and quantization-config code, plus its LICENSE, at the
# commit its release tag pointed to when pinned. The copies are committed; run
# this only to move a pin, then review the diff. It refuses a tag that has
# moved. Nothing here is on an include path or a build dependency.
#
#   bash third_party/quant-references/update.sh            # all
#   bash third_party/quant-references/update.sh flashinfer # one
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# name | url | tag | commit | sparse patterns (gitignore syntax, anchored at the
# repository root; a directory pattern includes its whole subtree, `!` removes).
# Scope is FP8 and NVFP4 on the native low-bit tensor cores, so weight-only
# kernels (Marlin, Machete, GPTQ, AWQ, AllSpark) and SM90-only W4A8 are left out,
# and so are Triton tuning tables (JSON tuned for other GPUs and shapes, which
# the repository's *.json rule would not commit anyway).
read -r -d '' REPOS <<'SPEC' || true
vllm|https://github.com/vllm-project/vllm.git|v0.30.0|ced6857afa0ea7b2e3f0846a62e1394e90f15607|/csrc/libtorch_stable/quantization/ !/csrc/libtorch_stable/quantization/marlin/ !/csrc/libtorch_stable/quantization/machete/ !/csrc/libtorch_stable/quantization/gptq/ !/csrc/libtorch_stable/quantization/gptq_allspark/ !/csrc/libtorch_stable/quantization/awq/ !/csrc/libtorch_stable/quantization/cutlass_w4a8/ /csrc/quantization/ /csrc/cutlass_extensions/ /csrc/core/ /vllm/model_executor/layers/quantization/ !/vllm/model_executor/layers/quantization/utils/marlin_utils* !/vllm/model_executor/layers/quantization/utils/configs/ /LICENSE*
sglang|https://github.com/sgl-project/sglang.git|v0.5.20|94602c9c2b7cbdb8efd5c52802dac6a1c180089e|/python/sglang/kernels/jit/csrc/gemm/ !/python/sglang/kernels/jit/csrc/gemm/marlin/ !/python/sglang/kernels/jit/csrc/gemm/marlin_moe/ !/python/sglang/kernels/jit/csrc/gemm/awq_dequantize.cuh /python/sglang/kernels/jit/include/ /python/sglang/kernels/kda_kernels/ /python/sglang/kernels/aot/csrc/gemm/ /python/sglang/kernels/aot/csrc/cutlass_extensions/ /python/sglang/kernels/ops/gemm/ !/python/sglang/kernels/ops/gemm/csgmv_configs/ /python/sglang/kernels/ops/diffusion/ /python/sglang/srt/layers/quantization/ /LICENSE*
flashinfer|https://github.com/flashinfer-ai/flashinfer.git|v0.7.0|4d75a33f19aaf48b44d5b1c5dbca33bc1eca5c58|/csrc/cute_sm12x_gemm/ /csrc/nv_internal/ /csrc/nvfp4_attention_sm120/ /csrc/*gemm*.cu /csrc/*quant*.cu /csrc/*fp4*.cu /csrc/*fp8*.cu /csrc/*sm120*.cu /include/flashinfer/gemm/ /include/flashinfer/attention/sm120/ /flashinfer/gemm/ /flashinfer/quantization/ /flashinfer/jit/gemm/ /flashinfer/cute_dsl/rmsnorm_fp4quant.py /flashinfer/cute_dsl/add_rmsnorm_fp4quant.py /LICENSE*
SPEC

refresh_one() {
    local name=$1 url=$2 tag=$3 commit=$4; shift 4
    local src="$work/$name"
    git init -q "$src"
    git -C "$src" remote add origin "$url"
    # Non-cone patterns, written directly: `sparse-checkout set` flags differ
    # across git versions, the pattern file does not.
    git -C "$src" config core.sparseCheckout true
    git -C "$src" config core.sparseCheckoutCone false
    printf '%s\n' "$@" > "$src/.git/info/sparse-checkout"
    git -C "$src" fetch -q --depth 1 --filter=blob:none origin "refs/tags/$tag:refs/tags/$tag"
    local got
    got="$(git -C "$src" rev-parse "refs/tags/$tag^{commit}")"
    if [ "$got" != "$commit" ]; then
        echo "$name: tag $tag now points at $got, pinned $commit; refusing" >&2
        exit 1
    fi
    git -C "$src" -c advice.detachedHead=false checkout -q "$commit"
    rm -rf "${here:?}/$name"
    mkdir -p "$here/$name"
    tar -C "$src" --exclude=.git -cf - . | tar -C "$here/$name" -xf -
    echo "$name: $tag $commit, $(find "$here/$name" -type f | wc -l) files"
}

want=("$@")
while IFS='|' read -r name url tag commit patterns; do
    [ -n "$name" ] || continue
    if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
    # shellcheck disable=SC2086
    refresh_one "$name" "$url" "$tag" "$commit" $patterns
done <<< "$REPOS"
