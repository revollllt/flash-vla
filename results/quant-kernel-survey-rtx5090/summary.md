
### mxfp8

| shape | M×K×N | calls | class | floor µs | best µs | ×floor | best backend | quant µs | BF16 cuBLAS µs |
|---|---|---:|---|---:|---:|---:|---|---:|---:|
| groot/vit_qkv | 512×1024×3072 | 24 | small-weight | 3.25 (c) | 8.39 | 2.58 | flashinfer:cudnn+tuned | 1.06 | 25.43 |
| groot/vit_o | 512×1024×1024 | 24 | small-weight | 1.08 (c) | 3.93 | 3.62 | flashinfer:b12x+tuned | 1.17 | 8.22 |
| groot/vit_up | 512×1024×4096 | 24 | mid-M | 4.34 (c) | 9.12 | 2.10 | flashinfer:cudnn+tuned | 1.17 | 25.80 |
| groot/vit_down | 512×4096×1024 | 24 | down (K>N) | 4.34 (c) | 12.93 | 2.98 | flashinfer:b12x+tuned | 1.96 | 28.91 |
| groot/llm_qkv | 156×2048×4096 | 16 | mid-M | 5.41 (m) | 8.81 | 1.63 | flashinfer:b12x+tuned | 1.15 | 25.68 |
| groot/llm_o | 156×2048×2048 | 32 | mid-M | 2.71 (m) | 6.69 | 2.47 | flashinfer:b12x+tuned | 1.15 | 15.87 |
| groot/llm_gateup | 156×2048×12288 | 16 | mid-M | 16.24 (m) | 21.50 | 1.32 | flashinfer:b12x+tuned | 1.15 | 51.84 |
| groot/llm_down | 156×6144×2048 | 16 | down (K>N) | 8.12 (m) | 15.46 | 1.90 | flashinfer:b12x+tuned | 1.86 | 27.76 |
| groot/ref_up | 156×2048×8192 | 4 | mid-M | 10.83 (m) | 16.73 | 1.55 | flashinfer:cutlass+tuned | 1.15 | 45.30 |
| groot/ref_down | 156×8192×2048 | 4 | down (K>N) | 10.83 (m) | 19.99 | 1.85 | flashinfer:b12x+tuned | 2.21 | 36.37 |
| groot/dit_crosskv | 156×2048×3072 | 16 | mid-M | 4.06 (m) | 8.45 | 2.08 | flashinfer:b12x+tuned | 1.16 | 15.17 |
| groot/dit_qkv | 41×1536×4608 | 64 | small-M | 4.57 (m) | 6.07 | 1.33 | flashinfer:b12x | 0.87 | 14.14 |
| groot/dit_o | 41×1536×1536 | 192 | small-weight | 1.52 (m) | 4.34 | 2.85 | flashinfer:b12x+tuned | 0.90 | 7.32 |
| groot/dit_up | 41×1536×6144 | 128 | small-M | 6.09 (m) | 7.73 | 1.27 | flashinfer:b12x+tuned | 0.90 | 15.93 |
| groot/dit_down | 41×6144×1536 | 128 | down (K>N) | 6.09 (m) | 9.45 | 1.55 | flashinfer:b12x+splitK4 | 1.80 | 16.52 |
| pi05/vit_qkv | 768×1152×3456 | 27 | mid-M | 6.18 (c) | 11.52 | 1.86 | flashinfer:cudnn+tuned | 1.48 | 29.64 |
| pi05/vit_o | 768×1152×1152 | 27 | small-weight | 2.06 (c) | 5.43 | 2.64 | flashinfer:b12x | 1.51 | 15.85 |
| pi05/vit_up | 768×1152×4352 | 27 | mid-M | 7.78 (c) | 18.26 | 2.35 | flashinfer:b12x | 1.50 | 55.30 |
| pi05/vit_down | 768×4352×1152 | 27 | down (K>N) | 7.78 (c) | 17.52 | 2.25 | flashinfer:b12x | 2.95 | 43.14 |
| pi05/llm_qkv | 968×2048×2560 | 18 | large-M | 10.25 (c) | 18.57 | 1.81 | flashinfer:cudnn+tuned | 1.93 | 49.98 |
| pi05/llm_o | 968×2048×2048 | 18 | large-M | 8.20 (c) | 15.89 | 1.94 | flashinfer:cudnn+tuned | 1.93 | 49.44 |
| pi05/llm_gateup | 968×2048×32768 | 18 | large-M | 131.22 (c) | 238.92 | 1.82 | flashinfer:cudnn | 2.11 | 654.35 |
| pi05/llm_down | 968×16384×2048 | 18 | down (K>N) | 65.61 (c) | 118.91 | 1.81 | flashinfer:cudnn+tuned | 8.26 | 341.42 |
| pi05/exp_qkv | 50×1024×2560 | 180 | small-weight | 1.69 (m) | 3.64 | 2.15 | flashinfer:b12x+tuned | 0.93 | 7.10 |
| pi05/exp_o | 50×2048×1024 | 180 | small-weight | 1.35 (m) | 5.00 | 3.69 | flashinfer:b12x+splitK4 | 1.08 | 7.03 |
| pi05/exp_gateup | 50×1024×8192 | 180 | small-M | 5.41 (m) | 6.97 | 1.29 | flashinfer:b12x | 0.93 | 12.47 |
| pi05/exp_down | 50×4096×1024 | 180 | down (K>N) | 2.71 (m) | 6.23 | 2.30 | flashinfer:b12x+splitK4 | 1.44 | 9.45 |

### nvfp4

| shape | M×K×N | calls | class | floor µs | best µs | ×floor | best backend | quant µs | BF16 cuBLAS µs |
|---|---|---:|---|---:|---:|---:|---|---:|---:|
| groot/vit_qkv | 512×1024×3072 | 24 | small-weight | 1.67 (c) | 5.88 | 3.52 | flashinfer:b12x | 0.92 | 25.43 |
| groot/vit_o | 512×1024×1024 | 24 | small-weight | 0.56 (c) | 2.73 | 4.90 | flashinfer:b12x+tuned | 1.09 | 8.22 |
| groot/vit_up | 512×1024×4096 | 24 | mid-M | 2.23 (c) | 6.22 | 2.79 | flashinfer:b12x | 1.06 | 25.80 |
| groot/vit_down | 512×4096×1024 | 24 | down (K>N) | 2.23 (c) | 7.92 | 3.55 | flashinfer:b12x+tuned | 1.66 | 28.91 |
| groot/llm_qkv | 156×2048×4096 | 16 | mid-M | 2.95 (m) | 5.53 | 1.87 | flashinfer:b12x+tuned | 0.82 | 25.68 |
| groot/llm_o | 156×2048×2048 | 32 | mid-M | 1.48 (m) | 4.86 | 3.29 | flashinfer:b12x+tuned | 0.85 | 15.87 |
| groot/llm_gateup | 156×2048×12288 | 16 | mid-M | 8.86 (m) | 13.09 | 1.48 | flashinfer:b12x+tuned | 0.84 | 51.84 |
| groot/llm_down | 156×6144×2048 | 16 | down (K>N) | 4.43 (m) | 10.38 | 2.34 | flashinfer:b12x+tuned | 1.25 | 27.76 |
| groot/ref_up | 156×2048×8192 | 4 | mid-M | 5.91 (m) | 9.38 | 1.59 | flashinfer:b12x+tuned | 0.83 | 45.30 |
| groot/ref_down | 156×8192×2048 | 4 | down (K>N) | 5.91 (m) | 14.18 | 2.40 | flashinfer:b12x+tuned | 1.47 | 36.37 |
| groot/dit_crosskv | 156×2048×3072 | 16 | mid-M | 2.21 (m) | 5.51 | 2.49 | flashinfer:b12x | 0.83 | 15.17 |
| groot/dit_qkv | 41×1536×4608 | 64 | small-M | 2.49 (m) | 4.32 | 1.73 | flashinfer:b12x+tuned | 0.89 | 14.14 |
| groot/dit_o | 41×1536×1536 | 192 | small-weight | 0.83 (m) | 2.35 | 2.83 | flashinfer:b12x | 0.89 | 7.32 |
| groot/dit_up | 41×1536×6144 | 128 | small-M | 3.32 (m) | 5.38 | 1.62 | flashinfer:b12x | 0.89 | 15.93 |
| groot/dit_down | 41×6144×1536 | 128 | down (K>N) | 3.32 (m) | 6.83 | 2.06 | flashinfer:b12x+splitK4 | 0.98 | 16.52 |
| pi05/vit_qkv | 768×1152×3456 | 27 | mid-M | 3.17 (c) | 7.09 | 2.23 | flashinfer:b12x | 1.22 | 29.64 |
| pi05/vit_o | 768×1152×1152 | 27 | small-weight | 1.06 (c) | 3.77 | 3.56 | flashinfer:b12x+tuned | 1.24 | 15.85 |
| pi05/vit_up | 768×1152×4352 | 27 | mid-M | 4.00 (c) | 10.30 | 2.58 | flashinfer:cudnn+tuned | 1.22 | 55.30 |
| pi05/vit_down | 768×4352×1152 | 27 | down (K>N) | 4.00 (c) | 10.11 | 2.53 | flashinfer:b12x+tuned | 2.33 | 43.14 |
| pi05/llm_qkv | 968×2048×2560 | 18 | large-M | 5.27 (c) | 10.16 | 1.93 | flashinfer:b12x | 1.66 | 49.98 |
| pi05/llm_o | 968×2048×2048 | 18 | large-M | 4.22 (c) | 9.31 | 2.21 | flashinfer:b12x+tuned | 1.65 | 49.44 |
| pi05/llm_gateup | 968×2048×32768 | 18 | large-M | 67.45 (c) | 121.85 | 1.81 | flashinfer:cutlass+tuned | 1.81 | 654.35 |
| pi05/llm_down | 968×16384×2048 | 18 | down (K>N) | 33.72 (c) | 56.75 | 1.68 | flashinfer:b12x | 7.16 | 341.42 |
| pi05/exp_qkv | 50×1024×2560 | 180 | small-weight | 0.92 (m) | 2.72 | 2.95 | flashinfer:b12x+tuned | 0.80 | 7.10 |
| pi05/exp_o | 50×2048×1024 | 180 | small-weight | 0.74 (m) | 2.78 | 3.77 | flashinfer:b12x+tuned | 0.89 | 7.03 |
| pi05/exp_gateup | 50×1024×8192 | 180 | small-M | 2.95 (m) | 4.73 | 1.60 | flashinfer:b12x+tuned | 0.79 | 12.47 |
| pi05/exp_down | 50×4096×1024 | 180 | down (K>N) | 1.48 (m) | 5.23 | 3.54 | flashinfer:b12x+splitK4 | 0.92 | 9.45 |

### fp8_1d2d

| shape | M×K×N | calls | class | floor µs | best µs | ×floor | best backend | quant µs | BF16 cuBLAS µs |
|---|---|---:|---|---:|---:|---:|---|---:|---:|
| groot/vit_qkv | 512×1024×3072 | 24 | small-weight | 6.39 (c) | 14.66 | 2.29 | vllm:cutlass | 1.72 | 25.43 |
| groot/vit_o | 512×1024×1024 | 24 | small-weight | 2.13 (c) | 14.46 | 6.79 | vllm:cutlass | 1.83 | 8.22 |
| groot/vit_up | 512×1024×4096 | 24 | mid-M | 8.52 (c) | 15.44 | 1.81 | vllm:cutlass | 1.84 | 25.80 |
| groot/vit_down | 512×4096×1024 | 24 | down (K>N) | 8.52 (c) | 49.14 | 5.77 | vllm:cutlass | 3.51 | 28.91 |
| groot/llm_qkv | 156×2048×4096 | 16 | mid-M | 5.25 (m) | 14.98 | 2.85 | vllm:cutlass | 1.66 | 25.68 |
| groot/llm_o | 156×2048×2048 | 32 | mid-M | 2.63 (m) | 14.75 | 5.62 | vllm:cutlass | 1.66 | 15.87 |
| groot/llm_gateup | 156×2048×12288 | 16 | mid-M | 15.75 (m) | 27.27 | 1.73 | vllm:cutlass | 1.66 | 51.84 |
| groot/llm_down | 156×6144×2048 | 16 | down (K>N) | 7.88 (m) | 38.32 | 4.87 | vllm:cutlass | 2.10 | 27.76 |
| groot/ref_up | 156×2048×8192 | 4 | mid-M | 10.50 (m) | 27.00 | 2.57 | vllm:cutlass | 1.65 | 45.30 |
| groot/ref_down | 156×8192×2048 | 4 | down (K>N) | 10.50 (m) | 50.04 | 4.77 | vllm:cutlass | 2.43 | 36.37 |
| groot/dit_crosskv | 156×2048×3072 | 16 | mid-M | 3.94 (m) | 14.82 | 3.76 | vllm:cutlass | 1.66 | 15.17 |
| groot/dit_qkv | 41×1536×4608 | 64 | small-M | 4.43 (m) | 7.61 | 1.72 | vllm:cutlass | 1.47 | 14.14 |
| groot/dit_o | 41×1536×1536 | 192 | small-weight | 1.48 (m) | 7.24 | 4.90 | vllm:cutlass | 1.48 | 7.32 |
| groot/dit_up | 41×1536×6144 | 128 | small-M | 5.91 (m) | 8.59 | 1.45 | vllm:cutlass | 1.47 | 15.93 |
| groot/dit_down | 41×6144×1536 | 128 | down (K>N) | 5.91 (m) | 20.54 | 3.48 | vllm:cutlass | 1.63 | 16.52 |
| pi05/vit_qkv | 768×1152×3456 | 27 | mid-M | 12.12 (c) | 17.42 | 1.44 | vllm:cutlass | 2.11 | 29.64 |
| pi05/vit_o | 768×1152×1152 | 27 | small-weight | 4.04 (c) | 15.93 | 3.94 | vllm:cutlass | 2.10 | 15.85 |
| pi05/vit_up | 768×1152×4352 | 27 | mid-M | 15.27 (c) | 30.79 | 2.02 | vllm:cutlass | 2.10 | 55.30 |
| pi05/vit_down | 768×4352×1152 | 27 | down (K>N) | 15.27 (c) | 52.07 | 3.41 | vllm:cutlass | 4.73 | 43.14 |
| pi05/llm_qkv | 968×2048×2560 | 18 | large-M | 20.12 (c) | 27.52 | 1.37 | vllm:cutlass | 3.35 | 49.98 |
| pi05/llm_o | 968×2048×2048 | 18 | large-M | 16.10 (c) | 26.98 | 1.68 | vllm:cutlass | 3.32 | 49.44 |
| pi05/llm_gateup | 968×2048×32768 | 18 | large-M | 257.58 (c) | 336.37 | 1.31 | vllm:cutlass | 3.36 | 654.35 |
| pi05/llm_down | 968×16384×2048 | 18 | down (K>N) | 128.79 (c) | 190.51 | 1.48 | vllm:cutlass | 18.75 | 341.42 |
| pi05/exp_qkv | 50×1024×2560 | 180 | small-weight | 1.64 (m) | 5.75 | 3.50 | vllm:cutlass | 1.59 | 7.10 |
| pi05/exp_o | 50×2048×1024 | 180 | small-weight | 1.31 (m) | 8.54 | 6.50 | vllm:cutlass | 1.60 | 7.03 |
| pi05/exp_gateup | 50×1024×8192 | 180 | small-M | 5.25 (m) | 7.77 | 1.48 | vllm:cutlass | 1.60 | 12.47 |
| pi05/exp_down | 50×4096×1024 | 180 | down (K>N) | 2.63 (m) | 14.41 | 5.49 | vllm:cutlass | 1.62 | 9.45 |

### Per observation (sum of count × time, ms)

| model | format | floor | best library | ×floor | + unfused quantize | BF16 cuBLAS | best vs BF16 |
|---|---|---:|---:|---:|---:|---:|---:|
| groot | mxfp8 | 3.17 | 5.47 | 1.73 | 0.84 | 11.35 | 2.07× |
| groot | nvfp4 | 1.72 | 3.64 | 2.12 | 0.68 | 11.35 | 3.12× |
| groot | fp8_1d2d | 3.39 | 10.16 | 3.00 | 1.17 | 11.35 | 1.12× |

| pi05 | mxfp8 | 6.53 | 12.42 | 1.90 | 1.25 | 30.09 | 2.42× |
| pi05 | nvfp4 | 3.42 | 7.19 | 2.10 | 1.00 | 30.09 | 4.18× |
| pi05 | fp8_1d2d | 10.82 | 20.17 | 1.86 | 1.97 | 30.09 | 1.49× |


### Gap by class (count-weighted best / floor)

| class | mxfp8 | nvfp4 |
|---|---:|---:|
| small-M | 1.29 | 1.63 |
| down (K>N) | 1.90 | 2.27 |
| small-weight | 2.83 | 3.24 |
| mid-M | 1.88 | 2.21 |
| large-M | 1.83 | 1.84 |
