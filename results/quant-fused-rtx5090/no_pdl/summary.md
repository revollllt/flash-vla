# Fused producer-quantize, RTX 5090

FlashInfer 0.7.0, torch 2.13.0+cu130. Times in µs per call (isolated) or per producer + its GEMMs (chain); saving per observation in ms.

## groot mxfp8

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 512×1024 | 24 | 2.14 | 1.12 / 1.06 | 2.21 | 1.05 | 15.42 | 14.09 | 1.32 |
| vit attn -> o | 512×1024 | 24 | — | 1.24 / 1.17 | (standalone) | | | | |
| vit ln2 -> up | 512×1024 | 24 | 2.27 | 1.24 / 1.18 | 2.36 | 1.15 | 15.64 | 14.26 | 1.38 |
| vit gelu -> down | 512×4096 | 24 | 2.65 | 1.91 / 1.96 | 2.77 | 1.80 | 20.81 | 18.76 | 2.07 |
| llm rms1 -> qkv | 156×2048 | 16 | 2.20 | 1.14 / 1.14 | 2.27 | 1.08 | 13.75 | 12.36 | 1.38 |
| llm attn -> o | 156×2048 | 16 | — | 1.14 / 1.15 | (standalone) | | | | |
| llm rms2 -> gateup | 156×2048 | 16 | 2.17 | 1.14 / 1.15 | 2.26 | 1.05 | 26.18 | 25.50 | 0.64 |
| llm swiglu -> down | 156×6144 | 16 | 2.39 | 1.49 / 1.88 | 2.58 | 1.31 | 26.75 | 25.22 | 1.52 |
| ref ln1 -> q,k,v | 156×2048 | 4 | 2.17 | 1.14 / 1.16 | 2.26 | 1.05 | 30.30 | 28.99 | 1.30 |
| ref attn -> o | 156×2048 | 4 | — | 1.15 / 1.16 | (standalone) | | | | |
| ref ln3 -> up | 156×2048 | 4 | 2.16 | 1.14 / 1.14 | 2.23 | 1.08 | 23.96 | 22.96 | 0.99 |
| ref gelu -> down | 156×8192 | 4 | 2.01 | 1.65 / 2.23 | 2.23 | 1.44 | 32.77 | 31.16 | 1.60 |
| context -> cross k,v | 156×2048 | 1 | — | 1.15 / 1.17 | (standalone) | | | | |
| dit adaln -> qkv | 41×1536 | 64 | 1.88 | 1.04 / 0.92 | 1.98 | 0.95 | 10.48 | 9.45 | 1.02 |
| dit adaln -> cross q | 41×1536 | 64 | 1.88 | 1.05 / 0.90 | 1.97 | 0.96 | 8.71 | 7.57 | 1.15 |
| dit attn -> o | 41×1536 | 128 | — | 1.05 / 0.91 | (standalone) | | | | |
| dit ln3 -> up | 41×1536 | 128 | 1.59 | 1.05 / 0.89 | 1.68 | 0.95 | 11.52 | 10.36 | 1.16 |
| dit gelu -> down | 41×6144 | 128 | 1.24 | 1.12 / 1.78 | 1.30 | 1.05 | 16.42 | 15.13 | 1.27 |

Saving per observation: isolated 0.544 ms, chain 0.638 ms. Standalone quantize left (attention outputs, context): 0.188 ms.

## groot nvfp4

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 512×1024 | 24 | 2.29 | 1.22 / 1.08 | 2.36 | 1.15 | 10.66 | 9.31 | 1.36 |
| vit attn -> o | 512×1024 | 24 | — | 1.22 / 1.08 | (standalone) | | | | |
| vit ln2 -> up | 512×1024 | 24 | 2.29 | 1.21 / 1.08 | 2.36 | 1.14 | 11.02 | 9.63 | 1.39 |
| vit gelu -> down | 512×4096 | 24 | 2.68 | 1.82 / 1.75 | 2.65 | 1.84 | 20.32 | 18.21 | 2.10 |
| llm rms1 -> qkv | 156×2048 | 16 | 2.20 | 1.11 / 1.05 | 2.23 | 1.06 | 10.45 | 9.23 | 1.22 |
| llm attn -> o | 156×2048 | 16 | — | 1.11 / 1.05 | (standalone) | | | | |
| llm rms2 -> gateup | 156×2048 | 16 | 2.20 | 1.11 / 1.05 | 2.24 | 1.06 | 21.88 | 20.89 | 0.99 |
| llm swiglu -> down | 156×6144 | 16 | 2.39 | 1.46 / 1.43 | 2.58 | 1.27 | 16.63 | 15.15 | 1.48 |
| ref ln1 -> q,k,v | 156×2048 | 4 | 2.18 | 1.11 / 1.05 | 2.23 | 1.05 | 20.06 | 18.89 | 1.16 |
| ref attn -> o | 156×2048 | 4 | — | 1.11 / 1.05 | (standalone) | | | | |
| ref ln3 -> up | 156×2048 | 4 | 2.20 | 1.12 / 1.05 | 2.23 | 1.08 | 14.19 | 12.88 | 1.30 |
| ref gelu -> down | 156×8192 | 4 | 2.01 | 1.59 / 1.53 | 2.20 | 1.40 | 19.48 | 17.90 | 1.58 |
| context -> cross k,v | 156×2048 | 1 | — | 1.11 / 1.05 | (standalone) | | | | |
| dit adaln -> qkv | 41×1536 | 64 | 1.88 | 1.01 / 0.90 | 1.95 | 0.93 | 8.93 | 7.77 | 1.15 |
| dit adaln -> cross q | 41×1536 | 64 | 1.88 | 1.02 / 0.89 | 1.94 | 0.95 | 7.35 | 6.32 | 1.03 |
| dit attn -> o | 41×1536 | 128 | — | 1.02 / 0.90 | (standalone) | | | | |
| dit ln3 -> up | 41×1536 | 128 | 1.59 | 1.01 / 0.90 | 1.66 | 0.95 | 9.40 | 8.05 | 1.35 |
| dit gelu -> down | 41×6144 | 128 | 1.24 | 1.08 / 1.11 | 1.27 | 1.05 | 13.65 | 12.43 | 1.23 |

Saving per observation: isolated 0.543 ms, chain 0.660 ms. Standalone quantize left (attention outputs, context): 0.182 ms.

## pi05 mxfp8

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 768×1152 | 27 | 2.62 | 1.59 / 1.50 | 2.81 | 1.40 | 18.98 | 17.45 | 1.53 |
| vit attn -> o | 768×1152 | 27 | — | 1.59 / 1.49 | (standalone) | | | | |
| vit ln2 -> up | 768×1152 | 27 | 2.52 | 1.46 / 1.40 | 2.71 | 1.27 | 23.26 | 21.74 | 1.51 |
| vit gelu -> down | 768×4352 | 27 | 3.49 | 2.71 / 2.97 | 3.96 | 2.22 | 25.31 | 23.02 | 2.27 |
| llm rms1 -> qkv | 968×2048 | 18 | 3.23 | 1.88 / 1.94 | 3.22 | 1.88 | 28.80 | 26.97 | 1.90 |
| llm attn -> o | 968×2048 | 18 | — | 1.89 / 1.95 | (standalone) | | | | |
| llm rms2 -> gateup | 968×2048 | 18 | 3.26 | 1.88 / 1.94 | 3.26 | 1.89 | 305.08 | 299.01 | 5.47 |
| llm geglu -> down | 968×16384 | 18 | 17.24 | 8.84 / 8.55 | 17.57 | 8.42 | 210.66 | 196.85 | 13.83 |
| exp adarms1 -> qkv | 50×1024 | 180 | 1.77 | 1.05 / 0.92 | 1.85 | 0.97 | 8.04 | 6.85 | 1.19 |
| exp attn -> o | 50×2048 | 180 | — | 1.05 / 1.08 | (standalone) | | | | |
| exp adarms2 -> gateup | 50×1024 | 180 | 1.76 | 1.05 / 0.92 | 1.85 | 0.95 | 10.91 | 9.80 | 1.11 |
| exp geglu -> down | 50×4096 | 180 | 1.43 | 1.11 / 1.46 | 1.53 | 1.01 | 12.71 | 11.48 | 1.24 |

Saving per observation: isolated 0.880 ms, chain 1.163 ms. Standalone quantize left (attention outputs, context): 0.266 ms.

## pi05 nvfp4

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 768×1152 | 27 | 2.65 | 1.52 / 1.24 | 2.84 | 1.33 | 12.99 | 11.29 | 1.70 |
| vit attn -> o | 768×1152 | 27 | — | 1.40 / 1.24 | (standalone) | | | | |
| vit ln2 -> up | 768×1152 | 27 | 2.65 | 1.51 / 1.24 | 2.82 | 1.33 | 16.99 | 15.32 | 1.68 |
| vit gelu -> down | 768×4352 | 27 | 3.54 | 2.63 / 2.33 | 3.93 | 2.23 | 22.70 | 20.14 | 2.56 |
| llm rms1 -> qkv | 968×2048 | 18 | 3.25 | 1.75 / 1.68 | 3.19 | 1.82 | 16.67 | 14.50 | 2.16 |
| llm attn -> o | 968×2048 | 18 | — | 1.75 / 1.67 | (standalone) | | | | |
| llm rms2 -> gateup | 968×2048 | 18 | 3.80 | 2.02 / 1.93 | 3.75 | 2.09 | 135.03 | 132.65 | 2.38 |
| llm geglu -> down | 968×16384 | 18 | 17.21 | 8.87 / 7.62 | 17.47 | 8.61 | 111.26 | 80.13 | 31.19 |
| exp adarms1 -> qkv | 50×1024 | 180 | 1.75 | 1.01 / 0.83 | 1.85 | 0.92 | 7.53 | 6.40 | 1.13 |
| exp attn -> o | 50×2048 | 180 | — | 1.02 / 0.95 | (standalone) | | | | |
| exp adarms2 -> gateup | 50×1024 | 180 | 1.75 | 1.01 / 0.82 | 1.85 | 0.93 | 9.05 | 7.87 | 1.19 |
| exp geglu -> down | 50×4096 | 180 | 1.43 | 1.08 / 1.02 | 1.48 | 1.03 | 10.96 | 9.71 | 1.26 |

Saving per observation: isolated 0.878 ms, chain 1.446 ms. Standalone quantize left (attention outputs, context): 0.253 ms.

