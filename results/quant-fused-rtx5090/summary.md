# Fused producer-quantize, RTX 5090

FlashInfer 0.7.0, torch 2.13.0+cu130. Times in µs per call (isolated) or per producer + its GEMMs (chain); saving per observation in ms.

## groot mxfp8

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 512×1024 | 24 | 1.56 | 1.05 / 1.05 | 1.46 | 1.15 | 12.91 | 12.19 | 0.72 |
| vit attn -> o | 512×1024 | 24 | — | 1.14 / 1.17 | (standalone) | | | | |
| vit ln2 -> up | 512×1024 | 24 | 1.66 | 1.14 / 1.16 | 1.56 | 1.23 | 13.40 | 13.14 | 0.25 |
| vit gelu -> down | 512×4096 | 24 | 2.52 | 1.81 / 1.96 | 2.61 | 1.72 | 19.85 | 17.76 | 2.08 |
| llm rms1 -> qkv | 156×2048 | 16 | 1.37 | 0.98 / 1.14 | 1.37 | 0.97 | 11.78 | 12.14 | -0.37 |
| llm attn -> o | 156×2048 | 16 | — | 0.98 / 1.15 | (standalone) | | | | |
| llm rms2 -> gateup | 156×2048 | 16 | 1.37 | 0.97 / 1.15 | 1.37 | 0.97 | 24.64 | 25.11 | -0.42 |
| llm swiglu -> down | 156×6144 | 16 | 2.01 | 1.23 / 1.88 | 2.26 | 0.97 | 24.53 | 23.34 | 1.19 |
| ref ln1 -> q,k,v | 156×2048 | 4 | 1.40 | 0.95 / 1.16 | 1.36 | 0.99 | 28.35 | 27.36 | 1.00 |
| ref attn -> o | 156×2048 | 4 | — | 0.95 / 1.14 | (standalone) | | | | |
| ref ln3 -> up | 156×2048 | 4 | 1.39 | 0.95 / 1.14 | 1.34 | 1.00 | 22.30 | 21.45 | 0.89 |
| ref gelu -> down | 156×8192 | 4 | 1.91 | 1.37 / 2.23 | 2.33 | 0.95 | 30.53 | 29.25 | 1.26 |
| context -> cross k,v | 156×2048 | 1 | — | 0.96 / 1.14 | (standalone) | | | | |
| dit adaln -> qkv | 41×1536 | 64 | 1.48 | 0.73 / 0.91 | 1.56 | 0.65 | 8.80 | 7.82 | 0.98 |
| dit adaln -> cross q | 41×1536 | 64 | 1.46 | 0.73 / 0.90 | 1.56 | 0.63 | 7.03 | 6.06 | 0.97 |
| dit attn -> o | 41×1536 | 128 | — | 0.73 / 0.92 | (standalone) | | | | |
| dit ln3 -> up | 41×1536 | 128 | 1.17 | 0.73 / 0.90 | 1.27 | 0.63 | 9.74 | 8.89 | 0.84 |
| dit gelu -> down | 41×6144 | 128 | 0.97 | 0.88 / 1.78 | 0.99 | 0.87 | 14.81 | 13.80 | 1.01 |

Saving per observation: isolated 0.429 ms, chain 0.454 ms. Standalone quantize left (attention outputs, context): 0.141 ms.

## groot nvfp4

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 512×1024 | 24 | 1.69 | 0.97 / 1.08 | 1.56 | 1.08 | 8.59 | 7.47 | 1.12 |
| vit attn -> o | 512×1024 | 24 | — | 0.97 / 1.08 | (standalone) | | | | |
| vit ln2 -> up | 512×1024 | 24 | 1.69 | 0.99 / 1.08 | 1.56 | 1.11 | 8.94 | 7.94 | 1.01 |
| vit gelu -> down | 512×4096 | 24 | 2.52 | 1.78 / 1.75 | 2.49 | 1.81 | 19.23 | 17.16 | 2.07 |
| llm rms1 -> qkv | 156×2048 | 16 | 1.37 | 0.90 / 1.05 | 1.34 | 0.93 | 8.64 | 8.29 | 0.35 |
| llm attn -> o | 156×2048 | 16 | — | 0.90 / 1.05 | (standalone) | | | | |
| llm rms2 -> gateup | 156×2048 | 16 | 1.37 | 0.90 / 1.05 | 1.33 | 0.94 | 20.00 | 18.59 | 1.42 |
| llm swiglu -> down | 156×6144 | 16 | 1.98 | 1.14 / 1.40 | 2.23 | 0.90 | 14.65 | 13.56 | 1.09 |
| ref ln1 -> q,k,v | 156×2048 | 4 | 1.38 | 0.89 / 1.05 | 1.34 | 0.94 | 18.22 | 17.21 | 1.01 |
| ref attn -> o | 156×2048 | 4 | — | 0.89 / 1.05 | (standalone) | | | | |
| ref ln3 -> up | 156×2048 | 4 | 1.40 | 0.92 / 1.05 | 1.34 | 0.98 | 12.08 | 11.28 | 0.81 |
| ref gelu -> down | 156×8192 | 4 | 1.91 | 1.34 / 1.51 | 2.30 | 0.96 | 17.62 | 16.36 | 1.25 |
| context -> cross k,v | 156×2048 | 1 | — | 0.90 / 1.05 | (standalone) | | | | |
| dit adaln -> qkv | 41×1536 | 64 | 1.46 | 0.70 / 0.89 | 1.56 | 0.60 | 7.14 | 6.11 | 1.03 |
| dit adaln -> cross q | 41×1536 | 64 | 1.47 | 0.69 / 0.89 | 1.56 | 0.60 | 5.66 | 4.74 | 0.92 |
| dit attn -> o | 41×1536 | 128 | — | 0.69 / 0.89 | (standalone) | | | | |
| dit ln3 -> up | 41×1536 | 128 | 1.17 | 0.69 / 0.89 | 1.27 | 0.60 | 7.44 | 6.43 | 1.00 |
| dit gelu -> down | 41×6144 | 128 | 0.98 | 0.83 / 1.11 | 0.95 | 0.85 | 12.03 | 10.98 | 1.04 |

Saving per observation: isolated 0.414 ms, chain 0.545 ms. Standalone quantize left (attention outputs, context): 0.131 ms.

## pi05 mxfp8

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 768×1152 | 27 | 1.91 | 1.53 / 1.50 | 2.14 | 1.30 | 17.45 | 15.98 | 1.48 |
| vit attn -> o | 768×1152 | 27 | — | 1.53 / 1.40 | (standalone) | | | | |
| vit ln2 -> up | 768×1152 | 27 | 1.94 | 1.53 / 1.50 | 2.17 | 1.31 | 21.42 | 20.30 | 1.13 |
| vit gelu -> down | 768×4352 | 27 | 3.35 | 2.61 / 2.97 | 3.83 | 2.13 | 23.91 | 21.85 | 2.02 |
| llm rms1 -> qkv | 968×2048 | 18 | 3.03 | 1.91 / 1.94 | 3.07 | 1.88 | 27.77 | 26.07 | 1.65 |
| llm attn -> o | 968×2048 | 18 | — | 1.91 / 1.96 | (standalone) | | | | |
| llm rms2 -> gateup | 968×2048 | 18 | 3.06 | 1.91 / 1.94 | 3.07 | 1.88 | 301.99 | 297.95 | 4.19 |
| llm geglu -> down | 968×16384 | 18 | 16.82 | 8.77 / 8.53 | 17.12 | 8.53 | 210.17 | 196.05 | 14.10 |
| exp adarms1 -> qkv | 50×1024 | 180 | 1.14 | 0.73 / 0.92 | 1.21 | 0.66 | 6.31 | 5.31 | 1.01 |
| exp attn -> o | 50×2048 | 180 | — | 0.79 / 1.08 | (standalone) | | | | |
| exp adarms2 -> gateup | 50×1024 | 180 | 1.14 | 0.73 / 0.92 | 1.21 | 0.66 | 9.25 | 8.53 | 0.72 |
| exp geglu -> down | 50×4096 | 180 | 1.09 | 0.88 / 1.46 | 1.18 | 0.79 | 11.05 | 10.02 | 1.03 |

Saving per observation: isolated 0.729 ms, chain 0.981 ms. Standalone quantize left (attention outputs, context): 0.218 ms.

## pi05 nvfp4

| site | M×K | per obs | bf16 producer | + quantize (ours / FlashInfer) | fused | isolated saving | chain unfused | chain fused | chain saving |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit ln1 -> qkv | 768×1152 | 27 | 1.95 | 1.56 / 1.24 | 2.14 | 1.36 | 10.67 | 9.29 | 1.39 |
| vit attn -> o | 768×1152 | 27 | — | 1.53 / 1.25 | (standalone) | | | | |
| vit ln2 -> up | 768×1152 | 27 | 1.95 | 1.56 / 1.24 | 2.13 | 1.37 | 14.56 | 13.31 | 1.24 |
| vit gelu -> down | 768×4352 | 27 | 3.38 | 2.62 / 2.33 | 3.80 | 2.21 | 21.73 | 19.16 | 2.56 |
| llm rms1 -> qkv | 968×2048 | 18 | 3.06 | 1.91 / 1.69 | 3.04 | 1.94 | 15.28 | 13.55 | 1.73 |
| llm attn -> o | 968×2048 | 18 | — | 1.91 / 1.67 | (standalone) | | | | |
| llm rms2 -> gateup | 968×2048 | 18 | 3.73 | 2.21 / 2.02 | 3.71 | 2.22 | 132.60 | 128.54 | 3.94 |
| llm geglu -> down | 968×16384 | 18 | 17.11 | 9.03 / 7.59 | 17.50 | 8.63 | 110.80 | 79.69 | 31.09 |
| exp adarms1 -> qkv | 50×1024 | 180 | 1.15 | 0.69 / 0.82 | 1.21 | 0.63 | 5.71 | 4.68 | 1.03 |
| exp attn -> o | 50×2048 | 180 | — | 0.70 / 0.94 | (standalone) | | | | |
| exp adarms2 -> gateup | 50×1024 | 180 | 1.15 | 0.69 / 0.82 | 1.21 | 0.63 | 7.19 | 6.21 | 0.97 |
| exp geglu -> down | 50×4096 | 180 | 1.08 | 0.79 / 1.02 | 1.14 | 0.73 | 9.31 | 10.99 | -1.68 |

Saving per observation: isolated 0.722 ms, chain 0.860 ms. Standalone quantize left (attention outputs, context): 0.202 ms.

