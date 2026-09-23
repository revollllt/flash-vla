| recipe | rel RMS mean | p90 | max | min cosine | max abs |
|---|---:|---:|---:|---:|---:|
| bf16 | 0.0000 | 0.0000 | 0.0000 | 1.00000 | 0.0000 |
| mxfp8 | 0.0060 | 0.0090 | 0.0374 | 0.99933 | 0.0938 |
| mxfp8_gate_up_only | 0.0058 | 0.0092 | 0.0260 | 0.99969 | 0.0938 |
| mxfp8_down_only | 0.0042 | 0.0060 | 0.0356 | 0.99937 | 0.0938 |

Sensitivity: all MXFP8 with one group in NVFP4, most sensitive first.

| NVFP4 group | rel RMS mean | increase over all-MXFP8 | max |
|---|---:|---:|---:|
