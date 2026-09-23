"""LIBERO's chunk 10 and the original chunk 50 must respect output bounds."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [10, 50])
@pytest.mark.parametrize("inner", [2048, 4096])
def test_expert_residual_rows(rows, inner):
    from flash_vla.hardware.nvidia.rtx5090.pi05.backends.cutlass_expert_residual import make_wrappers
    from flash_vla.runtime.runner import Scratch

    torch.manual_seed(7)
    # Dyadic inputs keep the FP32 reduction exact, exposing row and epilogue errors.
    x = (torch.randint(-2, 3, (rows, inner), device="cuda").float() / 32).bfloat16()
    weight = (torch.randint(-2, 3, (inner, 1024), device="cuda").float() / 32).bfloat16()
    gate = torch.linspace(-1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    storage = torch.full((rows + 4, 1024), 57, device="cuda", dtype=torch.bfloat16)
    out = storage[:rows]
    expected = ((x @ weight).float() * gate.float() + residual.float()).bfloat16()
    kernel = make_wrappers(Scratch(torch.device("cuda")))["action_expert_ffn_down_residual"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.copy_(residual)
        kernel(x, weight, gate, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out.copy_(residual)
            kernel(x, weight, gate, out)
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(expected, out)
    assert torch.all(storage[rows:] == 57)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("queries,keys", [(80, 722), (400, 1018)])
def test_qk_rectangular_bounds(queries, keys):
    import triton
    from flash_vla.hardware.nvidia.rtx5090.pi05.backends.triton_qk_attention import _qk

    q = (torch.randint(-2, 3, (queries, 256), device="cuda").float() / 32).bfloat16()
    k = (torch.randint(-2, 3, (keys, 256), device="cuda").float() / 32).bfloat16()
    storage = torch.full((queries + 1, keys), 57.0, device="cuda")
    expected = q.float() @ k.float().T
    _qk[(triton.cdiv(queries, 32), triton.cdiv(keys, 32))](
        q, k, storage[:queries], queries, keys, num_warps=4, num_stages=3,
        enable_fp_fusion=False, enable_reflect_ftz=False)
    torch.cuda.synchronize()
    assert torch.equal(storage[:queries], expected)
    assert torch.all(storage[queries:] == 57)
