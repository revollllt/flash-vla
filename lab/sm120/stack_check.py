import torch, json
print("torch", torch.__version__, "cuda", torch.version.cuda)
p = torch.cuda.get_device_properties(0); print("device", p.name, f"cc {p.major}.{p.minor}")

# --- A2a: bf16 matmul correctness on sm_120
a = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
b = torch.randn(768, 256, device="cuda", dtype=torch.bfloat16)
got = (a @ b).float()
want = (a.float() @ b.float())
err = (got - want).abs().max().item() / want.abs().max().item()
print(f"A2a bf16 matmul: max rel err {err:.3e} -> {'PASS' if err < 2e-2 else 'FAIL'}")

# --- A2b: CUDA graph capture + replay (runtime's core dependency)
try:
    static_in = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
    static_out = torch.empty(512, 256, device="cuda", dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_out.copy_(static_in @ b)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out.copy_(static_in @ b)
    static_in.copy_(a)
    g.replay()
    torch.cuda.synchronize()
    ref = (a @ b)
    rerr = (static_out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    print(f"A2b CUDA graph capture+replay: rel err {rerr:.3e} -> {'PASS' if rerr < 1e-6 else 'FAIL'}")
except Exception as e:
    print("A2b CUDA graph: FAIL ->", type(e).__name__, e)

# --- A3: CUPTI kernel activity tracing without counter permission
try:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            (a @ b)
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0)]
    kern = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
    print(f"A3 CUPTI: {len(kern)} entries with device time")
    for e in kern[:4]:
        print(f"    {e.key[:58]:60} {e.self_device_time_total:8.1f} us")
    print("A3 ->", "PASS (timeline available without counter permission)" if kern else "FAIL (no device timing)")
except Exception as e:
    print("A3 CUPTI: FAIL ->", type(e).__name__, e)
