import torch
from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import wrappers as cw
from flash_vla.hardware.nvidia.rtx5090.pi0.backends.cuda import pointwise as cu
from flash_vla.hardware.nvidia.rtx5090.pi0.backends import torch_ops as tt
DEV, DT = "cuda", torch.bfloat16
def cmp(name, ref, got):
    c = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    print(f"  {name:38} {'PASS' if c>=0.999 else '*** FAIL ***':12} cos {c:.6f}")
torch.manual_seed(0)
R = lambda *s: torch.randn(*s, device=DEV, dtype=DT)*0.1

# layer_norm kernel alone
x = R(768, 1152); w = R(1152); b = R(1152); o = torch.empty_like(x)
cu.layer_norm(x, w, b, o); torch.cuda.synchronize()
cmp("layer_norm kernel", tt._ln(x, w, b), o)

# gelu_ kernel alone
g = R(768, 4304); g2 = g.clone(); cu.gelu_(g2); torch.cuda.synchronize()
cmp("gelu_ kernel", tt._gelu(g), g2)

# residual addmm, out aliasing input
x = R(51, 1024); ww = R(1024, 1024)
o1 = R(51, 1024); o2 = o1.clone()
tt.action_expert_out_proj_residual(x, ww, None, o1)
cw.action_expert_out_proj_residual(x, ww, None, o2)
torch.cuda.synchronize(); cmp("residual addmm (out aliases in)", o1, o2)

# vision out-proj residual, res DISTINCT from out
xv = R(3, 256, 1152); wv = R(1152, 1152); bv = R(1152); res = R(3, 256, 1152)
oa = torch.empty_like(res); ob = torch.empty_like(res)
tt.vision_encoder_out_proj_residual(xv, wv, bv, res, oa)
cw.vision_encoder_out_proj_residual(xv, wv, bv, res, ob)
torch.cuda.synchronize(); cmp("vision out_proj (res distinct)", oa, ob)

# vision out-proj residual, res ALIASES out -- the graph may bind it that way
oa2 = res.clone(); ob2 = res.clone()
tt.vision_encoder_out_proj_residual(xv, wv, bv, oa2, oa2)
cw.vision_encoder_out_proj_residual(xv, wv, bv, ob2, ob2)
torch.cuda.synchronize(); cmp("vision out_proj (res ALIASES out)", oa2, ob2)

# vision norm_qkv / ffn_up
class S:
    def __init__(self): self.b = {}
    def __call__(self, n, shape, dt, dev):
        self.b.setdefault(n, torch.empty(shape, dtype=dt, device=dev)); return self.b[n]
sc = S()
qw = R(1152, 3456); qb = R(3456)
oa3 = torch.empty(3, 256, 3456, device=DEV, dtype=DT); ob3 = torch.empty_like(oa3)
tt.vision_encoder_norm_qkv(xv, w, b, qw, qb, oa3)
cw.vision_encoder_norm_qkv(xv, w, b, qw, qb, ob3, scratch=sc)
torch.cuda.synchronize(); cmp("vision_encoder_norm_qkv", oa3, ob3)

fw = R(1152, 4304); fb = R(4304)
oa4 = torch.empty(3, 256, 4304, device=DEV, dtype=DT); ob4 = torch.empty_like(oa4)
tt.vision_encoder_norm_ffn_up(xv, w, b, fw, fb, oa4)
cw.vision_encoder_norm_ffn_up(xv, w, b, fw, fb, ob4, scratch=sc)
torch.cuda.synchronize(); cmp("vision_encoder_norm_ffn_up", oa4, ob4)

# projector
pw = R(1152, 2048); pb = R(2048)
xn1 = torch.empty_like(xv); xn2 = torch.empty_like(xv)
oa5 = torch.empty(768, 2048, device=DEV, dtype=DT); ob5 = torch.empty_like(oa5)
tt.llm_backbone_projector(xv, w, b, pw, pb, oa5, xn1)
cw.llm_backbone_projector(xv, w, b, pw, pb, ob5, xn2)
torch.cuda.synchronize(); cmp("llm_backbone_projector", oa5, ob5)
