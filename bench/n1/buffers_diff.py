"""Find tensors that differ between a fp32 load and a bf16 load upcast to fp32 (CPU)."""
import gc, os, sys, torch
PROJECT = os.path.expanduser("~/acestep-f0/ACE-Step-1.5"); sys.path.insert(0, PROJECT)
from transformers import AutoModel
CK = os.path.join(PROJECT, "checkpoints", "acestep-v15-turbo")
def snap(dtype):
    m = AutoModel.from_pretrained(CK, trust_remote_code=True, dtype=dtype)
    m.decoder = None  # not needed, frees memory early
    gc.collect()
    m = m.to(torch.float32)
    bufs = {k: v.clone() for k, v in m.named_buffers()}
    pars = {k: (v.double().sum().item(), v.double().abs().sum().item()) for k, v in m.named_parameters()}
    del m; gc.collect(); return bufs, pars
b32, p32 = snap(torch.float32)
b16, p16 = snap(torch.bfloat16)
print("buffers", len(b32), len(b16), "params", len(p32), len(p16))
for k in b32:
    a, b = b32[k], b16.get(k)
    if b is None or a.shape != b.shape or not torch.equal(a, b):
        print("BUFFER DIFF", k, tuple(a.shape), a.dtype, "max_abs", None if b is None else (a - b).abs().max().item())
nd = [k for k in p32 if p32[k] != p16.get(k)]
print("param checksum diffs", len(nd), nd[:10])
