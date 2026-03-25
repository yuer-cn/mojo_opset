import time, torch
import sys
from mojo_opset.backends.ttx.kernels.npu.sample import top_k_sampling_impl

def top_k_sampling_ref(logits, top_k=10, min_tokens_to_keep=1):
    logits = logits.to(torch.float32)
    top_k = min(top_k, logits.size(-1))
    top_k = max(top_k, min_tokens_to_keep)
    
    sorted_topk_logits, sorted_topk_indices = torch.topk(logits, top_k)
    final_probs_dist = torch.nn.functional.softmax(sorted_topk_logits, dim=-1)
    select_index = torch.multinomial(final_probs_dist, num_samples=1)
    
    next_tokens = torch.gather(sorted_topk_indices, dim=-1, index=select_index)
    next_probs = torch.gather(final_probs_dist, dim=-1, index=select_index)
    
    return next_probs, next_tokens

print("Setting up tensor...", flush=True)
torch.manual_seed(42)
if hasattr(torch, 'npu'):
    torch.npu.manual_seed_all(42)
x = torch.randn(20, 151936, device='npu', dtype=torch.float32)

print("Checking accuracy vs PyTorch ref...", flush=True)
torch.manual_seed(42)
if hasattr(torch, 'npu'):
    torch.npu.manual_seed_all(42)
p_ref, t_ref = top_k_sampling_ref(x, top_k=10, min_tokens_to_keep=1)

torch.manual_seed(42)
if hasattr(torch, 'npu'):
    torch.npu.manual_seed_all(42)
# Ensure there are no caching effects internally
triton_out = top_k_sampling_impl(x, top_k=10, min_tokens_to_keep=1)

# Ensure type matches if top_k_sampling_impl returns differing order or tuple sizes
if isinstance(triton_out, tuple) and len(triton_out) == 2:
    p_impl, t_impl = triton_out[0], triton_out[1]
else:
    p_impl, t_impl = triton_out, triton_out 

print("Ref prob:", p_ref.flatten()[:5])
print("Imp prob:", p_impl.flatten()[:5])
print("Ref tokn:", t_ref.flatten()[:5])
print("Imp tokn:", t_impl.flatten()[:5] if hasattr(t_impl, 'flatten') else "N/A")

try:
    torch.testing.assert_close(p_impl.to(torch.float32), p_ref.to(torch.float32), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(t_impl.to(torch.int64), t_ref.to(torch.int64))
    print("Accuracy check PASSED!", flush=True)
except Exception as e:
    print("Accuracy check FAILED:", e, flush=True)

print("Starting loop...", flush=True)
for i in range(2):
    t0=time.time()
    print(f"iter {i} starting...", flush=True)
    p,t=top_k_sampling_impl(x, top_k=10, min_tokens_to_keep=1)
    print(f"iter {i} kernel done, synchronizing...", flush=True)
    torch.npu.synchronize()
    print(f"iter {i} time: {time.time()-t0}, out: {p.shape}, {t.shape}", flush=True)
