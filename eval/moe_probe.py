"""Feasibility probe for a MoE backbone: parity, memory, speed, routing hooks."""
import sys, time, json, random
sys.path.insert(0, "/Users/rxiii/Documents/GitHub/PsiLM")
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim, mlx_lm
from mlx.optimizers import clip_grad_norm
from transformers import AutoTokenizer
from psilm.mlx.moe_patch import patch_moe_gather, router_logits
print('patched:', patch_moe_gather(), flush=True)

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
t0 = time.time()
model, tok = mlx_lm.load(MODEL)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
inner = model.model
print(f"layers={len(inner.layers)} hidden={model.args.hidden_size}", flush=True)
blk = inner.layers[0]
print("block attrs:", [a for a in dir(blk) if not a.startswith('_')][:20], flush=True)
mlp = getattr(blk, "mlp", None)
print("mlp type:", type(mlp).__name__,
      "| has gate:", hasattr(mlp, "gate"),
      "| has switch:", hasattr(mlp, "switch_mlp"), flush=True)
for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "decoder_sparse_step"):
    if hasattr(model.args, k):
        print(f"  args.{k} = {getattr(model.args, k)}", flush=True)
print(f"peak mem after load: {mx.get_peak_memory()/1e9:.1f} GB", flush=True)

from psilm.mlx.staged import MlxStream
ids = mx.array([tok.encode("The capital of France is")])
ref = model(ids)
st = MlxStream(model, ids); st.run(0, len(inner.layers))
diff = mx.abs(ref - st.finish()).max().item()
print(f"PARITY: {diff:.2e}", flush=True)
assert diff < 2e-2, "parity failed"

from psilm.mlx.fno import convert_from_torch
from psilm.mlx.bridges import PsiBridgesMLX
from psilm.mlx.model import PsiLMMLX
from eval.mlx_stage2_train import to_mlx_batch
from psilm.stage2.qa import QABuilder, make_batch as tmb
hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
fno = convert_from_torch("results/stage2/fno.pt")
bridges = PsiBridgesMLX(d_model=model.args.hidden_size, gate_bias=0.0)
psi = PsiLMMLX(model, tok, fno, bridges, l_rev=round(len(inner.layers)*0.75))
builder = QABuilder(hf_tok)
items = json.loads(open("data/stage2_qa_train.json").read())
opt = optim.AdamW(learning_rate=3e-4)
def wrapped(b_, batch):
    psi.phi = b_
    return psi.loss_fn(batch)
lag = nn.value_and_grad(bridges, wrapped)
rng = random.Random(3)
times = []
for step in range(4):
    t0 = time.time()
    batch = to_mlx_batch(tmb(builder, rng.sample(items, 4), "cpu"))
    (loss, aux), grads = lag(bridges, batch)
    grads, _ = clip_grad_norm(grads, 1.0)
    opt.update(bridges, grads)
    mx.eval(bridges.parameters(), opt.state)
    dt = time.time()-t0; times.append(dt)
    print(f"step {step}: loss={loss.item():.3f} attn={aux[6].item():.3f} {dt:.1f}s (batch 4)", flush=True)
print(f"steady s/step (batch 4): {min(times):.1f} | peak mem: {mx.get_peak_memory()/1e9:.1f} GB", flush=True)

# MoE-specific diagnostic: does injection change expert routing?
from psilm.mlx.staged import MlxStream
item = json.loads(open("data/stage2_qa_val.json").read())[0]
ids2 = mx.array([builder.prompt_ids(item)])
probe_layer = psi.l_rev
s_un = MlxStream(model, ids2); s_un.run(0, probe_layer)
r_un = router_logits(model, s_un.hidden, probe_layer)
s_c = MlxStream(model, ids2)
psi._couple(s_c, mx.ones(ids2.shape, dtype=mx.bool_))
s_c2 = MlxStream(model, ids2); s_c2.run(0, psi.l_fwd)
pr, x0h, _, _ = psi.phi.fwd(s_c2.hidden, mx.ones(ids2.shape, dtype=mx.bool_))
from psilm.mlx.bridges import build_ic_mlx
ic = build_ic_mlx(pr); fts = fno.features(ic); uf = fno.proj(fts).squeeze(-1)
toks, _ = psi.phi.rev(fts, uf, x0h)
s_c2.run(psi.l_fwd, probe_layer)
h_inj, _ = psi.phi.inject(s_c2.hidden, toks)
r_c = router_logits(model, h_inj, probe_layer)
top_un = mx.argpartition(r_un, kth=-8, axis=-1)[..., -8:]
top_c = mx.argpartition(r_c, kth=-8, axis=-1)[..., -8:]
same = sum(len(set(top_un[0, i].tolist()) & set(top_c[0, i].tolist())) for i in range(top_un.shape[1]))
tot = top_un.shape[1] * 8
print(f"ROUTING: expert overlap {same}/{tot} = {same/tot:.2%} (untrained bridges; drift measurable)", flush=True)
print("MOE PROBE OK", flush=True)
