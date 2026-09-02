import sys, time, json, random
sys.path.insert(0, "/Users/rxiii/Documents/GitHub/PsiLM")
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim, mlx_lm
from transformers import AutoTokenizer
from psilm.mlx.staged import MlxStream
from psilm.mlx.fno import convert_from_torch
from psilm.mlx.bridges import PsiBridgesMLX
from psilm.mlx.model import PsiLMMLX
from eval.mlx_stage2_train import to_mlx_batch
from psilm.stage2.qa import QABuilder, make_batch as tmb

t0 = time.time()
model, tok = mlx_lm.load("mlx-community/Qwen3-8B-4bit")
print(f"loaded in {time.time()-t0:.0f}s: layers={len(model.model.layers)} hidden={model.args.hidden_size}", flush=True)
ids = mx.array([tok.encode("The capital of France is")])
ref = model(ids)
st = MlxStream(model, ids); st.run(0, len(model.model.layers))
diff = mx.abs(ref - st.finish()).max().item()
print(f"parity: {diff:.2e}", flush=True)
assert diff < 2e-2
hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
fno = convert_from_torch("results/stage2/fno.pt")
bridges = PsiBridgesMLX(d_model=model.args.hidden_size)
psi = PsiLMMLX(model, tok, fno, bridges)
builder = QABuilder(hf_tok)
items = json.loads(open("data/stage2_qa_train.json").read())
opt = optim.AdamW(learning_rate=3e-4)
def wrapped(b_, batch):
    psi.phi = b_
    return psi.loss_fn(batch)
lag = nn.value_and_grad(bridges, wrapped)
rng = random.Random(3)
for step in range(3):
    t0 = time.time()
    batch = to_mlx_batch(tmb(builder, rng.sample(items, 8), "cpu"))
    (loss, aux), grads = lag(bridges, batch)
    opt.update(bridges, grads)
    mx.eval(bridges.parameters(), opt.state)
    print(f"step {step}: loss={loss.item():.3f} {time.time()-t0:.2f}s (batch 8)", flush=True)
print("8B SETUP OK", flush=True)
