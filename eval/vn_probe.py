"""Train the TD value probe per layer, rank the input dimensions, prune, retrain.

Stage 2 of the value-neuron identification of arXiv:2602.00986. Reads the
trajectories eval/vn_collect.py wrote and produces, for each layer, the index
list of the dimensions that carry state value -- the dimensions the constitution
bridge will write into at ``l_rev``.

The probe
  V(s) = sigmoid(w2 . relu(W1 s)), hidden 1024, AdamW lr 1e-4 wd 0.01, batch 4
  trajectories, 80/20 split by trajectory (seeded), gamma = 1 - 1e-5.

The TD loss, and the reading of it this file implements: a trajectory is a chain
of states s_0 .. s_{T-1} with reward only at the end, so with the paper's
delta_t = gamma V(s_t) - V(s_{t-1}) for the interior and the reward as the
target at the final state,

    interior  t = 1 .. T-1 :  delta_t = gamma V(s_t) - V(s_{t-1})
    terminal              :  delta_T = r - V(s_{T-1})
    loss = mean over all T residuals of delta^2

and the bootstrap target gamma V(s_t) is detached (semi-gradient TD(0), the
textbook form; --td-grad residual minimizes the full residual instead, which has
the same fixed point on deterministic trajectories). s_0 is the prompt-final
position; s_t (t >= 1) is the position of generated token t.

Standardization (--standardize, default zscore). The paper prunes by the L1 norm
of the first layer's weight columns, which is an importance measure only if the
input dimensions share a scale. They do not. Measured on Qwen2.5-0.5B layer 15
over the states of a rollout (the massive-activation attention sink at prompt
position 0 is not among them, so nothing saturates): per-dimension std 0.32
median with an 8x tail, per-dimension mean up to 7.6. That is enough to break the
statistic -- run --self-test --self-test-scaled, whose synthetic dimensions carry
the same scale profile with the signal planted at a fixed per-dimension SNR:
--standardize none puts 0 of 9 planted dimensions in the top 1% and the AUC after
99% pruning falls to chance (0.47), while zscore recovers 9 of 9 at AUC 1.00. So
each dimension is standardized with the mean/std of the TRAIN split's states
before the probe sees it -- the device psilm.mlx.bridges uses for its Gemma
readout (readout_norm="dim"), for the same reason. The ranking and the index
lists are over the original dimension indices either way, so a bridge or an
ablation still addresses raw stream dimensions. --standardize none reproduces the
literal recipe; rms keeps the offsets and normalizes the scale only.

Per layer it writes results/value_neurons/<tag>/layer<l>.json with exactly the
keys another script needs (layer, d_model, n_train, n_test, auc_full,
auc_by_ratio, auc_random99, ranking, top1pct, top5pct, epochs, sec_per_epoch)
plus layer<l>.extra.json for the bookkeeping that does not belong in that
contract, and results/value_neurons/<tag>/summary.json over layers.

Usage:
  .venv/bin/python eval/vn_probe.py --tag qwen0.5b --epochs 30
  .venv/bin/python eval/vn_probe.py --self-test --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from eval.bench_common import read_jsonl  # noqa: E402
from psilm.mlx.value_neurons import GAMMA, ValueProbe, auc_rank, keep_count  # noqa: E402

DEFAULT_RATIOS = "0,0.5,0.8,0.9,0.95,0.98,0.99"


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_trajs(root: Path, layer: int, max_n: Optional[int] = None) -> List[Dict]:
    """[{S: (T, d) float16, r: int, idx: int}] for one layer.

    Only the states of the trajectory are kept: row prompt_len - 1 (s_0, the
    prompt-final position) onwards. The prompt-internal positions are not states
    of the rollout and would swamp the TD chain with a deterministic prefix.
    """
    rows = read_jsonl(root / "index.jsonl")
    rows.sort(key=lambda r: int(r["idx"]))
    if max_n:
        rows = rows[:max_n]
    out = []
    for r in rows:
        p = root / "traj" / f"{int(r['idx'])}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            key = f"h{layer}"
            if key not in z:
                raise KeyError(f"{p} has no {key} (has {list(z.keys())})")
            S = np.asarray(z[key][int(r["prompt_len"]) - 1:])
        if S.shape[0] < 1:
            continue
        out.append({"S": S, "r": int(r["reward"]), "idx": int(r["idx"])})
    return out


def synth_trajs(n: int = 240, d: int = 896, n_signal: int = 9, seed: int = 0,
                t_lo: int = 20, t_hi: int = 60, scaled: bool = False) -> tuple:
    """Synthetic trajectories with the value signal planted in `n_signal` of `d`
    dimensions: the self-test's ground truth. Reward is a coin flip; the planted
    dimensions carry (2r-1) * 1.2 sigma_j at every position (so V(s_0) is already
    predictive), every other dimension is noise.

    scaled=True gives every dimension its own offset and scale, drawn to look
    like the real thing (Qwen2.5-0.5B layer 15 over generated positions:
    per-dim std 0.32 median with a 8x tail, per-dim mean up to 7.6). The signal
    is scaled with the dimension, so the detectability of a planted dimension is
    the same as in the unscaled case and only the pruning statistic changes --
    the L1 norm of a weight column is not scale-free, so this is the variant
    that says whether --standardize is needed."""
    rng = np.random.default_rng(seed)
    planted = np.sort(rng.choice(d, size=n_signal, replace=False))
    sig = (np.exp(rng.normal(np.log(0.32), 0.5, size=d)) if scaled
           else np.ones(d)).astype(np.float32)
    off = (rng.normal(0.0, 2.0, size=d) if scaled else np.zeros(d)).astype(np.float32)
    trajs = []
    for i in range(n):
        r = int(rng.integers(0, 2))
        T = int(rng.integers(t_lo, t_hi))
        S = (off + sig * rng.normal(0.0, 1.0, size=(T, d))).astype(np.float32)
        S[:, planted] += (2 * r - 1) * 1.2 * sig[planted]
        trajs.append({"S": S.astype(np.float16), "r": r, "idx": i})
    return trajs, planted


def split_trajs(trajs: List[Dict], seed: int, frac: float = 0.8):
    order = list(range(len(trajs)))
    random.Random(seed).shuffle(order)
    cut = int(round(frac * len(order)))
    return [trajs[i] for i in order[:cut]], [trajs[i] for i in order[cut:]]


def norm_stats(trajs: List[Dict], mode: str, d: int):
    """(mu, sigma) over all states of the given (train) trajectories."""
    if mode == "none":
        return np.zeros((d,), np.float32), np.ones((d,), np.float32)
    n = 0
    s1 = np.zeros((d,), np.float64)
    s2 = np.zeros((d,), np.float64)
    for t in trajs:
        X = t["S"].astype(np.float64)
        n += X.shape[0]
        s1 += X.sum(axis=0)
        s2 += (X * X).sum(axis=0)
    mu = s1 / max(1, n)
    var = np.maximum(s2 / max(1, n) - mu * mu, 0.0)
    if mode == "rms":                      # scale only, keep the mean offset
        mu = np.zeros_like(mu)
        var = s2 / max(1, n)
    return mu.astype(np.float32), (np.sqrt(var) + 1e-6).astype(np.float32)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

def prepare(trajs: Sequence[Dict], dims: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """Standardize once and keep only the selected dimensions.

    Done per training run rather than per step: the cast + standardization of a
    (T, 896) float16 block is the dominant cost of a step whose matmul is a few
    hundred MFLOP, and 30 epochs would pay it 30 times. Values are identical
    either way. Costs ~0.5 GB for 1000 trajectories at the full width, which is
    why train_probe drops the cache before it returns.
    """
    out = []
    for t in trajs:
        X = ((t["S"].astype(np.float32) - mu) / sigma)[:, dims]
        out.append({"X": np.ascontiguousarray(X), "r": t["r"]})
    return out


def _batch(trajs: Sequence[Dict]):
    """(X, prev, next, term, rewards) for a batch of prepared trajectories.

    X stacks every state of every trajectory in the batch; prev/next index the
    interior TD transitions and term the final state of each trajectory, so one
    probe forward covers the whole batch.
    """
    mats, prev, nxt, term, rew = [], [], [], [], []
    off = 0
    for t in trajs:
        X = t["X"]
        mats.append(X)
        T = X.shape[0]
        if T > 1:
            prev.append(np.arange(off, off + T - 1))
            nxt.append(np.arange(off + 1, off + T))
        term.append(off + T - 1)
        rew.append(float(t["r"]))
        off += T
    X = np.concatenate(mats, axis=0)
    p = np.concatenate(prev) if prev else None
    q = np.concatenate(nxt) if nxt else None
    # the reward of every state's trajectory, for the Monte-Carlo target
    rew_all = np.concatenate([np.full(t["X"].shape[0], float(t["r"]), np.float32) for t in trajs])
    return (mx.array(X),
            None if p is None else mx.array(p.astype(np.int32)),
            None if q is None else mx.array(q.astype(np.int32)),
            mx.array(np.asarray(term, dtype=np.int32)),
            mx.array(np.asarray(rew, dtype=np.float32)),
            mx.array(rew_all))


def _loss(probe, X, prev, nxt, term, rew, rew_all, gamma, semi, target="td"):
    """target "td": the paper's temporal-difference loss (interior transitions
    bootstrap on the next state, the final state regresses on the reward).
    target "mc": every state regresses on its trajectory's reward directly --
    the fixed point the TD loss converges to at gamma -> 1, reached without
    the backward propagation TD needs. With one reward-bearing residual among
    ~230 per trajectory, TD at 30 epochs left V nearly constant on the 0.5B
    (held-out AUC 0.42-0.60 across layers) where the same probe on the MC
    target reaches 0.67-0.69 from the prompt-final position in 8 epochs
    (2026-09-12); the ranking and pruning are unchanged by the choice."""
    v = probe(X)
    if target == "mc":
        res = v - rew_all
        return (res * res).mean()
    parts = []
    if prev is not None:
        tgt = gamma * v[nxt]
        parts.append(v[prev] - (mx.stop_gradient(tgt) if semi else tgt))
    parts.append(v[term] - rew)
    res = mx.concatenate(parts) if len(parts) > 1 else parts[0]
    return (res * res).mean()


def v_of_s0(probe, trajs: Sequence[Dict], chunk: int = 256):
    """V at s_0 (the first state, the prompt-final position) for every prepared
    trajectory."""
    out = []
    for a in range(0, len(trajs), chunk):
        blk = trajs[a:a + chunk]
        X = np.stack([t["X"][0] for t in blk])
        v = probe(mx.array(X))
        mx.eval(v)
        out.extend(float(x) for x in np.array(v))
    return out


def train_probe(train: Sequence[Dict], test: Sequence[Dict], dims: np.ndarray,
                mu, sigma, epochs: int, seed: int, lr: float, wd: float, batch: int,
                gamma: float, semi: bool, verbose: bool = False, target: str = "td"):
    """-> (auc, sec_per_epoch, final_loss, probe). `dims` selects the input
    dimensions (the retained ones after pruning); AUC is AUC(V(s_0), reward) on
    the held-out trajectories."""
    train = prepare(train, dims, mu, sigma)
    test = prepare(test, dims, mu, sigma)
    mx.random.seed(seed)
    probe = ValueProbe(int(dims.size))
    mx.eval(probe.parameters())
    opt = optim.AdamW(learning_rate=lr, weight_decay=wd)
    step = nn.value_and_grad(probe, lambda p, *a: _loss(p, *a, gamma, semi, target))
    rng = random.Random(seed)
    order = list(range(len(train)))
    t0 = time.perf_counter()
    last = float("nan")
    for ep in range(epochs):
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for a in range(0, len(order), batch):
            blk = [train[i] for i in order[a:a + batch]]
            X, prev, nxt, term, rew, rew_all = _batch(blk)
            loss, grads = step(probe, X, prev, nxt, term, rew, rew_all)
            opt.update(probe, grads)
            mx.eval(probe.parameters(), opt.state, loss)
            tot += float(loss)
            nb += 1
        last = tot / max(1, nb)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep + 1}/{epochs} loss={last:.5f}", flush=True)
    sec_per_epoch = (time.perf_counter() - t0) / max(1, epochs)
    auc = auc_rank(v_of_s0(probe, test), [t["r"] for t in test])
    del train, test
    return auc, sec_per_epoch, last, probe


def run_layer(trajs: List[Dict], layer: int, args, ratios: Sequence[float]) -> Dict:
    """The whole per-layer protocol: full probe -> ranking -> retrain per ratio
    -> random-dims control at 0.99."""
    d = int(trajs[0]["S"].shape[1])
    train, test = split_trajs(trajs, args.seed)
    mu, sigma = norm_stats(train, args.standardize, d)
    all_dims = np.arange(d, dtype=np.int64)
    kw = dict(mu=mu, sigma=sigma, epochs=args.epochs, seed=args.seed, lr=args.lr,
              wd=args.wd, batch=args.batch, gamma=args.gamma, target=args.target,
              semi=(args.td_grad == "semi"))
    t0 = time.perf_counter()
    auc_full, spe, loss, probe = train_probe(train, test, all_dims, verbose=args.verbose, **kw)
    col = probe.column_l1()
    mx.eval(col)
    col = np.array(col, dtype=np.float64)
    ranking = np.argsort(-col, kind="stable")
    print(f"  layer {layer}: d={d} n_train={len(train)} n_test={len(test)} "
          f"auc_full={auc_full} loss={loss:.5f} sec/epoch={spe:.2f}", flush=True)

    auc_by_ratio, keeps = {}, {}
    for p in ratios:
        k = keep_count(d, p)
        keeps[f"{p:g}"] = k
        if k == d:                 # ratio 0: the retrain is the full probe, same seed
            auc_by_ratio[f"{p:g}"] = auc_full
            print(f"    ratio {p:g} keep={k} auc={auc_full} (= full probe)", flush=True)
            continue
        dims = np.sort(ranking[:k]).astype(np.int64)
        a, _, l, _ = train_probe(train, test, dims, **kw)
        auc_by_ratio[f"{p:g}"] = a
        print(f"    ratio {p:g} keep={k} auc={a} loss={l:.5f}", flush=True)

    k99 = keep_count(d, 0.99)
    auc_random99 = []
    for s in range(args.random_seeds):
        rs = np.random.default_rng(10_000 + s).choice(d, size=k99, replace=False)
        a, _, l, _ = train_probe(train, test, np.sort(rs).astype(np.int64), **kw)
        auc_random99.append(a)
        print(f"    random99 seed={s} keep={k99} auc={a} loss={l:.5f}", flush=True)

    n1 = keep_count(d, 0.99)
    n5 = keep_count(d, 0.95)
    rec = {"layer": int(layer), "d_model": d, "n_train": len(train), "n_test": len(test),
           "auc_full": auc_full, "auc_by_ratio": auc_by_ratio, "auc_random99": auc_random99,
           "ranking": [int(i) for i in ranking],
           "top1pct": sorted(int(i) for i in ranking[:n1]),
           "top5pct": sorted(int(i) for i in ranking[:n5]),
           "epochs": int(args.epochs), "sec_per_epoch": round(spe, 3)}
    extra = {"layer": int(layer), "standardize": args.standardize, "td_grad": args.td_grad,
             "gamma": args.gamma, "lr": args.lr, "wd": args.wd, "batch": args.batch,
             "seed": args.seed, "keep_by_ratio": keeps, "n_random_seeds": args.random_seeds,
             "reward_rate_train": round(float(np.mean([t["r"] for t in train])), 4),
             "reward_rate_test": round(float(np.mean([t["r"] for t in test])), 4),
             "states_train": int(sum(t["S"].shape[0] for t in train)),
             "states_test": int(sum(t["S"].shape[0] for t in test)),
             "final_loss_full": round(float(loss), 6),
             "col_l1_top": [round(float(col[i]), 4) for i in ranking[:n1]],
             "col_l1_median": round(float(np.median(col)), 4),
             "mu_top1pct": [round(float(mu[i]), 3) for i in ranking[:n1]],
             "sigma_top1pct": [round(float(sigma[i]), 3) for i in ranking[:n1]],
             "sec_layer": round(time.perf_counter() - t0, 1)}
    return {"rec": rec, "extra": extra, "ranking": ranking}


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="qwen0.5b")
    ap.add_argument("--out-root", default="results/value_neurons")
    ap.add_argument("--layers", default="", help="default: every layer in the index")
    ap.add_argument("--candidate-rev", default="12,14,15,16,18",
                    help="layers eligible to be the chosen injection depth")
    ap.add_argument("--epochs", type=int, default=30,
                    help="the paper used 100; sec_per_epoch is reported so a full run can choose")
    ap.add_argument("--batch", type=int, default=4, help="trajectories per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", default=DEFAULT_RATIOS)
    ap.add_argument("--random-seeds", type=int, default=3, help="random-dims controls at 0.99")
    ap.add_argument("--td-grad", default="semi", choices=["semi", "residual"])
    ap.add_argument("--target", default="td", choices=["td", "mc"],
                    help="td: the paper's TD loss; mc: regress every state on its trajectory's "
                         "reward (the gamma->1 fixed point, better conditioned; see _loss)")
    ap.add_argument("--resume", action="store_true",
                    help="skip layers whose layer<l>.json already exists (a killed run)")
    ap.add_argument("--standardize", default="zscore", choices=["zscore", "rms", "none"])
    ap.add_argument("--max-n", type=int, default=0, help="cap trajectories (debug)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-n", type=int, default=240)
    ap.add_argument("--self-test-scaled", action="store_true",
                    help="give the synthetic dimensions heterogeneous offsets/scales "
                         "(the variant that exercises --standardize)")
    return ap.parse_args()


def self_test(args, ratios):
    d, n_sig = 896, 9
    trajs, planted = synth_trajs(n=args.self_test_n, d=d, n_signal=n_sig, seed=args.seed,
                                 scaled=args.self_test_scaled)
    print(f"self-test: n={len(trajs)} d={d} scaled={args.self_test_scaled} "
          f"standardize={args.standardize} planted={[int(i) for i in planted]}")
    out = run_layer(trajs, -1, args, ratios)
    rec, ranking = out["rec"], out["ranking"]
    top1 = set(rec["top1pct"])
    hit = sorted(set(int(i) for i in planted) & top1)
    auc99 = rec["auc_by_ratio"]["0.99"]
    rnd = [a for a in rec["auc_random99"] if a is not None]
    print(f"  planted in top1pct: {len(hit)}/{n_sig} {hit}")
    print(f"  auc_full={rec['auc_full']} auc@0.99={auc99} random99={rnd}")
    assert len(hit) == n_sig, f"ranking missed planted dims: {len(hit)}/{n_sig}"
    assert auc99 is not None and auc99 > 0.95, f"auc at 0.99 pruning too low: {auc99}"
    assert rnd and abs(float(np.mean(rnd)) - 0.5) < 0.15, f"random control not chance: {rnd}"
    print("VN-PROBE SELF-TEST OK")


def main():
    args = parse_args()
    ratios = [float(x) for x in args.ratios.split(",") if x.strip() != ""]
    if args.self_test:
        self_test(args, ratios)
        return
    root = Path(args.out_root) / args.tag
    rows = read_jsonl(root / "index.jsonl")
    assert rows, f"no trajectories in {root / 'index.jsonl'}"
    have = rows[0]["layers"]
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""] or list(have)
    missing = [l for l in layers if l not in have]
    assert not missing, f"layers {missing} were not collected (have {have})"
    print(f"tag={args.tag} trajectories={len(rows)} layers={layers} "
          f"reward_rate={np.mean([r['reward'] for r in rows]):.3f} "
          f"standardize={args.standardize} target={args.target} td_grad={args.td_grad} epochs={args.epochs}")

    summary = {"tag": args.tag, "n_trajectories": len(rows), "layers": {},
               "args": {k: v for k, v in vars(args).items()}}
    for l in layers:
        if args.resume and (root / f"layer{l}.json").exists() and (root / f"layer{l}.extra.json").exists():
            out = {"rec": json.loads((root / f"layer{l}.json").read_text()),
                   "extra": json.loads((root / f"layer{l}.extra.json").read_text())}
            print(f"  layer {l}: resumed from disk (auc_full={out['rec']['auc_full']})", flush=True)
        else:
            trajs = load_trajs(root, l, args.max_n or None)
            out = run_layer(trajs, l, args, ratios)
            (root / f"layer{l}.json").write_text(json.dumps(out["rec"]))
            (root / f"layer{l}.extra.json").write_text(json.dumps(out["extra"], indent=1))
        rnd = [a for a in out["rec"]["auc_random99"] if a is not None]
        summary["layers"][str(l)] = {
            "auc_full": out["rec"]["auc_full"],
            "auc_99": out["rec"]["auc_by_ratio"].get("0.99"),
            "random99_mean": (round(float(np.mean(rnd)), 4) if rnd else None),
            "sec_per_epoch": out["rec"]["sec_per_epoch"],
            "sec_layer": out["extra"]["sec_layer"]}
        trajs = None
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    cands = [int(x) for x in args.candidate_rev.split(",") if x.strip() != ""]
    scored = [(summary["layers"][str(l)]["auc_99"], l) for l in cands
              if str(l) in summary["layers"] and summary["layers"][str(l)]["auc_99"] is not None]
    summary["candidate_rev"] = cands
    summary["chosen_layer"] = (max(scored)[1] if scored else None)
    summary["chosen_auc_99"] = (max(scored)[0] if scored else None)
    (root / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{'layer':>6s} {'auc_full':>9s} {'auc@0.99':>9s} {'random99':>9s} "
          f"{'sec/epoch':>10s} {'sec/layer':>10s}")
    print("-" * 60)
    for l in layers:
        b = summary["layers"][str(l)]
        f = lambda v: ("  none" if v is None else f"{v:.4f}")  # noqa: E731
        print(f"{l:6d} {f(b['auc_full']):>9s} {f(b['auc_99']):>9s} {f(b['random99_mean']):>9s} "
              f"{b['sec_per_epoch']:10.2f} {b['sec_layer']:10.1f}")
    print(f"chosen layer (best auc@0.99 among {cands}): {summary['chosen_layer']} "
          f"auc={summary['chosen_auc_99']}")
    print(f"VN-PROBE DONE layers={len(layers)}")


if __name__ == "__main__":
    main()
