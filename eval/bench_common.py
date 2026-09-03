"""Shared pieces for the PsiLM benchmark harnesses (guard-rail, future A/B runs).

Contents
  * dataset selection: GSM8K test (seeded prefix), a fixed MMLU 5-subject
    slice, and the PsiLM physics QA control (data/stage2_qa_val.json)
  * prompt construction with the Qwen chat template, enable_thinking=False,
    normalized exactly as psilm/stage2/qa.py does
  * the answer-marker protocol used by the existing evals
    ("Answer: <number>" / "Answer: <letter>") plus the trained physics reply
    ("u at x = 0.86 equals -0.10." -> last decimal number), and parsers
  * StagedDecoder: KV-cached greedy decoding through the staged backbone in
    three arms on the SAME weights
        base    frozen LLM alone (no bridges touched)
        psilm   bridges attached, gate free
        zeroed  bridges attached, the physics injection multiplied by 0;
                the gate sigma is still computed and reported
    psilm/mlx/model.py's PsiLMMLX.generate re-runs the full sequence every
    step (no cache) - fine for a 24-token physics reply, prohibitive for a
    GSM8K chain of thought at 8B. The coupling only reads the prompt (the
    forward bridge pools over prompt_mask, so the physics tokens are a
    constant of the prompt) and the injection is per position, so a cached
    decode that re-injects the same tokens at every new position is the same
    computation. Verified against the full-recompute path on a synthetic
    backbone (see bench_guardrail.py --self-test).
  * gate-sigma statistics per question, per-arm aggregates, and a paired
    (McNemar) comparison of arms, which is far more sensitive than raw
    accuracy differences at N=100.

Nothing here loads model weights at import time; mlx_lm / datasets /
transformers are imported lazily inside the functions that need them, so
the --dry-run path of a harness stays weight-free.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# protocol constants (kept textually identical to the existing evals)
# ----------------------------------------------------------------------------

SYSTEM = "You are a helpful assistant."          # psilm.stage2.qa.SYSTEM
# eval/mlx_stage2_eval.py NUDGE / eval/stage2_eval.py nudge
NUDGE_NUMBER = "\nEnd your reply with a line of the form \"Answer: <number>\"."
NUDGE_LETTER = ("\nAnswer with only the letter (A, B, C, or D) of the correct "
                "choice, in the form \"Answer: <letter>\".")

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_HF_TOKENIZER = "Qwen/Qwen3-8B"
DEFAULT_CKPT = "results/stage2_mlx8b4/bridges.npz"
DEFAULT_FNO = "results/stage2/fno.pt"
PHYSICS_DATA = "data/stage2_qa_val.json"
PHYSICS_TOL = 0.05                               # acc@0.05, as in every stage-2 eval

# A fixed, documented slice. college_physics is a deliberate near-domain
# distractor: if the gate opens on generic physics vocabulary rather than on
# the Burgers prompt the bridges were trained on, it shows up here first.
MMLU_SUBJECTS = ["high_school_mathematics", "college_physics", "philosophy",
                 "high_school_biology", "us_foreign_policy"]

ARMS = ("base", "psilm", "zeroed")
LETTERS = "ABCD"


# ----------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------

@dataclass
class Prompt:
    ids: List[int]
    text: str
    protocol: str                 # number | letter | physics_trained | physics_nudge
    max_new: int
    x0_span: Optional[Tuple[int, int]] = None
    span_fallback: bool = False   # QABuilder.x0_span hit its whole-prompt fallback


@dataclass
class Task:
    dataset: str                  # gsm8k | mmlu | physics
    qid: str
    gold: Any
    prompt: Prompt
    meta: Dict[str, Any] = field(default_factory=dict)
    arm_prompts: Dict[str, Prompt] = field(default_factory=dict)

    def prompt_for(self, arm: str) -> Prompt:
        return self.arm_prompts.get(arm, self.prompt)


@dataclass
class GenResult:
    gen_ids: List[int]
    text: str
    stopped_eos: bool
    sigma_prompt: List[float]     # gate at every prompt position (empty in base)
    sigma_gen: List[float]        # gate at every generated position fed back in
    diag: Dict[str, Any]
    sec: float


# ----------------------------------------------------------------------------
# tokenization / prompts
# ----------------------------------------------------------------------------

def chat_ids(tok, user: str, system: str = SYSTEM) -> List[int]:
    """Chat-templated prompt ids, enable_thinking=False, normalized the way
    psilm.stage2.qa.QABuilder.prompt_ids does (transformers 5 returns a
    BatchEncoding)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    if not isinstance(out, list):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def eos_id_set(hf_tok, mlx_tok=None) -> set:
    ids = set()
    if hf_tok.eos_token_id is not None:
        ids.add(int(hf_tok.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = hf_tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != getattr(hf_tok, "unk_token_id", None):
            ids.add(i)
    extra = getattr(mlx_tok, "eos_token_ids", None) if mlx_tok is not None else None
    if extra:
        ids |= {int(i) for i in extra}
    return ids


def mmlu_user(question: str, choices: List[str]) -> str:
    lines = [question.strip(), ""]
    for L, c in zip(LETTERS, choices):
        lines.append(f"{L}. {c}")
    return "\n".join(lines) + "\n" + NUDGE_LETTER


def gsm8k_user(question: str, nudge: bool = True) -> str:
    return question.strip() + (NUDGE_NUMBER if nudge else "")


def gsm8k_gold(answer: str) -> float:
    return float(answer.split("####")[-1].strip().replace(",", "").replace("$", ""))


# ----------------------------------------------------------------------------
# parsers
# ----------------------------------------------------------------------------

_NUM = r"(-?\$?[\d,]*\.?\d+)"
_ANS_NUM = re.compile(r"Answer\s*:?\s*\**\s*\$?\s*(?:\\boxed\{)?\s*" + _NUM, re.I)
_ANY_NUM = re.compile(r"-?[\d,]*\.?\d+")
_DECIMAL = re.compile(r"-?\d+\.\d+")
_ANS_LET = re.compile(r"Answer\s*:?\s*\**\s*\(?([ABCD])\b", re.I)
_LINE_LET = re.compile(r"(?:^|\n)\s*\(?([ABCD])[\.\):]")
_OPT_LET = re.compile(r"\b(?:option|choice)\s*\(?([ABCD])\b", re.I)


def _to_float(s: str) -> Optional[float]:
    s = s.replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_number(text: str, fallback: str = "any") -> Optional[float]:
    """'Answer: <number>' protocol (last marker wins; tolerates **bold**, $,
    \\boxed{}, thousands separators). Fallback: the last number in the text
    ('any') or the last decimal ('decimal', the trained physics reply)."""
    m = _ANS_NUM.findall(text)
    if m:
        v = _to_float(m[-1])
        if v is not None:
            return v
    pat = _DECIMAL if fallback == "decimal" else _ANY_NUM
    m = [x for x in pat.findall(text) if _to_float(x) is not None]
    return _to_float(m[-1]) if m else None


def parse_last_decimal(text: str) -> Optional[float]:
    """The trained physics reply protocol (eval/mlx_stage2_train.parse_value)."""
    m = _DECIMAL.findall(text)
    return float(m[-1]) if m else None


def parse_letter(text: str) -> Optional[str]:
    m = _ANS_LET.findall(text)
    if m:
        return m[-1].upper()
    m = _LINE_LET.findall(text)
    if m:
        return m[0]
    m = _OPT_LET.findall(text)
    if m:
        return m[0].upper()
    return None


def score(protocol: str, text: str, gold: Any, num_tol: float = 1e-6):
    """-> (pred, ok)."""
    if protocol == "number":
        p = parse_number(text)
        ok = p is not None and abs(p - gold) <= num_tol * max(1.0, abs(gold))
    elif protocol == "letter":
        p = parse_letter(text)
        ok = p is not None and p == gold
    elif protocol == "physics_trained":
        p = parse_last_decimal(text)
        ok = p is not None and abs(p - gold) <= PHYSICS_TOL
    elif protocol == "physics_nudge":
        p = parse_number(text, fallback="decimal")
        ok = p is not None and abs(p - gold) <= PHYSICS_TOL
    else:
        raise ValueError(protocol)
    return p, bool(ok)


# ----------------------------------------------------------------------------
# datasets (CPU-only downloads via `datasets`)
# ----------------------------------------------------------------------------

def _seeded_prefix(n_total: int, n: int, seed) -> List[int]:
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    return idx[:n]


def load_gsm8k(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in _seeded_prefix(len(ds), n, seed):
        r = ds[i]
        out.append({"qid": f"gsm8k:test:{i}", "question": r["question"],
                    "gold": gsm8k_gold(r["answer"]), "index": i})
    return out


def load_mmlu(n: int, seed: int, subjects: List[str] = MMLU_SUBJECTS) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    per = [n // len(subjects) + (1 if k < n % len(subjects) else 0) for k in range(len(subjects))]
    out = []
    for subj, k in zip(subjects, per):
        ds = load_dataset("cais/mmlu", subj, split="test")
        for i in _seeded_prefix(len(ds), k, f"{seed}:{subj}"):
            r = ds[i]
            out.append({"qid": f"mmlu:{subj}:{i}", "question": r["question"],
                        "choices": list(r["choices"]), "gold": LETTERS[int(r["answer"])],
                        "subject": subj, "index": i})
    return out


def load_physics(n: int, path: str = PHYSICS_DATA) -> List[Dict[str, Any]]:
    items = json.loads(Path(path).read_text())[:n]
    return [{"qid": f"physics:{Path(path).stem}:{i}", "item": it, "gold": it["u"], "index": i}
            for i, it in enumerate(items)]


# ----------------------------------------------------------------------------
# task construction
# ----------------------------------------------------------------------------

def build_tasks(dataset: str, records: List[Dict[str, Any]], hf_tok, max_new: int,
                builder=None, physics_base_protocol: str = "nudge",
                max_new_physics_base: int = 160, nonphys_span: str = "whole",
                gsm8k_nudge: bool = True) -> List[Task]:
    """Turn dataset records into Tasks (ids + protocol + span).

    nonphys_span: what the forward bridge's x0 pointer sees on a question that
    has no x0. 'whole' = mean over the prompt, which is exactly what
    QABuilder.x0_span's fallback yields for a prompt with no match (the
    deployment-faithful choice); 'learned' = the untrained attention pointer.
    """
    tasks = []
    if dataset == "gsm8k":
        for r in records:
            ids = chat_ids(hf_tok, gsm8k_user(r["question"], nudge=gsm8k_nudge))
            span = (0, len(ids)) if nonphys_span == "whole" else None
            p = Prompt(ids, hf_tok.decode(ids), "number", max_new, span)
            tasks.append(Task("gsm8k", r["qid"], r["gold"], p,
                              {"index": r["index"]}))
    elif dataset == "mmlu":
        for r in records:
            ids = chat_ids(hf_tok, mmlu_user(r["question"], r["choices"]))
            span = (0, len(ids)) if nonphys_span == "whole" else None
            p = Prompt(ids, hf_tok.decode(ids), "letter", max_new, span)
            tasks.append(Task("mmlu", r["qid"], r["gold"], p,
                              {"index": r["index"], "subject": r["subject"]}))
    elif dataset == "physics":
        from psilm.stage2.qa import QUESTION
        assert builder is not None, "physics tasks need a QABuilder"
        for r in records:
            it = r["item"]
            ids = builder.prompt_ids(it)
            span = tuple(builder.x0_span(ids, it))
            fb = span == (0, len(ids))
            p = Prompt(ids, hf_tok.decode(ids), "physics_trained", max_new, span, fb)
            t = Task("physics", r["qid"], r["gold"], p, {"index": r["index"], "item": it})
            if physics_base_protocol == "nudge":
                q = QUESTION.format(a=it["a"], phi=it["phi"], x0=it["x0"]) + NUDGE_NUMBER
                bids = chat_ids(hf_tok, q)
                t.arm_prompts["base"] = Prompt(bids, hf_tok.decode(bids), "physics_nudge",
                                               max_new_physics_base, None)
            tasks.append(t)
    else:
        raise ValueError(dataset)
    return tasks


def task_manifest(t: Task, hf_tok=None, full_text: bool = False) -> Dict[str, Any]:
    d = {"dataset": t.dataset, "qid": t.qid, "gold": t.gold, "meta": t.meta,
         "n_tokens": len(t.prompt.ids), "protocol": t.prompt.protocol,
         "max_new": t.prompt.max_new, "x0_span": t.prompt.x0_span,
         "span_fallback": t.prompt.span_fallback}
    if t.prompt.x0_span is not None and hf_tok is not None and t.dataset == "physics":
        s0, s1 = t.prompt.x0_span
        d["span_text"] = hf_tok.decode(t.prompt.ids[s0:s1])
    if full_text:
        d["prompt_text"] = t.prompt.text
    for arm, p in t.arm_prompts.items():
        d[f"{arm}_protocol"] = p.protocol
        d[f"{arm}_n_tokens"] = len(p.ids)
        d[f"{arm}_max_new"] = p.max_new
        if full_text:
            d[f"{arm}_prompt_text"] = p.text
    return d


# ----------------------------------------------------------------------------
# backbone / bridges loading (weights: only called on a real run)
# ----------------------------------------------------------------------------

def load_backbone(model_id: str = DEFAULT_MODEL, hf_tok_id: str = DEFAULT_HF_TOKENIZER):
    import mlx_lm
    from transformers import AutoTokenizer
    model, tok = mlx_lm.load(model_id)
    model.freeze()
    hf_tok = AutoTokenizer.from_pretrained(hf_tok_id)
    return model, tok, hf_tok


def load_physics_stack(ckpt: str, d_model: int, gate_bias: float = -2.0, fno_path: str = DEFAULT_FNO):
    import mlx.core as mx
    from psilm.mlx.bridges import PsiBridgesMLX
    from psilm.mlx.fno import convert_from_torch
    fno = convert_from_torch(fno_path)
    fno.freeze()
    meta_p = Path(str(ckpt) + ".meta")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    # v6+ checkpoints record their construction; older ones fall back to the CLI value
    margs = meta.get("args", {})
    bridges = PsiBridgesMLX(d_model=d_model, gate_bias=margs.get("gate_bias", gate_bias),
                            inj_cap=margs.get("inj_cap"), channel=margs.get("channel", "field"))
    bridges.load_weights(str(ckpt))
    bridges.freeze()
    mx.eval(bridges.parameters(), fno.parameters())
    return fno, bridges, meta


# ----------------------------------------------------------------------------
# the three-arm KV-cached staged decoder
# ----------------------------------------------------------------------------

class StagedDecoder:
    """Greedy decoding through model.model.layers with a KV cache, pausing at
    l_fwd (read the prompt into the physics model) and l_rev (inject).

    Layer split mirrors PsiLMMLX: l_fwd = round(n*10/24), l_rev = round(n*15/24)
    unless given. NOTE: the 8B v5 run used --l-rev 27 (36 layers); the
    checkpoint meta does not record it, so pass it explicitly.
    """

    def __init__(self, model, hf_tok, fno=None, bridges=None, l_fwd=None, l_rev=None,
                 eos_ids=None):
        import mlx.core as mx  # noqa: F401  (import check)
        self.model = model
        self.inner = model.model
        self.hf_tok = hf_tok
        self.fno = fno
        self.phi = bridges
        n = len(self.inner.layers)
        self.n_layers = n
        self.l_fwd = l_fwd if l_fwd is not None else round(n * 10 / 24)
        self.l_rev = l_rev if l_rev is not None else round(n * 15 / 24)
        assert 0 < self.l_fwd <= self.l_rev <= n, (self.l_fwd, self.l_rev, n)
        self.eos_ids = set(eos_ids) if eos_ids is not None else eos_id_set(hf_tok)

    # -- pieces -------------------------------------------------------------
    def _layers(self, h, lo, hi, mask, cache):
        for i in range(lo, hi):
            h = self.inner.layers[i](h, mask=mask, cache=cache[i])
        return h

    def _logits_last(self, h):
        h = self.inner.norm(h[:, -1:, :])
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(h)[:, -1, :]
        return self.inner.embed_tokens.as_linear(h)[:, -1, :]

    def _physics_tokens(self, h_prompt, x0_span):
        import mlx.core as mx
        from psilm.mlx.bridges import build_ic_mlx
        L = h_prompt.shape[1]
        pmask = mx.ones((1, L), dtype=mx.bool_)
        span = None if x0_span is None else mx.array([list(x0_span)], dtype=mx.int32)
        params_hat, x0_hat, x0_logits, w_x0 = self.phi.fwd(h_prompt, pmask, span)
        ic = build_ic_mlx(params_hat)
        feats = self.fno.features(ic)
        u_field = self.fno.proj(feats).squeeze(-1)
        tokens, u_hat = self.phi.rev(feats, u_field, x0_hat)
        if getattr(self.phi, "channel", "field") == "value":
            tokens = self.phi.val(u_hat)          # mirrors PsiLMMLX._couple
        mx.eval(tokens, u_hat, params_hat, x0_hat)
        diag = {"x0_hat": round(float(x0_hat[0].item()), 4),
                "u_hat": round(float(u_hat[0].item()), 4),
                "params_hat": [round(float(v), 4) for v in params_hat[0].tolist()]}
        return tokens, diag

    def _inject(self, h, tokens, mode):
        h_inj, sigma = self.phi.inject(h, tokens)
        if mode == "psilm":
            return h_inj, sigma
        return h, sigma                      # zeroed: gate measured, hidden untouched

    # -- public ---------------------------------------------------------------
    def prefill_logits(self, prompt_ids, mode="base", x0_span=None):
        """Last-position logits after a prefill (used by the parity check and
        the self-test); returns (logits, cache, tokens, sigma_prompt, diag)."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        h = self.inner.embed_tokens(mx.array([list(prompt_ids)]))
        mask = "causal" if len(prompt_ids) > 1 else None
        tokens, sig_p, diag = None, None, {}
        if mode == "base":
            h = self._layers(h, 0, self.n_layers, mask, cache)
        else:
            assert self.phi is not None and self.fno is not None, "bridges/fno required"
            h = self._layers(h, 0, self.l_fwd, mask, cache)
            tokens, diag = self._physics_tokens(h, x0_span)
            h = self._layers(h, self.l_fwd, self.l_rev, mask, cache)
            h, sig = self._inject(h, tokens, mode)
            h = self._layers(h, self.l_rev, self.n_layers, mask, cache)
            sig_p = sig[0, :, 0].astype(mx.float32)
        logits = self._logits_last(h)
        return logits, cache, tokens, sig_p, diag

    def generate(self, prompt_ids, mode="base", max_new=64, x0_span=None) -> GenResult:
        import mlx.core as mx
        assert mode in ARMS, mode
        t0 = time.perf_counter()
        logits, cache, tokens, sig_p, diag = self.prefill_logits(prompt_ids, mode, x0_span)
        y = mx.argmax(logits, axis=-1)
        mx.eval(y)
        if sig_p is not None:
            mx.eval(sig_p)
        sigma_prompt = [float(v) for v in np.array(sig_p)] if sig_p is not None else []
        gen, sigma_gen, stopped = [], [], False
        for _ in range(max_new):
            tid = int(y.item())
            gen.append(tid)
            if tid in self.eos_ids:
                stopped = True
                break
            if len(gen) == max_new:
                break
            h = self.inner.embed_tokens(y[None])          # (1,1,D)
            if mode == "base":
                h = self._layers(h, 0, self.n_layers, None, cache)
            else:
                h = self._layers(h, 0, self.l_rev, None, cache)
                h, sig = self._inject(h, tokens, mode)
                h = self._layers(h, self.l_rev, self.n_layers, None, cache)
                sigma_gen.append(float(sig[0, 0, 0].item()))
            y = mx.argmax(self._logits_last(h), axis=-1)
            mx.eval(y)
        del cache
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        text_ids = gen[:-1] if stopped else gen
        text = self.hf_tok.decode(text_ids, skip_special_tokens=True)
        return GenResult(gen, text, stopped, sigma_prompt, sigma_gen, diag,
                         time.perf_counter() - t0)

    def parity(self, prompt_ids):
        """Base-arm prefill vs the stock forward at the last prompt position.
        Returns (max_abs, rel, argmax_same): rel = max_abs / max|logits|. The
        KV-cached prefill takes a different Metal kernel path from the cache-free
        stock forward, so on a 4-bit 8B in bf16 max_abs is ~0.1 on logits of
        scale ~40 with identical rankings; the gate is relative + argmax."""
        import mlx.core as mx
        ref = self.model(mx.array([list(prompt_ids)]))[:, -1, :].astype(mx.float32)
        got, *_ = self.prefill_logits(prompt_ids, "base")
        got = got.astype(mx.float32)
        max_abs = float(mx.abs(ref - got).max().item())
        rel = max_abs / max(float(mx.abs(ref).max().item()), 1e-6)
        same = int(ref.argmax(-1).item()) == int(got.argmax(-1).item())
        return max_abs, rel, same


# ----------------------------------------------------------------------------
# gate statistics / aggregation
# ----------------------------------------------------------------------------

def _pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=np.float64), q)) if len(v) else None


def sigma_stats(sigma_prompt: List[float], sigma_gen: List[float]) -> Dict[str, Any]:
    allv = list(sigma_prompt) + list(sigma_gen)
    f = lambda v: (round(float(np.mean(v)), 5) if len(v) else None)  # noqa: E731
    g = lambda v: (round(float(np.max(v)), 5) if len(v) else None)   # noqa: E731
    return {"prompt_mean": f(sigma_prompt), "prompt_max": g(sigma_prompt),
            "gen_mean": f(sigma_gen), "gen_max": g(sigma_gen),
            "all_mean": f(allv), "all_p90": (round(_pct(allv, 90), 5) if allv else None),
            "all_max": g(allv), "n_prompt": len(sigma_prompt), "n_gen": len(sigma_gen)}


def aggregate_arm(rows: List[Dict[str, Any]], open_thresh: float) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ok = [r["ok"] for r in rows]
    parsed = [r["pred"] is not None for r in rows]
    means = [r["sigma"]["all_mean"] for r in rows if r.get("sigma") and r["sigma"]["all_mean"] is not None]
    gens = [r["sigma"]["gen_mean"] for r in rows if r.get("sigma") and r["sigma"]["gen_mean"] is not None]
    pmax = [r["sigma"]["all_max"] for r in rows if r.get("sigma") and r["sigma"]["all_max"] is not None]
    out = {"n": n, "acc": round(sum(ok) / n, 4), "n_correct": int(sum(ok)),
           "parse_rate": round(sum(parsed) / n, 4),
           "mean_n_gen": round(float(np.mean([r["n_gen"] for r in rows])), 1),
           "eos_rate": round(float(np.mean([r["stopped_eos"] for r in rows])), 3),
           "sec_per_q": round(float(np.mean([r["sec"] for r in rows])), 2),
           "tok_per_s": (round(float(sum(r["n_gen"] for r in rows) / max(1e-9, sum(r["sec"] for r in rows))), 2))}
    if means:
        out["sigma"] = {"mean": round(float(np.mean(means)), 5),
                        "p50": round(_pct(means, 50), 5), "p90": round(_pct(means, 90), 5),
                        "max_of_means": round(float(np.max(means)), 5),
                        "gen_mean": (round(float(np.mean(gens)), 5) if gens else None),
                        "mean_of_max": round(float(np.mean(pmax)), 5),
                        "open_rate": round(float(np.mean([m > open_thresh for m in means])), 4),
                        "open_thresh": open_thresh}
    # per-physics MAE when applicable
    errs = [abs(r["pred"] - r["gold"]) for r in rows if r["pred"] is not None and isinstance(r["gold"], float)]
    if errs and rows[0]["dataset"] == "physics":
        out["mae"] = round(float(np.mean(errs)), 4)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on discordant counts (b: A-only, c: B-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    ka = {r["qid"]: r["ok"] for r in rows_a}
    kb = {r["qid"]: r["ok"] for r in rows_b}
    common = sorted(set(ka) & set(kb))
    both = sum(ka[q] and kb[q] for q in common)
    a_only = sum(ka[q] and not kb[q] for q in common)
    b_only = sum(kb[q] and not ka[q] for q in common)
    neither = len(common) - both - a_only - b_only
    return {"n": len(common), "both": both, "a_only": a_only, "b_only": b_only,
            "neither": neither, "delta_acc": (round((sum(kb.values()) - sum(ka.values())) / len(common), 4)
                                              if common else None),
            "mcnemar_p": round(mcnemar_exact(a_only, b_only), 4)}


def summarize(rows: List[Dict[str, Any]], datasets: List[str], arms: List[str],
              open_thresh: float) -> Dict[str, Any]:
    summary, gate_table = {}, []
    for ds in datasets:
        block = {"arms": {}, "paired": {}}
        by_arm = {a: [r for r in rows if r["dataset"] == ds and r["arm"] == a] for a in arms}
        for a in arms:
            block["arms"][a] = aggregate_arm(by_arm[a], open_thresh)
            if "sigma" in block["arms"][a]:
                gate_table.append({"dataset": ds, "arm": a, **block["arms"][a]["sigma"],
                                   "n": block["arms"][a]["n"]})
        for a, b in (("base", "psilm"), ("base", "zeroed"), ("zeroed", "psilm")):
            if a in arms and b in arms and by_arm[a] and by_arm[b]:
                block["paired"][f"{a}_vs_{b}"] = paired(by_arm[a], by_arm[b])
        if ds == "mmlu":
            subj = {}
            for a in arms:
                for r in by_arm[a]:
                    s = r["meta"].get("subject", "?")
                    subj.setdefault(s, {}).setdefault(a, []).append(r["ok"])
            block["by_subject"] = {s: {a: round(sum(v) / len(v), 3) for a, v in d.items()}
                                   for s, d in subj.items()}
        block["n"] = max((len(v) for v in by_arm.values()), default=0)
        summary[ds] = block
    return {"summary": summary, "gate_table": gate_table}


def format_table(summary: Dict[str, Any], arms: List[str]) -> str:
    """One text table: accuracy per arm and gate sigma per (dataset, arm)."""
    lines = []
    hdr = f"{'dataset':9s} {'n':>4s} " + " ".join(f"{'acc:' + a:>11s}" for a in arms)
    hdr += "  | " + " ".join(f"{'sig:' + a:>10s} {'p90':>6s} {'open%':>6s}" for a in arms if a != "base")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for ds, blk in summary.items():
        row = f"{ds:9s} {blk['n']:4d} "
        for a in arms:
            ag = blk["arms"].get(a, {})
            row += f"{(ag.get('acc') if ag.get('n') else None)!s:>11s} "
        row += "  | "
        for a in arms:
            if a == "base":
                continue
            sg = blk["arms"].get(a, {}).get("sigma")
            if sg:
                row += f"{sg['mean']:10.4f} {sg['p90']:6.3f} {100 * sg['open_rate']:6.1f} "
            else:
                row += f"{'-':>10s} {'-':>6s} {'-':>6s} "
        lines.append(row)
    for ds, blk in summary.items():
        for k, pr in blk.get("paired", {}).items():
            lines.append(f"  {ds:8s} {k:16s} n={pr['n']:3d} both={pr['both']:3d} "
                         f"a_only={pr['a_only']:3d} b_only={pr['b_only']:3d} "
                         f"delta={pr['delta_acc']!s:>7s} McNemar p={pr['mcnemar_p']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# small I/O helpers
# ----------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def estimate_budget(tasks: List[Task], arms: List[str], prefill_tps: float = 250.0,
                    decode_tps: float = 18.0) -> Dict[str, Any]:
    """Rough wall-clock estimate for a real run (8B-4bit on Apple silicon,
    numbers are order-of-magnitude placeholders: tune from the first rows)."""
    prompt_tok, gen_tok = 0, 0
    for t in tasks:
        for a in arms:
            p = t.prompt_for(a)
            prompt_tok += len(p.ids)
            gen_tok += p.max_new
    sec = prompt_tok / prefill_tps + gen_tok / decode_tps
    return {"prompt_tokens": prompt_tok, "max_gen_tokens": gen_tok,
            "assumed_prefill_tps": prefill_tps, "assumed_decode_tps": decode_tps,
            "upper_bound_minutes": round(sec / 60, 1)}
