#!/usr/bin/env python3
"""PsiLM on Gemma 4 12B: one command, three numbers.

    python psilm_infer.py                          # a=1.28, phi=0.5, x0=0.76
    python psilm_infer.py --a 0.9 --phi 2.1 --x0 0.33
    python psilm_infer.py --question-only          # print the prompt, load nothing
    python psilm_infer.py --no-baseline            # skip the (slow) backbone-alone arm

Loads a frozen 4-bit language model (default ``mlx-community/gemma-4-12B-it-4bit``),
the trained PsiLM bridges for it (``bridges/<name>/{bridges.safetensors,config.json}``)
and the frozen 1D Burgers FNO (``physics/fno_burgers_singlemode.safetensors``), then
answers one Burgers field-value question three ways:

  PsiLM     the coupled system: the forward bridge reads the initial condition and
            the queried position x0 out of the prompt's hidden states, the FNO evolves
            the field, and the looked-up value u(x0) returns to the language model as
            soft tokens through a gated cross-attention. No text crosses the interface.
  backbone  the same language model alone, "Answer: <number>" protocol with answer
            forcing (the baseline arm of eval/mlx_stage2_eval.py in the repository).
  physics   the FNO's own value at x0 on the TRUE initial condition -- the reference
            the coupled answer should match to +-0.05 -- plus the spectral solver's
            ground truth for the same question.

Dependencies: ``pip install -r requirements.txt`` (mlx, mlx-lm, transformers, torch,
huggingface_hub, and the ``psilm`` package from https://github.com/ryoji-info/PsiLM).
If ``psilm`` is not installed, set ``PSILM_REPO=/path/to/PsiLM`` (a clone) instead.
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BACKBONE = "mlx-community/gemma-4-12B-it-4bit"
DEFAULT_BRIDGES = HERE / "bridges" / "gemma-4-12b-4bit-mlx-1d-value-selective"
DEFAULT_PHYSICS = HERE / "physics" / "fno_burgers_singlemode.safetensors"
TOL = 0.05


# --------------------------------------------------------------------------- imports
def _psilm_available():
    try:
        return importlib.util.find_spec("psilm.mlx") is not None
    except ModuleNotFoundError:
        return False


def _ensure_psilm():
    """Import path for the ``psilm`` package: the installed package first, else the
    optional ``PSILM_REPO`` environment variable (a clone of the GitHub repository)."""
    if _psilm_available():
        return None
    repo = os.environ.get("PSILM_REPO")
    if repo and (Path(repo) / "psilm" / "mlx").is_dir():
        sys.path.insert(0, str(Path(repo).resolve()))
        for name in [m for m in sys.modules if m == "psilm" or m.startswith("psilm.")]:
            del sys.modules[name]                 # a partial install may already be imported
        importlib.invalidate_caches()
        if _psilm_available():
            return str(Path(repo).resolve())
    sys.exit(
        "psilm_infer.py: the 'psilm' package (with its psilm.mlx subpackage) is not importable.\n"
        "  Install it:   pip install -r requirements.txt\n"
        "                (or: pip install 'git+https://github.com/ryoji-info/PsiLM')\n"
        "  Or point at a clone:  PSILM_REPO=/path/to/PsiLM python psilm_infer.py ..."
    )


PSILM_REPO = _ensure_psilm()

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
import mlx_lm  # noqa: E402
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from psilm.mlx.bridges import PsiBridgesMLX, build_ic_mlx  # noqa: E402
from psilm.mlx.fno import convert_from_torch, load_fno_safetensors  # noqa: E402
from psilm.mlx.gemma_loader import load_backbone_any  # noqa: E402
from psilm.mlx.model import PsiLMMLX  # noqa: E402
from psilm.physics.burgers import initial_condition, solve  # noqa: E402
from psilm.stage2.qa import QUESTION, SYSTEM, QABuilder, fourier_interp  # noqa: E402


# --------------------------------------------------------------- backbone-alone arm
# The baseline protocol is eval/mlx_stage2_eval.py's: the question plus the "Answer:"
# nudge, greedy generation, and answer forcing when the reply used its budget without
# an Answer line. When a repository clone is reachable (PSILM_REPO) the functions are
# imported from that file so the release cannot drift from the evaluation; otherwise
# the verbatim copies below are used.
NUDGE = "\nEnd your reply with a line of the form \"Answer: <number>\"."
FORCE_SUFFIX = "\n\nAnswer:"


def parse_value(text):
    m = re.findall(r"Answer:\s*\$?\\?\(?\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m[-1])
    m = re.findall(r"-?\d+\.\d+", text)
    return float(m[-1]) if m else None


def chat_generate(model, hf_tok, user, max_new=768, gen_tok=None, force_answer=True):
    """Returns (text, forced). gen_tok is the mlx-lm tokenizer wrapper (knows all of
    a backbone's stop ids, e.g. Gemma's <eos>/<turn|>); hf_tok builds the prompt."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = hf_tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    text = mlx_lm.generate(model, gen_tok or hf_tok, prompt=list(ids), max_tokens=max_new, verbose=False)
    if force_answer and "Answer:" not in text:
        cont = list(ids) + hf_tok.encode(text + FORCE_SUFFIX, add_special_tokens=False)
        tail = mlx_lm.generate(model, gen_tok or hf_tok, prompt=cont, max_tokens=16, verbose=False)
        return text + FORCE_SUFFIX + tail, True
    return text, False


def _baseline_protocol():
    """(chat_generate, parse_value, NUDGE, source): the repository's eval functions
    when a clone is reachable, else the copies in this file."""
    roots = [Path(PSILM_REPO)] if PSILM_REPO else []
    for root in roots:
        f = root / "eval" / "mlx_stage2_eval.py"
        if f.is_file():
            spec = importlib.util.spec_from_file_location("psilm_release_eval_stage2", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.chat_generate, mod.parse_value, mod.NUDGE, str(f)
    return chat_generate, parse_value, NUDGE, "vendored copy (eval/mlx_stage2_eval.py)"


def parse_psilm_answer(text):
    """The trained reply is 'u at x = {x0} equals {u}.'; the number after 'equals' is
    the answer (the evaluation scores only that). Returns (value, strict): strict is
    False when the reply left the template and the last decimal number is used."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m.group(1)), True
    m = re.findall(r"-?\d+\.\d+", text)
    return (float(m[-1]), False) if m else (None, False)


# ------------------------------------------------------------------------ bridges
def load_bridges(bridges_dir, d_model, n_layers, args):
    """PsiBridgesMLX from <dir>/config.json['construct'] (defaults when the file is
    absent, inferred from the tensors present) + <dir>/bridges.safetensors with
    load_weights(strict=False): the Gemma export omits the retired learned-pointer
    tensors (fwd.x0_query, fwd.x0_key.*), which the deterministic span pointer never
    uses. Shapes of every provided tensor are checked explicitly, since non-strict
    loading would not. Returns (bridges, config, coupling, report)."""
    bridges_dir = Path(bridges_dir)
    cfg_path = bridges_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    weights = mx.load(str(bridges_dir / "bridges.safetensors"))
    keys = set(weights)

    construct = {
        "d_model": int(weights["fwd.query"].shape[0]) if "fwd.query" in keys else d_model,
        "channel": "value" if any(k.startswith("val.") for k in keys) else "field",
        "gate_bias": -2.0,
        "inj_cap": None,
        "readout_norm": "dim" if "fwd.dim_mu" in keys else "rms",
    }
    construct.update(cfg.get("construct", {}))
    for name in ("channel", "gate_bias", "inj_cap", "readout_norm"):   # CLI overrides
        v = getattr(args, name)
        if v is not None:
            construct[name] = v
    if int(construct["d_model"]) != int(d_model):
        sys.exit(f"bridges were trained for hidden size {construct['d_model']} but the "
                 f"backbone has {d_model}: bridges do not transfer between backbones")
    bridges = PsiBridgesMLX(**construct)

    # explicit shape check + missing/unexpected report
    params = dict(tree_flatten(bridges.parameters()))
    missing = sorted(k for k in params if k not in keys)
    unexpected = sorted(k for k in keys if k not in params)
    bad = [(k, tuple(weights[k].shape), tuple(params[k].shape))
           for k in keys if k in params and tuple(weights[k].shape) != tuple(params[k].shape)]
    if bad:
        lines = "\n".join(f"  {k}: file {a} vs module {b}" for k, a, b in bad)
        sys.exit(f"bridge tensor shapes do not match the constructed bridges:\n{lines}")
    if unexpected:
        sys.exit(f"bridges.safetensors has tensors the bridges do not define: {unexpected}")
    bridges.load_weights(list(weights.items()), strict=False)
    mx.eval(bridges.parameters())

    coupling = dict(cfg.get("coupling", {}))
    if args.l_fwd is not None:
        coupling["l_fwd"] = args.l_fwd
    if args.l_rev is not None:
        coupling["l_rev"] = args.l_rev
    if "n_layers" in coupling and int(coupling["n_layers"]) != int(n_layers):
        sys.exit(f"config.json expects a {coupling['n_layers']}-layer backbone; "
                 f"this one has {n_layers}")
    n_trained = sum(int(np.prod(v.shape)) for k, v in params.items() if k in keys)
    report = {"construct": construct, "missing": missing, "n_tensors": len(keys),
              "n_params": n_trained}
    return bridges, cfg, coupling, report


def load_physics(path):
    path = str(path)
    if path.endswith(".pt"):
        return convert_from_torch(path)
    return load_fno_safetensors(path)


# --------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(
        description="PsiLM (frozen LLM + frozen Burgers FNO through latent bridges): "
                    "answer one field-value question three ways.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Training ranges: a in [0.5, 1.5], phi in [0, 6.28], x0 in [0, 0.99], "
               "all with two decimals. Outside them the bridges are extrapolating.")
    ap.add_argument("--a", type=float, default=1.28, help="amplitude of u(x,0) = a sin(2 pi x + phi)")
    ap.add_argument("--phi", type=float, default=0.5, help="phase (radians)")
    ap.add_argument("--x0", type=float, default=0.76, help="queried position in [0, 1)")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    help="mlx-lm checkpoint (Hugging Face id or local path)")
    ap.add_argument("--bridges", default=str(DEFAULT_BRIDGES),
                    help="directory with bridges.safetensors (+ config.json)")
    ap.add_argument("--physics", default=str(DEFAULT_PHYSICS),
                    help="FNO weights: safetensors export, or a PyTorch fno.pt")
    ap.add_argument("--hf-tokenizer", default=None,
                    help="HF tokenizer id for the chat template (default: config.json's "
                         "hf_tokenizer, else the backbone id)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the backbone-alone arm")
    ap.add_argument("--question-only", action="store_true",
                    help="print the question the models see and exit (loads nothing)")
    ap.add_argument("--max-new", type=int, default=24, help="PsiLM reply budget (tokens)")
    ap.add_argument("--baseline-max-new", type=int, default=768,
                    help="backbone-alone reply budget before answer forcing (the "
                         "evaluation used 768; Gemma 4 uses all of it, ~75 s)")
    g = ap.add_argument_group("bridge construction overrides (default: config.json)")
    g.add_argument("--channel", choices=["value", "field"], default=None)
    g.add_argument("--gate-bias", type=float, default=None)
    g.add_argument("--inj-cap", type=float, default=None)
    g.add_argument("--readout-norm", choices=["rms", "dim"], default=None)
    g.add_argument("--l-fwd", type=int, default=None, help="readout layer")
    g.add_argument("--l-rev", type=int, default=None, help="injection layer")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def _clean(text):
    return re.sub(r"<\|?[a-z_|]+\|?>", "", text).strip()


def main():
    args = build_parser().parse_args()
    item = {"a": round(args.a, 2), "phi": round(args.phi, 2), "x0": round(args.x0, 2)}
    if (item["a"], item["phi"], item["x0"]) != (args.a, args.phi, args.x0):
        print(f"note: inputs rounded to two decimals (the trained readout reads 2-decimal "
              f"numbers): a={item['a']} phi={item['phi']} x0={item['x0']}")
    if not (0.0 <= item["x0"] < 1.0):
        sys.exit("x0 must lie in [0, 1): the domain is periodic")
    if not (0.5 <= item["a"] <= 1.5 and 0.0 <= item["phi"] <= 6.28):
        print("warning: a or phi lies outside the training ranges (a in [0.5, 1.5], "
              "phi in [0, 6.28]); the bridges are extrapolating")
    question = QUESTION.format(a=item["a"], phi=item["phi"], x0=item["x0"])

    if args.question_only:
        print(f"[system] {SYSTEM}")
        print(f"[user]   {question}")
        print(f"\n(the backbone-alone arm appends: {NUDGE.strip()!r})")
        return

    # ---- load
    t0 = time.time()
    print(f"backbone : {args.backbone}", flush=True)
    model, stock, tok = load_backbone_any(args.backbone)
    n_layers = len(model.model.layers)
    d_model = int(model.args.hidden_size)
    bridges, cfg, coupling, rep = load_bridges(args.bridges, d_model, n_layers, args)
    hf_id = args.hf_tokenizer or cfg.get("hf_tokenizer") or args.backbone
    hf_tok = AutoTokenizer.from_pretrained(hf_id)
    fno = load_physics(args.physics)
    psi = PsiLMMLX(model, tok, fno, bridges, l_fwd=coupling.get("l_fwd"), l_rev=coupling.get("l_rev"))
    builder = QABuilder(hf_tok)
    t_load = time.time() - t0
    c = rep["construct"]
    print(f"bridges  : {args.bridges}\n"
          f"           channel={c['channel']} readout_norm={c['readout_norm']} "
          f"gate_bias={c['gate_bias']} inj_cap={c['inj_cap']} | "
          f"{rep['n_params']/1e6:.1f}M params in {rep['n_tensors']} tensors"
          + (f" | not in file (unused): {rep['missing']}" if rep["missing"] else ""))
    print(f"coupling : read @ layer {psi.l_fwd}, inject @ layer {psi.l_rev} of {psi.n_layers} "
          f"(hidden {d_model}"
          + (")" if cfg.get("coupling") else "; no config.json coupling entry: PsiLMMLX defaults)"))
    # parameter count with each complex spectral weight counted once (wr/wi are one number)
    n_fno = sum(int(np.prod(v.shape)) for k, v in tree_flatten(fno.parameters()) if not k.endswith(".wi"))
    print(f"physics  : {args.physics} (FNO1d, {n_fno/1e3:.0f}K params)")
    print(f"loaded in {t_load:.1f} s\n")
    print(f"question : {question}\n", flush=True)

    # ---- 1. PsiLM: the coupled system
    t1 = time.time()
    psi_text = psi.generate(builder, item, max_new=args.max_new)
    t_psi = time.time() - t1
    psi_val, strict = parse_psilm_answer(psi_text)
    print(f"[1] PsiLM (coupled)     : {_clean(psi_text)!r}")
    print(f"    value               : {psi_val if psi_val is not None else 'no number parsed'}"
          f"   ({t_psi:.1f} s{'' if strict or psi_val is None else '; reply left the trained template, last number taken'})",
          flush=True)

    # ---- 2. backbone alone
    base_val = None
    if not args.no_baseline:
        gen, parse, nudge, src = _baseline_protocol()
        t2 = time.time()
        base_text, forced = gen(stock, hf_tok, question + nudge, args.baseline_max_new,
                                gen_tok=tok, force_answer=True)
        t_base = time.time() - t2
        base_val = parse(base_text)
        tail = _clean(base_text)
        tail = tail if args.verbose or len(tail) <= 240 else "..." + tail[-240:]
        print(f"[2] backbone alone      : {tail!r}")
        print(f"    value               : {base_val if base_val is not None else 'no number parsed'}"
              f"   ({t_base:.1f} s, {len(tok.encode(base_text))} tokens"
              f"{', answer forced' if forced else ''}; protocol: {src})", flush=True)
    else:
        print("[2] backbone alone      : skipped (--no-baseline)")

    # ---- 3. the physics model on the true initial condition
    t3 = time.time()
    params = mx.array([[item["a"], math.sin(item["phi"]), math.cos(item["phi"])]], dtype=mx.float32)
    ic = build_ic_mlx(params)                                   # the bridges' IC parameterization
    u_field = np.array(fno(ic), dtype=np.float64)[0]            # u(x, t=0.5) on the 128-grid
    u_fno = fourier_interp(u_field, item["x0"])
    t_fno = time.time() - t3
    u_true = fourier_interp(solve(initial_condition(item["a"], item["phi"])), item["x0"])
    if args.verbose:
        ic_np = initial_condition(item["a"], item["phi"])
        print(f"    (IC parameterization check: max|build_ic - initial_condition| = "
              f"{float(np.abs(np.array(ic)[0] - ic_np).max()):.2e})")
    print(f"[3] physics model (FNO) : u({item['x0']}) = {u_fno:+.4f}   ({t_fno*1e3:.0f} ms; "
          f"the reference the coupled answer should match to +-{TOL})")
    print(f"    spectral solver     : u({item['x0']}) = {u_true:+.4f}   (ground truth)")

    # ---- verdict
    def mark(v):
        return "n/a" if v is None else ("match" if abs(v - u_fno) <= TOL else "off")
    if psi_val is None:
        verdict = "PsiLM n/a (no number in the reply)"
    else:
        verdict = (f"PsiLM {mark(psi_val)} (|{psi_val}-{u_fno:.2f}| "
                   f"{'<=' if abs(psi_val-u_fno) <= TOL else '>'} {TOL})")
    print("\n" + verdict + ("" if args.no_baseline else f"; backbone alone {mark(base_val)}"))


if __name__ == "__main__":
    main()
