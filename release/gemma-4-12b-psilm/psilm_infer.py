#!/usr/bin/env python3
"""PsiLM on Gemma 4 12B: one command, three numbers.

    python psilm_infer.py                          # 1D Burgers: a=1.28, phi=0.5, x0=0.76
    python psilm_infer.py --a 0.9 --phi 2.1 --x0 0.33
    python psilm_infer.py --question-only          # print the prompt, load nothing
    python psilm_infer.py --no-baseline            # skip the (slow) backbone-alone arm

    python psilm_infer.py --task multimode --modes "1:0.55:0.50,2:0.75:1.10" --x0 0.33
    python psilm_infer.py --task 2d --a 0.6 --cx 0.35 --cy 0.6 --w 0.07 --x0 0.4 --y0 0.55

Loads a frozen 4-bit language model (default ``mlx-community/gemma-4-12B-it-4bit``),
the trained PsiLM bridges for one task (``bridges/<name>/{bridges.safetensors,config.json}``)
and that task's frozen physics model, then answers one field-value question three ways:

  PsiLM     the coupled system: the forward bridge reads the initial condition and
            the queried position out of the prompt's hidden states, the physics model
            evolves the field, and the looked-up value returns to the language model
            as soft tokens through a gated cross-attention. No text crosses the interface.
  backbone  the same language model alone, "Answer: <number>" protocol with answer
            forcing (the baseline arm of the task's eval/mlx_stage2*_eval.py).
  physics   the physics model's own value at the queried point on the TRUE initial
            condition -- the reference the coupled answer should match to +-0.05 --
            plus the spectral solver's ground truth for the same question.

Tasks (``--task``, auto-detected from ``--bridges`` when not given; default 1d):

  1d         u(x,0) = a sin(2 pi x + phi), Burgers to t = 0.5, value at x0.
             physics/fno_burgers_singlemode.safetensors (FNO1d).
  multimode  u(x,0) = sum_m a_m sin(2 pi m x + phi_m), m in {1, 2}, same equation.
             --modes "m:a:phi,m:a:phi" and --x0; physics/fno_burgers_multimode.safetensors.
  2d         a Gaussian bump (height a, center (cx, cy), width w) under Fisher-KPP on the
             periodic unit square to t = 0.4, value at (x0, y0). The physics model is
             DPOT-Tiny (torch, MPS or CPU): physics/dpot_tiny_fisher2d_finetuned.safetensors
             on top of the upstream base checkpoint (--dpot-base, physics/model_Ti.pth,
             downloaded from hzk17/DPOT when absent). Needs the vendored DPOT definition,
             i.e. a repository clone reachable through PSILM_REPO (the pip package does
             not ship vendor/dpot_model.py), plus torch and einops.

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
TASKS = {
    "1d": {"bridges": "gemma-4-12b-4bit-mlx-1d-value-selective",
           "physics": "fno_burgers_singlemode.safetensors",
           "eval": "mlx_stage2_eval.py"},
    "multimode": {"bridges": "gemma-4-12b-4bit-mlx-multimode",
                  "physics": "fno_burgers_multimode.safetensors",
                  "eval": "mlx_stage2b_eval.py"},
    "2d": {"bridges": "gemma-4-12b-4bit-mlx-2d-dpot",
           "physics": "dpot_tiny_fisher2d_finetuned.safetensors",
           "eval": "mlx_stage2d_eval.py"},
}
DEFAULT_BRIDGES = HERE / "bridges" / TASKS["1d"]["bridges"]
DEFAULT_PHYSICS = HERE / "physics" / TASKS["1d"]["physics"]
DEFAULT_DPOT_BASE = HERE / "physics" / "model_Ti.pth"
DPOT_UPSTREAM = ("hzk17/DPOT", "model_Ti.pth")     # the repo psilm/physics/dpot_wrapper.py names
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
# The baseline protocol is eval/mlx_stage2_eval.py's (identical in the multi-mode and
# 2D evaluations, eval/mlx_stage2b_eval.py and eval/mlx_stage2d_eval.py): the question
# plus the "Answer:" nudge, greedy generation, and answer forcing when the reply used
# its budget without an Answer line. When a repository clone is reachable (PSILM_REPO)
# the functions are imported from the task's eval file so the release cannot drift
# from the evaluation; otherwise the verbatim copies below are used.
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


def _baseline_protocol(task="1d"):
    """(chat_generate, parse_value, NUDGE, source): the repository's eval functions
    for this task when a clone is reachable, else the copies in this file."""
    eval_file = TASKS[task]["eval"]
    roots = [Path(PSILM_REPO)] if PSILM_REPO else []
    for root in roots:
        f = root / "eval" / eval_file
        if f.is_file():
            spec = importlib.util.spec_from_file_location("psilm_release_eval_" + task, f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.chat_generate, mod.parse_value, mod.NUDGE, str(f)
    return chat_generate, parse_value, NUDGE, f"vendored copy (eval/{eval_file})"


def parse_psilm_answer(text):
    """The trained reply is 'u at x = {x0} equals {u}.' (2D: 'u at ({x0}, {y0}) equals
    {u}.'); the number after 'equals' is the answer (the evaluation scores only that).
    Returns (value, strict): strict is False when the reply left the template and the
    last decimal number is used."""
    m = re.search(r"equals\s*(-?\d+\.?\d*)", text)
    if m:
        return float(m.group(1)), True
    # no template -> no answer (the evaluation scripts score exactly this way;
    # a reply that stops at the x0 echo must not be scored as x0)
    return None, False


# ------------------------------------------------------------------------ bridges
def _load_checked(bridges, weights):
    """Explicit shape check + missing/unexpected report, then load_weights(strict=False).
    Returns (missing tensor names, number of parameters provided by the file)."""
    keys = set(weights)
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
    n_trained = sum(int(np.prod(v.shape)) for k, v in params.items() if k in keys)
    return missing, n_trained


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
    missing, n_trained = _load_checked(bridges, weights)

    coupling = dict(cfg.get("coupling", {}))
    if args.l_fwd is not None:
        coupling["l_fwd"] = args.l_fwd
    if args.l_rev is not None:
        coupling["l_rev"] = args.l_rev
    if "n_layers" in coupling and int(coupling["n_layers"]) != int(n_layers):
        sys.exit(f"config.json expects a {coupling['n_layers']}-layer backbone; "
                 f"this one has {n_layers}")
    report = {"construct": construct, "missing": missing, "n_tensors": len(keys),
              "n_params": n_trained}
    return bridges, cfg, coupling, report


# The config.json schema of the multi-mode and 2D bridge directories (MANIFEST.md,
# "config.json schema"). Every key is required; the script names the missing one.
CONFIG_REQUIRED = {
    "task": (),
    "construct": ("d_model", "channel", "inj_cap", "gate_bias", "readout_norm"),
    "coupling": ("l_fwd", "l_rev", "n_layers"),
    "physics": ("file",),
}


def read_task_config(bridges_dir, task):
    cfg_path = Path(bridges_dir) / "config.json"
    if not cfg_path.is_file():
        sys.exit(f"{cfg_path}: not found. The {task} task needs a config.json with the keys "
                 f"{list(CONFIG_REQUIRED)} (schema: MANIFEST.md, 'config.json schema')")
    cfg = json.loads(cfg_path.read_text())
    if isinstance(cfg.get("physics"), str):                # shorthand: the file path alone
        cfg["physics"] = {"file": cfg["physics"]}
    missing = []
    for key, subkeys in CONFIG_REQUIRED.items():
        if key not in cfg:
            missing.append(key)
            continue
        if subkeys and not isinstance(cfg[key], dict):
            missing.extend(f"{key}.{s}" for s in subkeys)
            continue
        missing.extend(f"{key}.{s}" for s in subkeys if s not in cfg[key])
    if missing:
        sys.exit(f"{cfg_path}: missing key(s) {missing}. The {task} task's config.json must "
                 "carry 'task', 'construct' {d_model, channel, inj_cap, gate_bias, readout_norm}, "
                 "'coupling' {l_fwd, l_rev, n_layers} and 'physics' {file} -- see MANIFEST.md, "
                 "'config.json schema'")
    if cfg["task"] != task:
        sys.exit(f"{cfg_path}: 'task' is {cfg['task']!r} but --task {task} was requested")
    return cfg


def load_bridges_task(bridges_dir, task, d_model, n_layers, args):
    """Bridges of the multi-mode or 2D task, constructed exactly as the trainers
    (eval/mlx_stage2b_train.py: make_bridges_multi(d_model, gate_bias, inj_cap, channel,
    readout_norm); eval/mlx_stage2d_train.py: PsiBridges2DMLX(d_model, gate_bias, inj_cap,
    channel, readout_norm)) from config.json['construct'], then bridges.safetensors with
    the same shape check + load_weights(strict=False) as the 1D path.
    Returns (bridges, config, coupling, report)."""
    bridges_dir = Path(bridges_dir)
    cfg = read_task_config(bridges_dir, task)
    construct = dict(cfg["construct"])
    for name in ("channel", "gate_bias", "inj_cap", "readout_norm"):   # CLI overrides
        v = getattr(args, name)
        if v is not None:
            construct[name] = v
    if int(construct["d_model"]) != int(d_model):
        sys.exit(f"bridges were trained for hidden size {construct['d_model']} but the "
                 f"backbone has {d_model}: bridges do not transfer between backbones")
    if task == "multimode":
        from psilm.mlx.multimode import make_bridges_multi
        bridges = make_bridges_multi(**construct)
    else:
        from psilm.mlx.bridges2d import PsiBridges2DMLX
        bridges = PsiBridges2DMLX(**construct)
    weights = mx.load(str(bridges_dir / "bridges.safetensors"))
    missing, n_trained = _load_checked(bridges, weights)

    coupling = dict(cfg["coupling"])
    if args.l_fwd is not None:
        coupling["l_fwd"] = args.l_fwd
    if args.l_rev is not None:
        coupling["l_rev"] = args.l_rev
    if int(coupling["n_layers"]) != int(n_layers):
        sys.exit(f"config.json expects a {coupling['n_layers']}-layer backbone; "
                 f"this one has {n_layers}")
    report = {"construct": construct, "missing": missing, "n_tensors": len(weights),
              "n_params": n_trained}
    return bridges, cfg, coupling, report


def load_physics(path):
    path = str(path)
    if path.endswith(".pt"):
        return convert_from_torch(path)
    return load_fno_safetensors(path)


# ------------------------------------------------------------------- 2D physics
def resolve_dpot_base(path):
    """The upstream DPOT-Tiny checkpoint DPOTPhysics loads before the fine-tuned
    weights replace every tensor: the local file when present, else the published
    one from hzk17/DPOT (Hugging Face, Apache-2.0) into the HF cache."""
    p = Path(path)
    if p.is_file():
        return p, "local"
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(DPOT_UPSTREAM[0], DPOT_UPSTREAM[1])
        return Path(f), f"not at {p}; downloaded from {DPOT_UPSTREAM[0]} (HF cache)"
    except Exception as e:                                    # noqa: BLE001
        sys.exit(f"the DPOT-Tiny base checkpoint {p} is missing and the download from "
                 f"{DPOT_UPSTREAM[0]}/{DPOT_UPSTREAM[1]} failed ({e}). Put the file at "
                 f"{DEFAULT_DPOT_BASE} or pass --dpot-base")


def load_physics_2d(ft_path, base_path, device):
    """TorchPhysics2D (psilm/mlx/physics2d.py) with the fine-tuned weights read from a
    safetensors export (or a torch .pt) instead of results/stage2d/dpot_ft.pt."""
    repo = os.environ.get("PSILM_REPO")          # an installed psilm has no vendor/: use the clone's
    if repo and (Path(repo) / "vendor" / "dpot_model.py").is_file():
        sys.path.insert(0, str(Path(repo).resolve() / "vendor"))
    try:
        from psilm.mlx.physics2d import TorchPhysics2D, pick_device
        from psilm.physics.dpot_wrapper import DPOTPhysics
    except ImportError as e:
        sys.exit(f"the 2d task needs torch, einops and the vendored DPOT-Tiny definition "
                 f"(vendor/dpot_model.py in the repository, which the pip package does not "
                 f"ship): {e}\n  Run with PSILM_REPO=/path/to/PsiLM (a clone) and "
                 f"'pip install torch einops'")
    import torch

    class ReleasePhysics2D(TorchPhysics2D):
        def __init__(self, ft, base, dev):
            self.device = pick_device(dev)
            self.phys = DPOTPhysics(ckpt_path=base, device=self.device)
            if str(ft).endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(str(ft))
            else:
                sd = torch.load(ft, map_location="cpu")
            self.phys.net.load_state_dict(sd, strict=True)
            self.phys.eval()
            for p in self.phys.parameters():
                p.requires_grad_(False)
            self.dtype = next(self.phys.parameters()).dtype

    return ReleasePhysics2D(ft_path, base_path, device)


# ------------------------------------------------------------------------ inputs
def parse_modes(text, n_modes):
    """'m:a:phi,m:a:phi' -> ([(m, a, phi), ...] sorted by m, with a and phi rounded to
    two decimals; whether any rounding happened)."""
    modes, seen, rounded = [], set(), False
    for chunk in text.split(","):
        parts = chunk.strip().split(":")
        if len(parts) != 3:
            sys.exit(f"--modes: {chunk.strip()!r} is not an m:a:phi triple")
        try:
            m, a, phi = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            sys.exit(f"--modes: {chunk.strip()!r} is not an m:a:phi triple of numbers")
        if not 1 <= m <= n_modes:
            sys.exit(f"--modes: mode {m} is outside 1..{n_modes} (the bridges read {n_modes} modes)")
        if m in seen:
            sys.exit(f"--modes: mode {m} given twice")
        seen.add(m)
        rounded |= (round(a, 2), round(phi, 2)) != (a, phi)
        modes.append((m, round(a, 2), round(phi, 2)))
    if not modes:
        sys.exit("--modes: no modes given")
    return sorted(modes), rounded


# --------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(
        description="PsiLM (frozen LLM + frozen physics model through latent bridges): "
                    "answer one field-value question three ways.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Training ranges. 1d: a in [0.5, 1.5], phi in [0, 6.28], x0 in [0, 0.99]. "
               "multimode: mode 1 a in [0.3, 0.7], mode 2 a in [0.5, 1.0], phi in [0, 6.28], "
               "x0 in [0, 0.99]. 2d: a in [0.3, 0.9], cx/cy/x0/y0 in [0, 0.99], w in "
               "[0.04, 0.10]. All with two decimals; outside them the bridges are extrapolating.")
    ap.add_argument("--task", choices=list(TASKS), default=None,
                    help="1d (default), multimode, or 2d; auto-detected from --bridges "
                         "(config.json 'task', else the directory name) when omitted")
    ap.add_argument("--a", type=float, default=None,
                    help="1d: amplitude of u(x,0) = a sin(2 pi x + phi) [1.28]; "
                         "2d: height of the Gaussian bump [0.6]")
    ap.add_argument("--phi", type=float, default=0.5, help="1d: phase (radians)")
    ap.add_argument("--x0", type=float, default=None,
                    help="queried position in [0, 1) [1d/multimode 0.76, 2d 0.4]")
    ap.add_argument("--modes", default="1:0.55:2.10",
                    help="multimode: the initial condition as m:a:phi triples, comma-separated "
                         "(u(x,0) = sum a sin(2 pi m x + phi), m in {1, 2}), e.g. "
                         "\"1:0.55:0.50,2:0.75:1.10\"")
    ap.add_argument("--cx", type=float, default=0.35, help="2d: bump center x")
    ap.add_argument("--cy", type=float, default=0.60, help="2d: bump center y")
    ap.add_argument("--w", type=float, default=0.07, help="2d: bump width")
    ap.add_argument("--y0", type=float, default=0.55, help="2d: queried position y in [0, 1)")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    help="mlx-lm checkpoint (Hugging Face id or local path)")
    ap.add_argument("--bridges", default=None,
                    help="directory with bridges.safetensors (+ config.json) "
                         "[bridges/<the task's directory>]")
    ap.add_argument("--physics", default=None,
                    help="1d/multimode: FNO weights (safetensors export, or a PyTorch fno.pt); "
                         "2d: the fine-tuned DPOT-Tiny weights [config.json 'physics.file', "
                         "else physics/<the task's file>]")
    ap.add_argument("--dpot-base", default=str(DEFAULT_DPOT_BASE),
                    help="2d: the upstream DPOT-Tiny checkpoint (model_Ti.pth) the wrapper loads "
                         "first; downloaded from hzk17/DPOT when the file is absent")
    ap.add_argument("--phys-device", default="mps", choices=["mps", "cuda", "cpu"],
                    help="2d: torch device for DPOT-Tiny (falls back to cpu)")
    ap.add_argument("--hf-tokenizer", default=None,
                    help="HF tokenizer id for the chat template (default: config.json's "
                         "hf_tokenizer, else the backbone id)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the backbone-alone arm")
    ap.add_argument("--question-only", action="store_true",
                    help="print the question the models see and exit (loads nothing)")
    ap.add_argument("--max-new", type=int, default=None,
                    help="PsiLM reply budget (tokens) [24; 2d 32]")
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


def detect_task(args):
    """--task, else config.json['task'] of --bridges, else the directory name, else 1d."""
    if args.task:
        return args.task
    if args.bridges:
        cfg_path = Path(args.bridges) / "config.json"
        if cfg_path.is_file():
            t = json.loads(cfg_path.read_text()).get("task")
            if t in TASKS:
                return t
            if t is not None:
                sys.exit(f"{cfg_path}: unknown 'task' {t!r} (expected one of {list(TASKS)})")
        name = Path(args.bridges).name.lower()
        if "multimode" in name:
            return "multimode"
        if "2d" in name or "dpot" in name:
            return "2d"
    return "1d"


def resolve_paths(args, task):
    """Fill --bridges / --physics / --a / --x0 / --max-new with the task's defaults."""
    if args.bridges is None:
        args.bridges = str(HERE / "bridges" / TASKS[task]["bridges"])
    if args.physics is None:
        args.physics = str(HERE / "physics" / TASKS[task]["physics"])
        if task != "1d":                                  # the task config may name the file
            cfg_path = Path(args.bridges) / "config.json"
            if cfg_path.is_file():
                phys = json.loads(cfg_path.read_text()).get("physics")
                f = phys.get("file") if isinstance(phys, dict) else phys
                if isinstance(f, str) and f:
                    args.physics = f if Path(f).is_absolute() else str(HERE / f)
    if args.a is None:
        args.a = 1.28 if task == "1d" else 0.6
    if args.x0 is None:
        args.x0 = 0.4 if task == "2d" else 0.76
    if args.max_new is None:
        args.max_new = 32 if task == "2d" else 24


def main():
    args = build_parser().parse_args()
    task = detect_task(args)
    resolve_paths(args, task)
    if task == "multimode":
        return run_multimode(args)
    if task == "2d":
        return run_2d(args)
    return run_1d(args)


# ------------------------------------------------------------------ task: 1d
def run_1d(args):
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


# --------------------------------------------------------- shared by the new tasks
def _print_bridges(args, rep, cfg, psi, d_model):
    c = rep["construct"]
    print(f"bridges  : {args.bridges}\n"
          f"           channel={c['channel']} readout_norm={c['readout_norm']} "
          f"gate_bias={c['gate_bias']} inj_cap={c['inj_cap']} | "
          f"{rep['n_params']/1e6:.1f}M params in {rep['n_tensors']} tensors"
          + (f" | not in file (unused): {rep['missing']}" if rep["missing"] else ""))
    print(f"coupling : read @ layer {psi.l_fwd}, inject @ layer {psi.l_rev} of {psi.n_layers} "
          f"(hidden {d_model})")


def _run_psilm_arm(psi, builder, item, args):
    t1 = time.time()
    psi_text = psi.generate(builder, item, max_new=args.max_new)
    t_psi = time.time() - t1
    psi_val, strict = parse_psilm_answer(psi_text)
    print(f"[1] PsiLM (coupled)     : {_clean(psi_text)!r}")
    print(f"    value               : {psi_val if psi_val is not None else 'no number parsed'}"
          f"   ({t_psi:.1f} s{'' if strict or psi_val is None else '; reply left the trained template, last number taken'})",
          flush=True)
    return psi_val


def _run_baseline_arm(task, stock, hf_tok, tok, question, args):
    if args.no_baseline:
        print("[2] backbone alone      : skipped (--no-baseline)")
        return None
    gen, parse, nudge, src = _baseline_protocol(task)
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
    return base_val


def _verdict(psi_val, base_val, u_ref, args):
    def mark(v):
        return "n/a" if v is None else ("match" if abs(v - u_ref) <= TOL else "off")
    if psi_val is None:
        verdict = "PsiLM n/a (no number in the reply)"
    else:
        verdict = (f"PsiLM {mark(psi_val)} (|{psi_val}-{u_ref:.2f}| "
                   f"{'<=' if abs(psi_val-u_ref) <= TOL else '>'} {TOL})")
    print("\n" + verdict + ("" if args.no_baseline else f"; backbone alone {mark(base_val)}"))


# ------------------------------------------------------------ task: multimode
def run_multimode(args):
    from psilm.mlx.multimode import N_MODES, PsiLMMLXMulti, build_ic_multi_mlx
    from psilm.physics.burgers import initial_condition_multi
    from psilm.stage2.qa2 import AMP, QA2Builder, ic_text
    from psilm.stage2.qa2 import QUESTION as QUESTION2

    modes, rounded = parse_modes(args.modes, N_MODES)
    x0 = round(args.x0, 2)
    if rounded or x0 != args.x0:
        print(f"note: inputs rounded to two decimals (the trained readout reads 2-decimal "
              f"numbers): modes={modes} x0={x0}")
    if not (0.0 <= x0 < 1.0):
        sys.exit("x0 must lie in [0, 1): the domain is periodic")
    for m, a, phi in modes:
        lo, hi = AMP[m]
        if not (lo <= a <= hi and 0.0 <= phi <= 6.28):
            print(f"warning: mode {m} lies outside the training ranges (a in [{lo}, {hi}], "
                  f"phi in [0, 6.28]); the bridges are extrapolating")
    if len(modes) > 1:
        print("note: two-mode initial conditions are the held-out combination family "
              "(the bridges were trained on single-mode questions)")
    # the item as data/stage2b_qa_train.json stores it (modes as [m, a, phi] lists, x0)
    item = {"modes": [[m, a, phi] for m, a, phi in modes], "x0": x0}
    question = QUESTION2.format(ic=ic_text(item["modes"]), x0=item["x0"])

    if args.question_only:
        print(f"[system] {SYSTEM}")
        print(f"[user]   {question}")
        print(f"\n(the backbone-alone arm appends: {NUDGE.strip()!r})")
        return

    # ---- load
    t0 = time.time()
    print(f"task     : multimode (1D Burgers, multi-mode initial condition)")
    print(f"backbone : {args.backbone}", flush=True)
    model, stock, tok = load_backbone_any(args.backbone)
    n_layers = len(model.model.layers)
    d_model = int(model.args.hidden_size)
    bridges, cfg, coupling, rep = load_bridges_task(args.bridges, "multimode", d_model, n_layers, args)
    hf_id = args.hf_tokenizer or cfg.get("hf_tokenizer") or args.backbone
    hf_tok = AutoTokenizer.from_pretrained(hf_id)
    fno = load_physics(args.physics)
    psi = PsiLMMLXMulti(model, tok, fno, bridges, l_fwd=coupling["l_fwd"], l_rev=coupling["l_rev"])
    builder = QA2Builder(hf_tok)
    t_load = time.time() - t0
    _print_bridges(args, rep, cfg, psi, d_model)
    n_fno = sum(int(np.prod(v.shape)) for k, v in tree_flatten(fno.parameters()) if not k.endswith(".wi"))
    print(f"physics  : {args.physics} (FNO1d, multi-mode, {n_fno/1e3:.0f}K params)")
    print(f"loaded in {t_load:.1f} s\n")
    print(f"question : {question}\n", flush=True)

    # ---- 1. PsiLM: the coupled system
    psi_val = _run_psilm_arm(psi, builder, item, args)

    # ---- 2. backbone alone
    base_val = _run_baseline_arm("multimode", stock, hf_tok, tok, question, args)

    # ---- 3. the physics model on the true initial condition
    t3 = time.time()
    p = [0.0] * (3 * N_MODES)                                   # (a, sin phi, cos phi) per mode
    for m, a, phi in modes:
        p[3 * m - 3: 3 * m] = [a, math.sin(phi), math.cos(phi)]
    ic = build_ic_multi_mlx(mx.array([p], dtype=mx.float32))    # the bridges' IC parameterization
    u_field = np.array(fno(ic), dtype=np.float64)[0]            # u(x, t=0.5) on the 128-grid
    u_fno = fourier_interp(u_field, x0)
    t_fno = time.time() - t3
    u_true = fourier_interp(solve(initial_condition_multi(modes)), x0)
    if args.verbose:
        ic_np = initial_condition_multi(modes)
        print(f"    (IC parameterization check: max|build_ic_multi - initial_condition_multi| = "
              f"{float(np.abs(np.array(ic)[0] - ic_np).max()):.2e})")
    print(f"[3] physics model (FNO) : u({x0}) = {u_fno:+.4f}   ({t_fno*1e3:.0f} ms; "
          f"the reference the coupled answer should match to +-{TOL})")
    print(f"    spectral solver     : u({x0}) = {u_true:+.4f}   (ground truth)")
    _verdict(psi_val, base_val, u_fno, args)


# ------------------------------------------------------------------- task: 2d
def run_2d(args):
    from psilm.stage2d.qa2d import QUANTITIES, QA2DBuilder
    from psilm.stage2d.qa2d import QUESTION as QUESTION2D

    raw = {"a": args.a, "cx": args.cx, "cy": args.cy, "w": args.w, "x0": args.x0, "y0": args.y0}
    item = {k: round(v, 2) for k, v in raw.items()}            # as data/stage2d_qa_train.json
    if any(item[k] != raw[k] for k in item):
        print("note: inputs rounded to two decimals (the trained readout reads 2-decimal "
              "numbers): " + " ".join(f"{k}={item[k]}" for k in QUANTITIES))
    for k in ("cx", "cy", "x0", "y0"):
        if not (0.0 <= item[k] < 1.0):
            sys.exit(f"{k} must lie in [0, 1): the unit square is periodic")
    if not (0.3 <= item["a"] <= 0.9 and 0.04 <= item["w"] <= 0.10):
        print("warning: a or w lies outside the training ranges (a in [0.3, 0.9], "
              "w in [0.04, 0.10]); the bridges are extrapolating")
    question = QUESTION2D.format(**{k: item[k] for k in QUANTITIES})

    if args.question_only:
        print(f"[system] {SYSTEM}")
        print(f"[user]   {question}")
        print(f"\n(the backbone-alone arm appends: {NUDGE.strip()!r})")
        return

    from psilm.physics.fisher2d import bilinear_periodic
    from psilm.physics.fisher2d import initial_condition as initial_condition_2d
    from psilm.physics.fisher2d import solve as solve_2d
    from psilm.mlx.model2d import PsiLM2DMLX

    # ---- load
    t0 = time.time()
    print("task     : 2d (2D Fisher-KPP, DPOT-Tiny)")
    print(f"backbone : {args.backbone}", flush=True)
    model, stock, tok = load_backbone_any(args.backbone)
    n_layers = len(model.model.layers)
    d_model = int(model.args.hidden_size)
    bridges, cfg, coupling, rep = load_bridges_task(args.bridges, "2d", d_model, n_layers, args)
    hf_id = args.hf_tokenizer or cfg.get("hf_tokenizer") or args.backbone
    hf_tok = AutoTokenizer.from_pretrained(hf_id)
    base_path, base_src = resolve_dpot_base(args.dpot_base)
    phys = load_physics_2d(args.physics, base_path, args.phys_device)
    psi = PsiLM2DMLX(model, tok, phys, bridges, l_fwd=coupling["l_fwd"], l_rev=coupling["l_rev"])
    builder = QA2DBuilder(hf_tok)
    t_load = time.time() - t0
    _print_bridges(args, rep, cfg, psi, d_model)
    n_dpot = sum(int(p.numel()) for p in phys.phys.parameters())
    print(f"physics  : {args.physics} (DPOT-Tiny, {n_dpot/1e6:.1f}M params, torch on {phys.device})\n"
          f"           base checkpoint {base_path} ({base_src})")
    print(f"loaded in {t_load:.1f} s\n")
    print(f"question : {question}\n", flush=True)

    # ---- 1. PsiLM: the coupled system
    psi_val = _run_psilm_arm(psi, builder, item, args)

    # ---- 2. backbone alone
    base_val = _run_baseline_arm("2d", stock, hf_tok, tok, question, args)

    # ---- 3. the physics model on the true initial condition
    t3 = time.time()
    params = np.array([[item["a"], item["cx"], item["cy"], item["w"]]], dtype=np.float32)
    field = np.array(phys(params), dtype=np.float64)[0]         # u(x, y, t=0.4) on the 128x128 grid
    u_dpot = bilinear_periodic(field, item["x0"], item["y0"])
    t_dpot = time.time() - t3
    u_true = bilinear_periodic(solve_2d(initial_condition_2d(item["a"], item["cx"], item["cy"], item["w"])),
                               item["x0"], item["y0"])
    if args.verbose:
        from psilm.stage2d.bridges2d import build_ic_2d
        import torch
        t_ = lambda k: torch.tensor([item[k]], dtype=torch.float32)   # noqa: E731
        ic = build_ic_2d(t_("a"), t_("cx"), t_("cy"), t_("w"))[0].numpy()
        ic_np = initial_condition_2d(item["a"], item["cx"], item["cy"], item["w"])
        print(f"    (IC parameterization check: max|build_ic_2d - initial_condition| = "
              f"{float(np.abs(ic - ic_np).max()):.2e})")
    pt = f"({item['x0']}, {item['y0']})"
    print(f"[3] physics model (DPOT): u{pt} = {u_dpot:+.4f}   ({t_dpot*1e3:.0f} ms; "
          f"the reference the coupled answer should match to +-{TOL})")
    print(f"    spectral solver     : u{pt} = {u_true:+.4f}   (ground truth)")
    _verdict(psi_val, base_val, u_dpot, args)


if __name__ == "__main__":
    main()
