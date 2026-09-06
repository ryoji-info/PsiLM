#!/usr/bin/env python3
"""Export a trained bridge checkpoint to the Hugging Face layout.

Reads ``<run>/bridges.npz`` (+ ``bridges.npz.meta``) and writes
``<out>/bridges.safetensors`` and ``<out>/config.json`` in the schema the
release inference script expects (see release/gemma-4-12b-psilm/MANIFEST.md).

The deterministic span pointer makes the learned-pointer tensors
(``fwd.x0_query``/``fwd.x0_key.*``) dead weight; they are dropped unless
``--keep-unused`` is given, and the pointer note in config.json records it.

Per-chunk held-out scores are recovered from the run's supervisor.log
("CHUNK DONE step=N acc@0.05=..."), split into the coupled and no-harm phases
by which steps have a ``bridges_step<N>_noharm.npz`` checkpoint beside them.

  python eval/export_bridges.py --run results/stage2b_gemma12b_2b \
      --out results/hf_export/bridges/gemma-4-12b-4bit-mlx-multimode-value-selective
"""
import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

# Layer counts of the backbones used so far; --n-layers overrides.
N_LAYERS = {
    "mlx-community/gemma-4-12B-it-4bit": 48,
    "mlx-community/Qwen3-8B-4bit": 36,
    "mlx-community/Qwen3-1.7B-4bit": 28,
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit": 24,
}

TASKS = {
    "stage2": {
        "task": "1d",
        "physics": "results/stage2/fno.pt (1D Burgers FNO, 70K params)",
        "physics_file": "fno_burgers_singlemode.safetensors",
    },
    "stage2b": {
        "task": "multimode",
        "physics": "results/stage2b/fno.pt (1D Burgers FNO, multi-mode initial conditions)",
        "physics_file": "fno_burgers_multimode.safetensors",
    },
    "stage2d": {
        "task": "2d",
        "physics": "results/stage2d/dpot_ft.pt (DPOT-Tiny fine-tuned on 2D Fisher-KPP)",
        "physics_file": "dpot_tiny_fisher2d_finetuned.safetensors",
    },
}

DROP = ("fwd.x0_query", "fwd.x0_key.weight", "fwd.x0_key.bias")


def phase_steps(run: Path):
    """(n_noharm_chunks, last coupled step) from the checkpoints kept in the run.

    A no-harm phase resumed from an earlier checkpoint replays step numbers, so
    the phases are told apart by the ``_noharm`` suffix, not by step order.
    """
    noharm = list(run.glob("bridges_step*_noharm.npz"))
    coupled = [p for p in run.glob("bridges_step*.npz") if "_noharm" not in p.name]
    last = max((int(re.search(r"step(\d+)", p.name).group(1)) for p in coupled),
               default=0)
    return len(noharm), last


def chunk_scores(run: Path, n_noharm: int):
    """(coupled, no_harm) held-out accuracy per chunk, from supervisor.log.

    The no-harm phase is the last ``n_noharm`` scored chunks; a replayed step
    number therefore stays with the phase that produced it.
    """
    log = run / "supervisor.log"
    if not log.exists():
        return [], []
    accs = []
    for line in log.read_text().splitlines():
        m = re.match(r"CHUNK DONE step=(\d+) acc@0\.05=([\d.]+|None)", line.strip())
        if m and m.group(2) != "None":
            accs.append(float(m.group(2)))
    if n_noharm == 0:
        return accs, []
    return accs[:-n_noharm], accs[-n_noharm:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run directory")
    ap.add_argument("--out", required=True, help="export directory")
    ap.add_argument("--ckpt", default="bridges.npz", help="checkpoint inside --run")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--keep-unused", action="store_true",
                    help="keep the retired learned-pointer tensors")
    ap.add_argument("--license", default="apache-2.0")
    ap.add_argument("--note", default=None, help="extra free-text note in config.json")
    args = ap.parse_args()

    run, out = Path(args.run), Path(args.out)
    meta = json.loads((run / (args.ckpt + ".meta")).read_text())
    z = np.load(run / args.ckpt)
    weights = {k: mx.array(z[k]) for k in z.files
               if args.keep_unused or k not in DROP}

    a = meta.get("args", {})
    model = meta["model"]
    task = TASKS[meta.get("task", "stage2")]
    n_layers = args.n_layers or N_LAYERS.get(model)
    if n_layers is None:
        raise SystemExit(f"unknown layer count for {model}; pass --n-layers")
    d_model = int(z["fwd.key.weight"].shape[1])
    readout_norm = a.get("readout_norm", "rms")
    n_noharm, last_coupled = phase_steps(run)
    coupled, noharm = chunk_scores(run, n_noharm)

    steps_total = int(meta["step"])
    warmup = int(a.get("readout_only", 0))
    per_chunk = int(a.get("steps", 500))
    noharm_steps = n_noharm * per_chunk
    coupled_steps = (last_coupled or steps_total - noharm_steps) - warmup
    cfg = {
        "backbone": model,
        "hf_tokenizer": a.get("hf_tokenizer", model),
        "task": task["task"],
        "loader": ("psilm.mlx.gemma_loader.load_backbone_any"
                   + (" (gemma4_unified -> text tower)" if "gemma" in model.lower()
                      else "")),
        "physics": task["physics"],
        "physics_file": task["physics_file"],
        "bridges_class": "psilm.mlx.bridges.PsiBridgesMLX",
        "construct": {
            "d_model": d_model,
            "channel": a.get("channel", "value"),
            "inj_cap": a.get("inj_cap"),
            "gate_bias": a.get("gate_bias", 0.0),
            "readout_norm": readout_norm,
        },
        "coupling": {
            "l_fwd": meta["l_fwd"],
            "l_rev": meta["l_rev"],
            "n_layers": n_layers,
        },
        "pointer": (
            "deterministic span pooling"
            + (" with calibrated per-dimension standardization "
               "(fwd.dim_mu/dim_sigma included" if readout_norm == "dim" else " (")
            + ("; fwd.x0_query/x0_key kept but unused)" if args.keep_unused
               else "; fwd.x0_query/x0_key omitted: unused)")
        ),
        "training": {
            "steps_total": steps_total,
            "phase_A_readout_only": warmup,
            "coupled_steps": coupled_steps,
            "no_harm_steps": noharm_steps,
            "no_harm_resumed_from": steps_total - noharm_steps,
            "batch": a.get("batch"),
            "lr": "3e-4 (warm-up, coupled), 1e-4 (no-harm phase)",
            "no_harm_arm": (
                f"non-physics prompts ({a.get('noharm_data')}) paired with the "
                "backbone's own greedy continuation; gate-only updates + "
                "mean-gate penalty" if a.get("noharm_data") else "not run"),
        },
        f"held_out_n{a.get('eval_n', 48)}_per_chunk": {
            "coupled": coupled,
            "no_harm_phase": noharm,
        },
        "license": args.license,
    }
    if args.note:
        cfg["note"] = args.note

    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "bridges.safetensors"), weights)
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"{len(weights)} tensors -> {out/'bridges.safetensors'}")
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
