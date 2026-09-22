#!/usr/bin/env python3
"""Rescore every recorded KL in results/bench under both reads of the coupling.

Until 2026-09-22 the teacher-forced KL pass let the coupling read the prompt AND
the base continuation, where every arm's decode reads the prompt alone
(eval/bench_common.py, StagedDecoder._hidden_full). The fix was held back until
the last queued arm had landed, so that every recorded KL shares one basis. This
drives `bench_guardrail.py --kl-rescore` over each finished run, rebuilding its
command line from the config its own summary recorded, so nothing about a run is
retyped here: same checkpoint, partner, task cache, budgets and arms; no
generation; the run's own base continuations.

The runs are found, not listed: every results/bench/*_guardrail_summary.json
whose config has --kl, the ΨLM-2 constitution and dual runs first (their KLs
carry the paper's claims), then the rest -- the physics bridges' leaky and
shuffled sweeps on Qwen3-8B and Gemma 4 among them.

  python3 eval/kl_rescore_all.py [--tags a,b,c] [--dry-run]

Each run writes results/bench/<tag>_klpool_guardrail.rows.jsonl (resumable);
eval/kl_pool_shift.py aggregates them. Progress goes to
results/qwen35/kl_rescore.log, which ends with KL-RESCORE COMPLETE only when
every run finished, and with KL-RESCORE INCOMPLETE (exit status 1) otherwise.
--dry-run prints the commands and writes nothing.
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# priority order: the arms whose KL carries a claim in the ΨLM-2 paper first
FIRST = ["const_qwen35_all", "const_qwen35_vn10e", "const_qwen35_vn10ebot",
        "const_qwen35_vn1e", "const_qwen35_match41", "const_qwen35_vn5e", "const_qwen35_match205",
        "const_qwen35_match0_rg", "const_qwen35_match1_rg", "const_qwen35_match2_rg",
        "const_qwen35_vn", "const_qwen35_vn5",
        "dual_qwen35_both", "dual_qwen35_phys", "guardrail_qwen35",
        "const_qwen35_all_rt400", "const_qwen35_allplain_rt400", "const_qwen35_vn10e_rt400",
        "const_qwen35_vn10ebot_rt400", "const_qwen35_vn1e_rt400", "const_qwen35_vn5e_rt400",
        "const_qwen0.5b_vn", "const_qwen0.5b_vn5", "const_qwen0.5b_all", "const_qwen0.5b_rand",
        "const_qwen0.5b_magmatch", "const_qwen0.5b_plainpartner", "const_qwen0.5b_allplain"]

# config keys that are command-line options with a value; everything that shapes
# the prompts, the stack or the arms. Booleans and run-control flags are not here (but see FLAG_KEYS).
VALUE_KEYS = ["model", "hf_tokenizer", "ckpt", "fno", "phys_ckpt", "dual_channels", "bridge_kind",
              "const_model", "l_fwd", "l_rev", "gate_bias", "n", "seed", "datasets", "mmlu_subjects",
              "physics_data", "redteam_data", "physics_base_protocol", "nonphys_span",
              "max_new_gsm8k", "max_new_mmlu", "max_new_boolq", "max_new_redteam",
              "max_new_physics", "max_new_physics_base", "tasks_cache", "shuffle_values_from",
              "base_gen_from", "gsm8k_nudge"]
# the one boolean the rebuilt run must share: without it a run recorded past a failing
# staged-vs-stock parity check (leaky_gemma: relative 1.2e-2, argmax unchanged) exits at startup
FLAG_KEYS = ["no_parity_check"]
LOG = Path("results/qwen35/kl_rescore.log")


def step(msg):
    line = f"{msg} {time.strftime('%F %H:%M')}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def discover():
    """Every finished --kl run, FIRST in its order and the rest sorted."""
    found = []
    for f in sorted(glob.glob("results/bench/*_guardrail_summary.json")):
        tag = Path(f).name[:-len("_guardrail_summary.json")]
        if tag.endswith("_klpool"):
            continue
        if (json.loads(Path(f).read_text()).get("config") or {}).get("kl"):
            found.append(tag)
    return [t for t in FIRST if t in found] + [t for t in found if t not in FIRST]


def plan(tag):
    summ = Path(f"results/bench/{tag}_guardrail_summary.json")
    rows = Path(f"results/bench/{tag}_guardrail.rows.jsonl")
    if not (summ.exists() and rows.exists()):
        return None, f"no summary or rows for {tag}"
    cfg = json.loads(summ.read_text())["config"]
    for k in ("ckpt", "phys_ckpt", "fno", "tasks_cache", "base_gen_from", "shuffle_values_from"):
        if cfg.get(k) and not Path(cfg[k]).exists():
            return None, f"{tag}: its {k} {cfg[k]} is not in this checkout"
    arms, n_rec = [], 0
    for line in rows.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # the zeroed arm writes nothing, so its KL is zero under any read
        if r.get("kl") and r["arm"] != "zeroed":
            n_rec += 1
            if r["arm"] not in arms:
                arms.append(r["arm"])
    if not arms:
        return None, f"{tag}: no recorded KL outside the zeroed arm"
    out_tag = f"{tag}_klpool"
    out_rows = Path(f"results/bench/{out_tag}_guardrail.rows.jsonl")
    n_done = sum(1 for l in out_rows.read_text().splitlines() if l.strip()) if out_rows.exists() else 0
    cmd = [sys.executable, "eval/bench_guardrail.py", "--tag", out_tag, "--kl", "--kl-rescore", str(rows),
           "--arms", ",".join(["base"] + arms), "--print-every", "5"]
    for k in VALUE_KEYS:
        if cfg.get(k) is not None:
            cmd += ["--" + k.replace("_", "-"), str(cfg[k])]
    cmd += ["--" + k.replace("_", "-") for k in FLAG_KEYS if cfg.get(k)]
    if out_rows.exists():
        cmd.append("--resume")
    return {"tag": tag, "cmd": cmd, "n_recorded": n_rec, "n_done": n_done, "out_rows": out_rows}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=None, help="comma-separated; default: every finished --kl run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--attempts", type=int, default=6)
    a = ap.parse_args()
    os.chdir(ROOT)
    tags = [t for t in a.tags.split(",") if t] if a.tags else discover()
    log = print if a.dry_run else step            # a dry run leaves the live log alone
    env = dict(os.environ, HF_HUB_DISABLE_XET="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    psilm2 = Path(__file__).resolve().parents[2] / "PsiLM-2"       # the dual coupler lives there
    if psilm2.is_dir():
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(psilm2), env.get("PYTHONPATH")]))
    log(f"KL-RESCORE START ({len(tags)} runs: {', '.join(tags)})")
    done, skipped, gave_up = [], [], []
    for tag in tags:
        for attempt in range(1, a.attempts + 2):
            p, why = plan(tag)
            if p is None:
                log(f"RESCORE {tag} SKIPPED: {why}")
                skipped.append(tag)
                break
            if p["n_done"] >= p["n_recorded"]:
                log(f"RESCORE {tag} COMPLETE ({p['n_done']} of {p['n_recorded']} KLs)")
                done.append(tag)
                break
            if a.dry_run:
                print(" ".join(p["cmd"]))
                break
            if attempt > a.attempts:
                log(f"RESCORE {tag} GAVE UP at {p['n_done']} of {p['n_recorded']}")
                gave_up.append(tag)
                break
            log(f"RESCORE {tag} attempt {attempt}: {p['n_done']} of {p['n_recorded']} done")
            with open(f"results/bench/{tag}_klpool_run.log", "a") as lf:
                rc = subprocess.call(p["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=env)
            if rc != 0:
                log(f"RESCORE {tag} attempt {attempt} exited {rc}; resuming")
                time.sleep(30)
    if a.dry_run:
        return 0
    if len(done) == len(tags):
        log(f"KL-RESCORE COMPLETE ({len(done)} of {len(tags)} runs)")
        return 0
    log(f"KL-RESCORE INCOMPLETE ({len(done)} of {len(tags)} runs; skipped: {', '.join(skipped) or 'none'}; "
        f"gave up: {', '.join(gave_up) or 'none'})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
