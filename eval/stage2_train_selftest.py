#!/usr/bin/env python3
"""The physics trainer's chunking options, checked end to end on the 4-bit 0.5B.

eval/mlx_stage2_train.py gained --l-fwd, --save-every and --init-seed for the
Bonsai run (results/bonsai/physics_bonsai.sh), where a chunk is hours long. What
must hold, and is checked here by running the trainer itself:

  init     two fresh runs with the same --init-seed end with the same bridges
  l-fwd    the read layer asked for is the one recorded (meta, log line), and a
           resume at another read layer is refused
  save     a run that saves mid-invocation (--save-every) and is resumed from
           there ends with the bridges of the run that was never interrupted,
           and the mid-run checkpoint carries its own step
  eval     eval/mlx_stage2_eval.py scores at the recorded read layer

  python eval/stage2_train_selftest.py        # about two minutes; exit 0 when all hold

Writes and removes results/stage2_selftest_*; touches nothing else.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
M, T = "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "Qwen/Qwen2.5-0.5B-Instruct"
C = ["--model", M, "--hf-tokenizer", T, "--batch", "2", "--lr", "3e-4", "--l-rev", "15", "--channel", "value",
     "--inj-cap", "0.2", "--gate-bias", "0.0", "--lam-x0", "1.0", "--clip", "module", "--detach-x0",
     "--readout-norm", "dim", "--calib-n", "4", "--readout-only", "2", "--eval-n", "1"]


def train(tag, *extra, ok=True):
    r = subprocess.run([PY, "eval/mlx_stage2_train.py", "--tag", tag, *C, *extra], cwd=ROOT,
                       capture_output=True, text=True)
    if ok and r.returncode != 0:
        raise SystemExit(f"trainer failed for {tag}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


def d(tag):
    return ROOT / f"results/stage2{tag}"


def weights(tag):
    import mlx.core as mx
    return {k: np.array(v) for k, v in mx.load(str(d(tag) / "bridges.npz")).items()}


def worst(a, b):
    assert a.keys() == b.keys()
    return max(float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in a)


def meta(tag):
    return json.loads((d(tag) / "bridges.npz.meta").read_text())


def main():
    tags = ["_selftest_a", "_selftest_b", "_selftest_one", "_selftest_two"]
    for t in tags:
        shutil.rmtree(d(t), ignore_errors=True)
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}{(' (' + detail + ')') if detail else ''}", flush=True)

    try:
        ra = train("_selftest_a", "--fresh", "--steps", "4", "--l-fwd", "9", "--init-seed", "1")
        train("_selftest_b", "--fresh", "--steps", "4", "--l-fwd", "9", "--init-seed", "1")
        w = worst(weights("_selftest_a"), weights("_selftest_b"))
        check("two fresh runs with one --init-seed end with the same bridges", w <= 1e-6, f"worst difference {w:.1e}")
        ma = meta("_selftest_a")
        check("the read layer asked for is recorded", ma["l_fwd"] == 9 and ma["l_rev"] == 15 and ma["step"] == 4
              and "coupling 9/15 of 24" in ra.stdout, f"meta {ma['l_fwd']}/{ma['l_rev']} step {ma['step']}")
        r = train("_selftest_a", "--steps", "1", "--l-fwd", "10", ok=False)
        check("a resume at another read layer is refused", r.returncode != 0 and "reads at layer 9" in (r.stdout + r.stderr)
              and meta("_selftest_a")["step"] == 4)
        r = train("_selftest_a", "--steps", "1", ok=False)       # no --l-fwd: the rule's 10, not the checkpoint's 9
        check("a resume that leaves --l-fwd out is refused too", r.returncode != 0 and meta("_selftest_a")["step"] == 4)
        for t in ("_selftest_one", "_selftest_two"):
            shutil.copytree(d("_selftest_a"), d(t))
        train("_selftest_one", "--steps", "6", "--l-fwd", "9")
        r1 = train("_selftest_two", "--steps", "3", "--l-fwd", "9", "--save-every", "2")
        mid = "SAVED step=6" in r1.stdout and meta("_selftest_two")["step"] == 7
        train("_selftest_two", "--steps", "3", "--l-fwd", "9", "--save-every", "2")
        w = worst(weights("_selftest_one"), weights("_selftest_two"))
        check("six steps in one invocation equal three and three with mid-run saves", mid and w <= 1e-6
              and meta("_selftest_one")["step"] == meta("_selftest_two")["step"] == 10, f"worst difference {w:.1e}")
        # a checkpoint written mid-run is a whole one: resume from a copy of it and land on the same bridges
        shutil.rmtree(d("_selftest_b"))
        shutil.copytree(d("_selftest_a"), d("_selftest_b"))
        p = subprocess.Popen([PY, "eval/mlx_stage2_train.py", "--tag", "_selftest_b", *C, "--steps", "6", "--l-fwd", "9",
                              "--save-every", "2", "--eval-n", "1"], cwd=ROOT, stdout=subprocess.PIPE, text=True)
        seen = False
        for line in p.stdout:
            if line.startswith("SAVED step=6"):
                seen = True
                p.kill()                                       # a crash after the mid-run save
                break
        p.wait()
        ms = meta("_selftest_b")["step"] if seen else None
        train("_selftest_b", "--steps", str(10 - (ms or 4)), "--l-fwd", "9")
        w = worst(weights("_selftest_one"), weights("_selftest_b"))
        check("a run killed after a mid-run save resumes to the same bridges", seen and ms == 6 and w <= 1e-6,
              f"resumed from step {ms}, worst difference {w:.1e}")
        r = subprocess.run([PY, "eval/mlx_stage2_eval.py", "--model", M, "--hf-tokenizer", T, "--tag", "_selftest_one",
                            "--n", "1", "--arms", "psilm", "--out", "selftest_eval.json"], cwd=ROOT,
                           capture_output=True, text=True)
        check("the evaluator scores at the recorded read layer", r.returncode == 0 and "couple 9/15 of 24" in r.stdout,
              (r.stdout + r.stderr)[-200:].replace("\n", " ") if r.returncode else "")
    finally:
        for t in tags:
            shutil.rmtree(d(t), ignore_errors=True)
    print(f"[self-test] eval/stage2_train_selftest.py: {sum(checks)} of {len(checks)} checks hold")
    return 0 if checks and all(checks) else 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.exit(main())
