"""Generate the Stage-0 physics QA set (UTOPIA-style relative comparisons).

Every item is a three-way multiple choice — (A) first object, (B) second
object, (C) about the same — scored against the closed-form simulator. Params
are sampled so that A/B answers have a >=15% margin, and C items are physics
traps whose compared quantities are exactly equal despite differing surface
parameters (mass in free fall, complementary launch angles, etc.).

Usage: python eval/generate_qa.py [--n-per-scene 12] [--seed 0] [--out data/qa_stage0.json]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psilm.simulator import SCENES, simulate  # noqa: E402

MARGIN = 0.15  # required relative gap for A/B answers


def _label(x, y):
    gap = abs(x - y) / max(abs(x), abs(y))
    if gap < 0.05:
        return "C", gap
    return ("A" if _first_wins(x, y) else "B"), gap


def _first_wins(x, y):
    return x > y


def make_free_fall(rng, trap):
    if trap:
        h = round(rng.uniform(5, 80), 1)
        m_a, m_b = rng.sample([1, 2, 3, 5, 8, 10], 2)
        params = {"h_a": h, "h_b": h}
        q = (
            f"Object A has a mass of {m_a} kg and object B has a mass of {m_b} kg. "
            f"Both are dropped from rest from a height of {h} m (no air resistance). "
            f"Which object takes longer to reach the ground?"
        )
        return params, q, "longer time"
    while True:
        h_a, h_b = (round(rng.uniform(5, 80), 1) for _ in range(2))
        r = simulate("free_fall", {"h_a": h_a, "h_b": h_b})
        if abs(r["t_a"] - r["t_b"]) / max(r["t_a"], r["t_b"]) >= MARGIN:
            break
    q = (
        f"Object A is dropped from rest from {h_a} m. Object B is dropped from rest "
        f"from {h_b} m (no air resistance). Which object takes longer to reach the ground?"
    )
    return {"h_a": h_a, "h_b": h_b}, q, "longer time"


def make_projectile(rng, trap):
    if trap:
        v = round(rng.uniform(10, 40), 1)
        a = rng.choice([20, 25, 30, 35, 40])
        params = {"v_a": v, "angle_a_deg": a, "v_b": v, "angle_b_deg": 90 - a}
        q = (
            f"Projectile A is launched at {v} m/s at {a} degrees above the horizontal. "
            f"Projectile B is launched at {v} m/s at {90 - a} degrees. Both start from "
            f"ground level (no air resistance). Which projectile lands farther away?"
        )
        return params, q, "greater range"
    while True:
        v_a, v_b = (round(rng.uniform(10, 40), 1) for _ in range(2))
        a_a, a_b = (rng.choice([20, 30, 40, 45, 50, 60, 70]) for _ in range(2))
        params = {"v_a": v_a, "angle_a_deg": a_a, "v_b": v_b, "angle_b_deg": a_b}
        r = simulate("projectile", params)
        if abs(r["range_a"] - r["range_b"]) / max(r["range_a"], r["range_b"]) >= MARGIN:
            break
    q = (
        f"Projectile A is launched at {v_a} m/s at {a_a} degrees above the horizontal. "
        f"Projectile B is launched at {v_b} m/s at {a_b} degrees. Both start from ground "
        f"level (no air resistance). Which projectile lands farther away?"
    )
    return params, q, "greater range"


def make_friction(rng, trap):
    if trap:
        v = round(rng.uniform(3, 12), 1)
        mu = round(rng.uniform(0.15, 0.6), 2)
        m_a, m_b = rng.sample([1, 2, 4, 6, 10], 2)
        params = {"v_a": v, "mu_a": mu, "v_b": v, "mu_b": mu}
        q = (
            f"Block A (mass {m_a} kg) and block B (mass {m_b} kg) both slide onto a "
            f"horizontal surface at {v} m/s. The friction coefficient is {mu} for both. "
            f"Which block slides farther before stopping?"
        )
        return params, q, "greater sliding distance"
    while True:
        v_a, v_b = (round(rng.uniform(3, 12), 1) for _ in range(2))
        mu_a, mu_b = (round(rng.uniform(0.15, 0.6), 2) for _ in range(2))
        params = {"v_a": v_a, "mu_a": mu_a, "v_b": v_b, "mu_b": mu_b}
        r = simulate("friction_slide", params)
        if abs(r["d_a"] - r["d_b"]) / max(r["d_a"], r["d_b"]) >= MARGIN:
            break
    q = (
        f"Block A slides onto a horizontal surface at {v_a} m/s where the friction "
        f"coefficient is {mu_a}. Block B slides on at {v_b} m/s where the friction "
        f"coefficient is {mu_b}. Which block slides farther before stopping?"
    )
    return params, q, "greater sliding distance"


def make_collision(rng, trap):
    if trap:
        m = rng.choice([1, 2, 3, 4])
        v1 = round(rng.uniform(2, 10), 1)
        params = {"m1": m, "v1": v1, "m2": m, "v2": 0.0}
        q = (
            f"Ball 1 (mass {m} kg) moves at {v1} m/s toward ball 2 (mass {m} kg), "
            f"which is at rest. They collide head-on, perfectly elastically. "
            f"Which ball has the greater speed after the collision?"
        )
        # equal masses, target at rest: ball 1 stops, ball 2 takes v1 -> answer B.
        return params, q, "greater speed after the collision"
    while True:
        m1, m2 = (rng.choice([1, 2, 3, 5, 8]) for _ in range(2))
        if m1 == m2:
            continue
        v1 = round(rng.uniform(2, 10), 1)
        v2 = round(rng.choice([0.0, -rng.uniform(1, 5)]), 1)
        params = {"m1": m1, "v1": v1, "m2": m2, "v2": v2}
        r = simulate("elastic_collision", params)
        hi, lo = max(r.values()), min(r.values())
        if hi > 0 and (hi - lo) / hi >= MARGIN:
            break
    v2_txt = "at rest" if v2 == 0 else f"moving toward it at {abs(v2)} m/s"
    q = (
        f"Ball 1 (mass {m1} kg) moves at {v1} m/s toward ball 2 (mass {m2} kg), "
        f"which is {v2_txt}. "
        f"They collide head-on, perfectly elastically. Which ball has the greater "
        f"speed after the collision?"
    )
    return params, q, "greater speed after the collision"


def make_incline(rng, trap):
    length = round(rng.uniform(3, 15), 1)
    if trap:
        angle = rng.choice([25, 30, 35, 40, 45])
        mu = round(rng.uniform(0.05, 0.3), 2)
        m_a, m_b = rng.sample([1, 2, 4, 7, 12], 2)
        params = {
            "angle_a_deg": angle, "mu_a": mu,
            "angle_b_deg": angle, "mu_b": mu, "length": length,
        }
        q = (
            f"Block A (mass {m_a} kg) and block B (mass {m_b} kg) are released from rest "
            f"at the top of identical {length} m slopes inclined at {angle} degrees, "
            f"friction coefficient {mu}. Which block reaches the bottom first?"
        )
        return params, q, "reaches the bottom first (shorter time)"
    while True:
        a_a, a_b = (rng.choice([20, 25, 30, 35, 40, 45, 50]) for _ in range(2))
        mu_a, mu_b = (round(rng.uniform(0.05, 0.35), 2) for _ in range(2))
        params = {
            "angle_a_deg": a_a, "mu_a": mu_a,
            "angle_b_deg": a_b, "mu_b": mu_b, "length": length,
        }
        try:
            r = simulate("incline", params)
        except ValueError:
            continue
        if abs(r["t_a"] - r["t_b"]) / max(r["t_a"], r["t_b"]) >= MARGIN:
            break
    q = (
        f"Block A is released from rest at the top of a {length} m slope inclined at "
        f"{a_a} degrees with friction coefficient {mu_a}. Block B is released on a "
        f"{length} m slope at {a_b} degrees with friction coefficient {mu_b}. "
        f"Which block reaches the bottom first?"
    )
    return params, q, "reaches the bottom first (shorter time)"


MAKERS = {
    "free_fall": make_free_fall,
    "projectile": make_projectile,
    "friction_slide": make_friction,
    "elastic_collision": make_collision,
    "incline": make_incline,
}

# For these quantities, "first" (A) means the LARGER value of output a vs b —
# except time-to-bottom on the incline, where smaller time wins.
SMALLER_WINS = {"incline"}


def answer_from_sim(scene, params):
    r = simulate(scene, params)
    vals = list(r.values())
    x, y = vals[0], vals[1]
    gap = abs(x - y) / max(abs(x), abs(y)) if max(abs(x), abs(y)) > 0 else 0.0
    if gap < 0.05:
        return "C", r
    if scene in SMALLER_WINS:
        return ("A" if x < y else "B"), r
    return ("A" if x > y else "B"), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-scene", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/qa_stage0.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = []
    for scene, maker in MAKERS.items():
        n_trap = args.n_per_scene // 3
        flags = [True] * n_trap + [False] * (args.n_per_scene - n_trap)
        rng.shuffle(flags)
        for trap in flags:
            params, question, compare_phrase = maker(rng, trap)
            answer, sim = answer_from_sim(scene, params)
            items.append({
                "id": f"{scene}_{len(items):03d}",
                "scene": scene,
                "question": question,
                "options": {
                    "A": "the first object (A / ball 1 / block A / projectile A)",
                    "B": "the second object (B / ball 2 / block B / projectile B)",
                    "C": "about the same (within 5%)",
                },
                "answer": answer,
                "params": params,
                "sim": {k: round(v, 4) for k, v in sim.items()},
            })
    rng.shuffle(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=1))
    counts = {}
    for it in items:
        counts[it["answer"]] = counts.get(it["answer"], 0) + 1
    print(f"wrote {len(items)} items to {out} — answer distribution: {counts}")


if __name__ == "__main__":
    main()
