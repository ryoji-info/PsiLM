"""Deterministic rigid-body physics, closed form. The Stage-0 physics hemisphere.

Five scene types modeled on UTOPIA (Mind's Eye, ICLR 2023): free fall,
projectile, friction slide, elastic collision, inclined plane. Each scene takes
a parameter dict and returns the physical quantities the question asks about.
SI units throughout; g = 9.8 m/s^2.
"""

import math

G = 9.8

# Each entry: (params spec for the tool prompt, computed outputs)
SCENES = {
    "free_fall": {
        "params": {"h_a": "drop height of object A in m", "h_b": "drop height of object B in m"},
        "outputs": ("t_a", "t_b"),
        "unit": "s",
        "quantity": "time to reach the ground",
    },
    "projectile": {
        "params": {
            "v_a": "launch speed of projectile A in m/s",
            "angle_a_deg": "launch angle of projectile A in degrees",
            "v_b": "launch speed of projectile B in m/s",
            "angle_b_deg": "launch angle of projectile B in degrees",
        },
        "outputs": ("range_a", "range_b"),
        "unit": "m",
        "quantity": "horizontal range",
    },
    "friction_slide": {
        "params": {
            "v_a": "initial speed of block A in m/s",
            "mu_a": "friction coefficient under block A",
            "v_b": "initial speed of block B in m/s",
            "mu_b": "friction coefficient under block B",
        },
        "outputs": ("d_a", "d_b"),
        "unit": "m",
        "quantity": "sliding distance before stopping",
    },
    "elastic_collision": {
        "params": {
            "m1": "mass of ball 1 in kg",
            "v1": "initial velocity of ball 1 in m/s (positive means moving toward ball 2)",
            "m2": "mass of ball 2 in kg",
            "v2": "initial velocity of ball 2 in m/s (NEGATIVE if moving toward ball 1; 0 if at rest)",
        },
        "outputs": ("speed_1_after", "speed_2_after"),
        "unit": "m/s",
        "quantity": "speed after the collision",
    },
    "incline": {
        "params": {
            "angle_a_deg": "incline angle for block A in degrees",
            "mu_a": "friction coefficient under block A",
            "angle_b_deg": "incline angle for block B in degrees",
            "mu_b": "friction coefficient under block B",
            "length": "slope length in m (same for both)",
        },
        "outputs": ("t_a", "t_b"),
        "unit": "s",
        "quantity": "time to slide down the slope",
    },
}


def simulate(scene: str, params: dict) -> dict:
    """Run one scene. Raises KeyError/ValueError on bad scene or parameters."""
    p = {k: float(v) for k, v in params.items()}
    if scene == "free_fall":
        return {
            "t_a": math.sqrt(2 * p["h_a"] / G),
            "t_b": math.sqrt(2 * p["h_b"] / G),
        }
    if scene == "projectile":
        return {
            "range_a": p["v_a"] ** 2 * math.sin(2 * math.radians(p["angle_a_deg"])) / G,
            "range_b": p["v_b"] ** 2 * math.sin(2 * math.radians(p["angle_b_deg"])) / G,
        }
    if scene == "friction_slide":
        return {
            "d_a": p["v_a"] ** 2 / (2 * p["mu_a"] * G),
            "d_b": p["v_b"] ** 2 / (2 * p["mu_b"] * G),
        }
    if scene == "elastic_collision":
        m1, m2, v1, v2 = p["m1"], p["m2"], p["v1"], p["v2"]
        v1f = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2f = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
        return {"speed_1_after": abs(v1f), "speed_2_after": abs(v2f)}
    if scene == "incline":
        out = {}
        for side, angle_key, mu_key in (("a", "angle_a_deg", "mu_a"), ("b", "angle_b_deg", "mu_b")):
            th = math.radians(p[angle_key])
            a = G * (math.sin(th) - p[mu_key] * math.cos(th))
            if a <= 0:
                raise ValueError(f"block {side} does not slide (a={a:.3f})")
            out[f"t_{side}"] = math.sqrt(2 * p["length"] / a)
        return out
    raise KeyError(f"unknown scene: {scene}")


def render_result(scene: str, result: dict) -> str:
    """Format simulator output as a sentence for the LLM's context."""
    spec = SCENES[scene]
    parts = [f"{k} = {v:.3f} {spec['unit']}" for k, v in result.items()]
    return f"Simulation result ({spec['quantity']}): " + ", ".join(parts)
