"""Ground truth for Stage 2d: 2D Fisher-KPP reaction-diffusion, periodic.

    u_t = D lap(u) + r u (1 - u),   (x, y) in [0,1)^2 periodic

Initial conditions are Gaussian bumps (height a, center (cx, cy), width w,
periodically wrapped). Fronts expand from the bump and saturate toward u=1 —
nonlinear, no closed form, and every parameter is text-describable.
Integrating-factor RK4 (the Stage-2 stiffness lesson), NumPy only.
"""

import numpy as np

N = 128
D = 0.001
R = 6.0
T_FINAL = 0.4
DT = 2e-3

_x = np.arange(N) / N
_kx = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
_ky = 2 * np.pi * np.fft.rfftfreq(N, d=1.0 / N)
_k2 = _kx[:, None] ** 2 + _ky[None, :] ** 2


def wrapped_sq_dist(x, c):
    d = np.abs(x - c)
    return np.minimum(d, 1.0 - d) ** 2


def initial_condition(a, cx, cy, w):
    dx2 = wrapped_sq_dist(_x[:, None], cx)
    dy2 = wrapped_sq_dist(_x[None, :], cy)
    return a * np.exp(-(dx2 + dy2) / (2 * w * w))


def _nonlinear(u_hat):
    u = np.fft.irfft2(u_hat, s=(N, N))
    return np.fft.rfft2(R * u * (1.0 - u))


def solve(u0, t_final=T_FINAL, dt=DT):
    u_hat = np.fft.rfft2(u0)
    L = -D * _k2
    E = np.exp(L * dt / 2.0)
    E2 = E * E
    for _ in range(int(round(t_final / dt))):
        k1 = _nonlinear(u_hat)
        k2 = _nonlinear(E * (u_hat + 0.5 * dt * k1))
        k3 = _nonlinear(E * u_hat + 0.5 * dt * k2)
        k4 = _nonlinear(E2 * u_hat + dt * E * k3)
        u_hat = E2 * u_hat + dt / 6.0 * (E2 * k1 + 2 * E * (k2 + k3) + k4)
    return np.fft.irfft2(u_hat, s=(N, N))


def bilinear_periodic(field, x0, y0):
    """Evaluate the grid field at (x0, y0) with periodic bilinear interpolation."""
    fx, fy = x0 * N, y0 * N
    i0, j0 = int(np.floor(fx)) % N, int(np.floor(fy)) % N
    i1, j1 = (i0 + 1) % N, (j0 + 1) % N
    tx, ty = fx - np.floor(fx), fy - np.floor(fy)
    return float(
        field[i0, j0] * (1 - tx) * (1 - ty)
        + field[i1, j0] * tx * (1 - ty)
        + field[i0, j1] * (1 - tx) * ty
        + field[i1, j1] * tx * ty
    )


def sample_ic(rng):
    return (round(rng.uniform(0.3, 0.9), 2),
            round(rng.uniform(0.0, 0.99), 2),
            round(rng.uniform(0.0, 0.99), 2),
            round(rng.uniform(0.04, 0.10), 2))


def sample_query(rng, cx, cy):
    """Query point at wrapped distance U(0, 0.3) from the bump center, so
    answers span the front profile instead of piling up at u ~ 0."""
    d = rng.uniform(0.0, 0.20)
    th = rng.uniform(0.0, 2 * np.pi)
    return (round((cx + d * np.cos(th)) % 1.0, 2),
            round((cy + d * np.sin(th)) % 1.0, 2))
