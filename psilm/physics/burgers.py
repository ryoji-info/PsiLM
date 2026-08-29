"""Ground truth: 1D viscous Burgers equation, periodic domain, spectral solver.

    u_t + u u_x = nu * u_xx,   x in [0, 1),  periodic

Initial conditions are one-parameter-family sinusoids u0(x) = a sin(2 pi x + phi).
Solved pseudo-spectrally (RK4, 2/3 dealiasing) — this is the "real physics"
that generates training data for the neural operator and the QA ground truth.
NumPy only; runs on CPU in milliseconds per trajectory.
"""

import numpy as np

N = 128          # spatial grid
NU = 0.02        # viscosity: strong enough to resolve shocks at N=128, weak
                 # enough that u(., 1) keeps O(0.5) amplitude for meaningful QA
T_FINAL = 0.5    # long enough for real nonlinear steepening, short enough
                 # that u keeps O(0.5) amplitude — answers stay non-trivial
DT = 1e-3

_x = np.arange(N) / N
_k = 2 * np.pi * np.fft.rfftfreq(N, d=1.0 / N)      # wavenumbers
_dealias = np.ones(N // 2 + 1)
_dealias[int(N // 3):] = 0.0


def grid():
    return _x.copy()


def initial_condition(a: float, phi: float):
    return a * np.sin(2 * np.pi * _x + phi)


def initial_condition_multi(modes):
    """modes: iterable of (m, a, phi) — u0 = sum a sin(2 pi m x + phi)."""
    u0 = np.zeros_like(_x)
    for m, a, phi in modes:
        u0 += a * np.sin(2 * np.pi * m * _x + phi)
    return u0


def _nonlinear(u_hat):
    u = np.fft.irfft(u_hat, n=N)
    ux = np.fft.irfft(1j * _k * u_hat, n=N)
    return -np.fft.rfft(u * ux) * _dealias


def solve(u0: np.ndarray, t_final: float = T_FINAL, dt: float = DT):
    """Integrate to t_final with integrating-factor RK4.

    The stiff viscous term (nu k^2 up to ~8e3 at N=128) is propagated exactly
    by its exponential; RK4 handles only the nonlinear term, so dt is limited
    by the advective CFL, not the diffusive one.
    """
    u_hat = np.fft.rfft(u0)
    steps = int(round(t_final / dt))
    L = -NU * _k**2
    E = np.exp(L * dt / 2.0)
    E2 = E * E
    for _ in range(steps):
        k1 = _nonlinear(u_hat)
        k2 = _nonlinear(E * (u_hat + 0.5 * dt * k1))
        k3 = _nonlinear(E * u_hat + 0.5 * dt * k2)
        k4 = _nonlinear(E2 * u_hat + dt * E * k3)
        u_hat = E2 * u_hat + dt / 6.0 * (E2 * k1 + 2 * E * (k2 + k3) + k4)
    return np.fft.irfft(u_hat, n=N)


def sample_params(rng):
    return round(rng.uniform(0.5, 1.5), 2), round(rng.uniform(0.0, 6.28), 2)


def make_dataset(n: int, seed: int = 0):
    """(u0, uT) pairs plus the generating parameters."""
    rng = np.random.default_rng(seed)
    u0s, uts, params = [], [], []
    for _ in range(n):
        a = round(float(rng.uniform(0.5, 1.5)), 2)
        phi = round(float(rng.uniform(0.0, 6.28)), 2)
        u0 = initial_condition(a, phi)
        u0s.append(u0)
        uts.append(solve(u0))
        params.append((a, phi))
    return np.stack(u0s), np.stack(uts), params
