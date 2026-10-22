"""Verification tests for the CO2 mass-balance model.

These use manufactured data where the true answer is known exactly, so
they check the *implementation* (verification), not whether the model
describes real rooms (validation, done in the analysis scripts).
"""
import numpy as np
import pytest

from occupancy.physics import (
    RoomParams,
    calibrate,
    derivative,
    estimate_occupancy,
    euler_stable_step,
    simulate,
)

P = RoomParams(V=150.0, Q=0.125, C_out=420.0)  # tau = 1200 s, 3 ACH


def step_input(n=200, h=60.0, level=10):
    t = np.arange(n) * h
    N = np.where(t >= 0, level, 0).astype(float)
    return t, N


def test_steady_state_and_ach():
    assert P.ach == pytest.approx(3.0)
    # 10 people, Q=0.125 m3/s -> 1e6*5.2e-6*10/0.125 = 416 ppm above outdoor
    assert P.steady_state(10) == pytest.approx(420 + 416)


def test_exact_matches_analytical():
    t, N = step_input()
    C = simulate(t, N, P, method="exact")
    css = P.steady_state(10)
    expected = css + (P.C_out - css) * np.exp(-t / P.tau)
    np.testing.assert_allclose(C, expected, rtol=1e-12)


@pytest.mark.parametrize("method,order", [("euler", 1), ("rk4", 4)])
def test_convergence_order(method, order):
    """Global error should fall as h^order (observed order within 0.3)."""
    T = 3600.0
    errs = []
    hs = [600.0, 300.0, 150.0]
    for h in hs:
        t = np.arange(0, T + h / 2, h)
        N = np.full_like(t, 10.0)
        C = simulate(t, N, P, method=method)
        exact = simulate(t, N, P, method="exact")
        errs.append(abs(C[-1] - exact[-1]))
    observed = np.log(errs[0] / errs[-1]) / np.log(hs[0] / hs[-1])
    assert observed == pytest.approx(order, abs=0.3)


def test_euler_instability_beyond_limit():
    h = 1.2 * euler_stable_step(P)  # beyond 2*tau -> diverges
    t = np.arange(30) * h
    C = simulate(t, np.zeros_like(t), P, C0=1000.0, method="euler")
    assert abs(C[-1] - P.C_out) > abs(1000.0 - P.C_out)
    # RK4 at a stable step decays as it should
    t2 = np.arange(30) * 300.0
    C2 = simulate(t2, np.zeros_like(t2), P, C0=1000.0, method="rk4")
    assert abs(C2[-1] - P.C_out) < 1.0


def test_inverse_recovers_constant_occupancy():
    t = np.arange(0, 6 * 3600, 60.0)
    N = np.full_like(t, 7.0)
    C = simulate(t, N, P, method="exact")
    Nhat = estimate_occupancy(C, 60.0, P, scheme="central")
    np.testing.assert_allclose(Nhat[2:-2], 7.0, atol=0.05)


def test_derivative_schemes_on_polynomial():
    dt = 10.0
    x = np.arange(50) * dt
    C = 3e-4 * x**2 + 2 * x + 400
    exact = 6e-4 * x + 2
    np.testing.assert_allclose(derivative(C, dt, "central")[1:-1], exact[1:-1], rtol=1e-10)
    np.testing.assert_allclose(derivative(C, dt, "savgol", 5), exact, rtol=1e-8)
    with pytest.raises(ValueError):
        derivative(C, dt, "savgol", 4)


def test_calibration_recovers_parameters():
    rng = np.random.default_rng(1)
    t = np.arange(0, 3 * 86400, 60.0)
    N = np.repeat(rng.integers(0, 12, len(t) // 60 + 1), 60)[: len(t)].astype(float)
    C = simulate(t, N, P, method="exact")
    est = calibrate(C, N, 60.0, scheme="central")
    assert est.V == pytest.approx(P.V, rel=0.05)
    assert est.Q == pytest.approx(P.Q, rel=0.05)
    assert est.C_out == pytest.approx(P.C_out, abs=5)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        RoomParams(V=-1, Q=0.1)
    with pytest.raises(ValueError):
        simulate(np.array([0.0, 0.0]), np.array([1.0, 1.0]), P)
    with pytest.raises(ValueError):
        simulate(np.array([0.0, 1.0]), np.array([1.0]), P)
    with pytest.raises(ValueError):
        simulate(np.array([0.0, 1.0]), np.array([1.0, 1.0]), P, method="bogus")
    with pytest.raises(ValueError):
        calibrate(np.full(100, 420.0), np.zeros(100), 60.0)  # unexcited room


def test_fast_simulation_matches_loop():
    from occupancy.physics import simulate_uniform_fast

    rng = np.random.default_rng(3)
    t = np.arange(0, 86400, 60.0)
    N = rng.integers(0, 15, len(t)).astype(float)
    ref = simulate(t, N, P, C0=800.0, method="exact")
    fast = simulate_uniform_fast(N, 60.0, P, C0=800.0)
    np.testing.assert_allclose(fast, ref, rtol=1e-10)
