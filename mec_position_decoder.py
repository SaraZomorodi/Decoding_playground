"""Position decoding for MEC grid-cell recordings.

This module provides a practical pipeline to compare:
1) decoding from spikes alone, and
2) decoding from spikes with a motion prior based on speed + head direction.

Data assumptions:
- spikes_by_cell: dict[cell_id -> 1D array of spike times in seconds]
- x, y, t: 1D arrays of animal position/time (typically 120 Hz)
- hd: 1D array of head direction in radians, same length as t
- speed: 1D array of speed, same length as t (cm/s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except Exception:
    _scipy_gaussian_filter = None


Array = np.ndarray


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> Array:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def _gaussian_filter_numpy(arr: Array, sigma: float) -> Array:
    if sigma <= 0:
        return arr.copy()
    k = _gaussian_kernel1d(sigma)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=out)
    return out


def gaussian_filter(arr: Array, sigma: float) -> Array:
    if _scipy_gaussian_filter is not None:
        return _scipy_gaussian_filter(arr, sigma=sigma)
    return _gaussian_filter_numpy(arr, sigma=sigma)


@dataclass
class DecoderConfig:
    n_particles: int = 1500
    n_x_bins: int = 40
    n_y_bins: int = 40
    rate_smoothing_sigma_bins: float = 1.2
    process_noise_cm: float = 2.0
    occupancy_epsilon_s: float = 1e-3
    rate_floor_hz: float = 1e-6
    random_seed: int = 0


@dataclass
class DecodeResult:
    x_hat: Array
    y_hat: Array
    error_cm: Array
    rmse_cm: float
    median_cm: float


@dataclass
class ParticleSnapshot:
    """Diagnostic state for visualizing a particle-filter step."""

    ti: int
    t_s: float
    true_x: float
    true_y: float
    hd_rad: float
    speed: float
    dx_motion: float
    dy_motion: float
    prev_px: Array
    prev_py: Array
    prev_w: Array
    prior_px: Array
    prior_py: Array
    post_w: Array
    x_hat: float
    y_hat: float
    ess: float
    did_resample: bool


@dataclass
class ModulationResult:
    cell_ids: List[str]
    beta_speed: Array
    beta_cos_hd: Array
    beta_sin_hd: Array
    hd_pref_rad: Array
    hd_strength: Array
    ll_offset_only: Array
    ll_full: Array
    delta_ll: Array


def _as_1d_float(x: Iterable[float], name: str) -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def validate_inputs(
    spikes_by_cell: Dict[str, Array],
    x: Iterable[float],
    y: Iterable[float],
    t: Iterable[float],
    hd: Iterable[float],
    speed: Iterable[float],
) -> Tuple[Array, Array, Array, Array, Array, List[str]]:
    """Validate and return cleaned arrays plus sorted cell IDs."""
    x = _as_1d_float(x, "x")
    y = _as_1d_float(y, "y")
    t = _as_1d_float(t, "t")
    hd = _as_1d_float(hd, "hd")
    speed = _as_1d_float(speed, "speed")

    n = len(t)
    for name, arr in (("x", x), ("y", y), ("hd", hd), ("speed", speed)):
        if len(arr) != n:
            raise ValueError(f"{name} length ({len(arr)}) must match t length ({n})")

    if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("t must be finite and strictly increasing")

    if len(spikes_by_cell) == 0:
        raise ValueError("spikes_by_cell is empty")

    cell_ids = sorted(k for k in spikes_by_cell.keys())
    return x, y, t, hd, speed, cell_ids


def binned_spike_counts(spikes_by_cell: Dict[str, Array], t: Array, cell_ids: List[str]) -> Array:
    """Bin spike times to behavioral timestamps.

    Returns:
        counts: shape (n_cells, n_time)
    """
    dt = float(np.median(np.diff(t)))
    edges = np.r_[t - dt / 2.0, t[-1] + dt / 2.0]

    counts = np.zeros((len(cell_ids), len(t)), dtype=np.int16)
    for i, cid in enumerate(cell_ids):
        st = np.asarray(spikes_by_cell[cid], dtype=float)
        st = st[np.isfinite(st)]
        if st.size == 0:
            continue
        counts[i], _ = np.histogram(st, bins=edges)
    return counts


def make_train_test_split(t: Array, train_fraction: float = 0.7) -> Tuple[Array, Array]:
    if not (0.1 <= train_fraction <= 0.95):
        raise ValueError("train_fraction should be in [0.1, 0.95]")
    n = len(t)
    n_train = int(np.floor(train_fraction * n))
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n)
    return train_idx, test_idx


def pick_nonconsecutive_timepoints(indices: Array, n: int = 10, min_sep: int = 2) -> Array:
    """Pick up to n indices that are not consecutive (diff >= min_sep).

    This is mainly intended for choosing diagnostic snapshot timepoints.
    """
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if indices.size == 0 or n <= 0:
        return np.array([], dtype=int)

    indices = np.unique(indices)
    indices.sort()

    stride = max(int(min_sep), int(indices.size // n) if n > 0 else int(min_sep))
    stride = max(stride, int(min_sep))
    pos = np.arange(0, indices.size, stride, dtype=int)
    if pos.size == 0:
        pos = np.array([0], dtype=int)

    n_use = int(min(n, pos.size))
    pos_pick = np.linspace(0, pos.size - 1, n_use, dtype=int)
    chosen = indices[pos[pos_pick]]

    # Final enforcement in case `indices` are irregular.
    kept: List[int] = [int(chosen[0])]
    for ti in chosen[1:]:
        if int(ti) - kept[-1] >= int(min_sep):
            kept.append(int(ti))
    return np.asarray(kept, dtype=int)


def _bin_edges(vals: Array, n_bins: int) -> Array:
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite position values")
    if vmin == vmax:
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, n_bins + 1)


def _digitize(vals: Array, edges: Array) -> Array:
    idx = np.digitize(vals, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def fit_rate_maps(
    x: Array,
    y: Array,
    t: Array,
    counts: Array,
    train_idx: Array,
    n_x_bins: int,
    n_y_bins: int,
    smoothing_sigma_bins: float,
    occupancy_epsilon_s: float,
    rate_floor_hz: float,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Estimate per-cell 2D firing-rate maps from training data.

    Returns:
        rate_maps_hz: (n_cells, n_x_bins, n_y_bins)
        occupancy_s: (n_x_bins, n_y_bins)
        x_edges: (n_x_bins+1,)
        y_edges: (n_y_bins+1,)
        dt_train: scalar-like array for convenience
    """
    x_edges = _bin_edges(x[train_idx], n_x_bins)
    y_edges = _bin_edges(y[train_idx], n_y_bins)

    dt = float(np.median(np.diff(t)))
    x_bin = _digitize(x, x_edges)
    y_bin = _digitize(y, y_edges)

    occupancy = np.zeros((n_x_bins, n_y_bins), dtype=float)
    for ti in train_idx:
        occupancy[x_bin[ti], y_bin[ti]] += dt
    occupancy += occupancy_epsilon_s
    occupancy = gaussian_filter(occupancy, sigma=smoothing_sigma_bins)

    n_cells = counts.shape[0]
    rate_maps = np.zeros((n_cells, n_x_bins, n_y_bins), dtype=float)

    for ci in range(n_cells):
        spike_map = np.zeros((n_x_bins, n_y_bins), dtype=float)
        for ti in train_idx:
            spike_map[x_bin[ti], y_bin[ti]] += counts[ci, ti]
        spike_map = gaussian_filter(spike_map, sigma=smoothing_sigma_bins)
        rate = spike_map / occupancy
        rate_maps[ci] = np.maximum(rate, rate_floor_hz)

    return rate_maps, occupancy, x_edges, y_edges, np.array([dt], dtype=float)


def _systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _poisson_log_likelihood_per_particle(count_t: Array, lam_dt_per_particle: Array) -> Array:
    # count_t shape: (n_cells,), lam_dt_per_particle: (n_particles, n_cells)
    lam = np.maximum(lam_dt_per_particle, 1e-12)
    return np.sum(count_t[None, :] * np.log(lam) - lam, axis=1)


def _init_particles_from_occupancy(
    occupancy: Array,
    x_edges: Array,
    y_edges: Array,
    n_particles: int,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Array]:
    probs = occupancy.ravel() / np.sum(occupancy)
    flat_idx = rng.choice(np.arange(probs.size), size=n_particles, p=probs)
    bx = flat_idx // occupancy.shape[1]
    by = flat_idx % occupancy.shape[1]

    x_low, x_high = x_edges[bx], x_edges[bx + 1]
    y_low, y_high = y_edges[by], y_edges[by + 1]
    px = x_low + (x_high - x_low) * rng.random(n_particles)
    py = y_low + (y_high - y_low) * rng.random(n_particles)
    w = np.full(n_particles, 1.0 / n_particles, dtype=float)
    return px, py, w


def decode_particle_filter(
    counts: Array,
    x: Array,
    y: Array,
    t: Array,
    hd: Array,
    speed: Array,
    test_idx: Array,
    rate_maps_hz: Array,
    x_edges: Array,
    y_edges: Array,
    occupancy: Array,
    cfg: DecoderConfig,
    use_motion_prior: bool,
) -> DecodeResult:
    """Decode position on test interval using particle filtering."""
    rng = np.random.default_rng(cfg.random_seed)
    dt = float(np.median(np.diff(t)))

    px, py, w = _init_particles_from_occupancy(
        occupancy=occupancy,
        x_edges=x_edges,
        y_edges=y_edges,
        n_particles=cfg.n_particles,
        rng=rng,
    )

    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    x_hat = np.zeros(len(test_idx), dtype=float)
    y_hat = np.zeros(len(test_idx), dtype=float)

    for j, ti in enumerate(test_idx):
        # 1) Predict
        if use_motion_prior and ti > 0:
            dx = speed[ti - 1] * dt * np.cos(hd[ti - 1])
            dy = speed[ti - 1] * dt * np.sin(hd[ti - 1])
        else:
            dx, dy = 0.0, 0.0

        px = px + dx + rng.normal(0.0, cfg.process_noise_cm, size=cfg.n_particles)
        py = py + dy + rng.normal(0.0, cfg.process_noise_cm, size=cfg.n_particles)

        px = np.clip(px, x_min, x_max)
        py = np.clip(py, y_min, y_max)

        # 2) Update with spike likelihood
        bx = _digitize(px, x_edges)
        by = _digitize(py, y_edges)

        lam_hz = rate_maps_hz[:, bx, by].T  # (n_particles, n_cells)
        lam_dt = lam_hz * dt
        ll = _poisson_log_likelihood_per_particle(counts[:, ti], lam_dt)

        ll -= np.max(ll)
        w = np.exp(ll) * w
        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.full(cfg.n_particles, 1.0 / cfg.n_particles)
        else:
            w /= w_sum

        # 3) Decode
        x_hat[j] = float(np.sum(w * px))
        y_hat[j] = float(np.sum(w * py))

        # 4) Resample when particle degeneracy is high
        ess = 1.0 / np.sum(w**2)
        if ess < 0.5 * cfg.n_particles:
            idx = _systematic_resample(w, rng)
            px = px[idx]
            py = py[idx]
            w = np.full(cfg.n_particles, 1.0 / cfg.n_particles)

    err = np.sqrt((x_hat - x[test_idx]) ** 2 + (y_hat - y[test_idx]) ** 2)
    return DecodeResult(
        x_hat=x_hat,
        y_hat=y_hat,
        error_cm=err,
        rmse_cm=float(np.sqrt(np.mean(err**2))),
        median_cm=float(np.median(err)),
    )


def decode_particle_filter_with_snapshots(
    counts: Array,
    x: Array,
    y: Array,
    t: Array,
    hd: Array,
    speed: Array,
    test_idx: Array,
    snapshot_idx: Array,
    rate_maps_hz: Array,
    x_edges: Array,
    y_edges: Array,
    occupancy: Array,
    cfg: DecoderConfig,
    use_motion_prior: bool,
) -> Tuple[DecodeResult, List[ParticleSnapshot]]:
    """Decode and capture particle clouds at selected time indices.

    Snapshots are taken *after* the motion prior step and *after* the observation update
    (before any resampling happens).
    """
    snapshot_set = set(int(v) for v in np.asarray(snapshot_idx, dtype=int).reshape(-1))

    rng = np.random.default_rng(cfg.random_seed)
    dt = float(np.median(np.diff(t)))

    px, py, w = _init_particles_from_occupancy(
        occupancy=occupancy,
        x_edges=x_edges,
        y_edges=y_edges,
        n_particles=cfg.n_particles,
        rng=rng,
    )

    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    x_hat = np.zeros(len(test_idx), dtype=float)
    y_hat = np.zeros(len(test_idx), dtype=float)
    snapshots: List[ParticleSnapshot] = []

    for j, ti in enumerate(test_idx):
        want_snapshot = int(ti) in snapshot_set
        if want_snapshot:
            prev_px = px.copy()
            prev_py = py.copy()
            prev_w = w.copy()

        # 1) Predict
        if use_motion_prior and ti > 0:
            dx = speed[ti - 1] * dt * np.cos(hd[ti - 1])
            dy = speed[ti - 1] * dt * np.sin(hd[ti - 1])
        else:
            dx, dy = 0.0, 0.0

        px = px + dx + rng.normal(0.0, cfg.process_noise_cm, size=cfg.n_particles)
        py = py + dy + rng.normal(0.0, cfg.process_noise_cm, size=cfg.n_particles)

        px = np.clip(px, x_min, x_max)
        py = np.clip(py, y_min, y_max)

        if want_snapshot:
            prior_px = px.copy()
            prior_py = py.copy()

        # 2) Update with spike likelihood
        bx = _digitize(px, x_edges)
        by = _digitize(py, y_edges)

        lam_hz = rate_maps_hz[:, bx, by].T  # (n_particles, n_cells)
        lam_dt = lam_hz * dt
        ll = _poisson_log_likelihood_per_particle(counts[:, ti], lam_dt)

        ll -= np.max(ll)
        w = np.exp(ll) * w
        w_sum = np.sum(w)
        if w_sum <= 0 or not np.isfinite(w_sum):
            w = np.full(cfg.n_particles, 1.0 / cfg.n_particles)
        else:
            w /= w_sum

        # 3) Decode
        x_hat[j] = float(np.sum(w * px))
        y_hat[j] = float(np.sum(w * py))

        # 4) Resample when particle degeneracy is high
        ess = 1.0 / np.sum(w**2)
        did_resample = ess < 0.5 * cfg.n_particles

        if want_snapshot:
            snapshots.append(
                ParticleSnapshot(
                    ti=int(ti),
                    t_s=float(t[ti]),
                    true_x=float(x[ti]),
                    true_y=float(y[ti]),
                    hd_rad=float(hd[ti]),
                    speed=float(speed[ti]),
                    dx_motion=float(dx),
                    dy_motion=float(dy),
                    prev_px=prev_px,
                    prev_py=prev_py,
                    prev_w=prev_w,
                    prior_px=prior_px,
                    prior_py=prior_py,
                    post_w=w.copy(),
                    x_hat=float(x_hat[j]),
                    y_hat=float(y_hat[j]),
                    ess=float(ess),
                    did_resample=bool(did_resample),
                )
            )

        if did_resample:
            idx = _systematic_resample(w, rng)
            px = px[idx]
            py = py[idx]
            w = np.full(cfg.n_particles, 1.0 / cfg.n_particles)

    err = np.sqrt((x_hat - x[test_idx]) ** 2 + (y_hat - y[test_idx]) ** 2)
    result = DecodeResult(
        x_hat=x_hat,
        y_hat=y_hat,
        error_cm=err,
        rmse_cm=float(np.sqrt(np.mean(err**2))),
        median_cm=float(np.median(err)),
    )
    return result, snapshots


def _poisson_nll(beta: Array, X: Array, y: Array, offset: Array) -> float:
    eta = offset + X @ beta
    eta = np.clip(eta, -30.0, 30.0)
    mu = np.exp(eta)
    return float(np.sum(mu - y * eta))


def _fit_poisson_glm_with_offset(X: Array, y: Array, offset: Array) -> Tuple[Array, float]:
    # Newton-Raphson / IRLS for small dense design matrices.
    p = X.shape[1]
    beta = np.zeros(p, dtype=float)
    reg = 1e-6

    for _ in range(40):
        eta = np.clip(offset + X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        grad = X.T @ (y - mu)
        W = np.clip(mu, 1e-9, 1e9)
        H = -(X.T @ (W[:, None] * X)) - reg * np.eye(p)

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        beta_new = beta - step
        if np.linalg.norm(beta_new - beta) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    ll = -_poisson_nll(beta, X, y, offset)
    return beta, float(ll)


def analyze_hd_speed_modulation(
    counts: Array,
    x: Array,
    y: Array,
    t: Array,
    hd: Array,
    speed: Array,
    rate_maps_hz: Array,
    x_edges: Array,
    y_edges: Array,
    idx: Array,
) -> ModulationResult:
    """Quantify speed/HD effects after accounting for position.

    Model per cell (time-binned Poisson):
        log E[n_t] = log(lambda_pos(x_t,y_t) * dt) +
                     beta_v * zspeed_t + beta_c * cos(hd_t) + beta_s * sin(hd_t)

    Returns coefficients and LL improvement over offset-only model.
    """
    dt = float(np.median(np.diff(t)))
    bx = _digitize(x, x_edges)
    by = _digitize(y, y_edges)

    zspeed = (speed - np.mean(speed[idx])) / (np.std(speed[idx]) + 1e-9)
    X = np.column_stack([zspeed[idx], np.cos(hd[idx]), np.sin(hd[idx])])

    n_cells = counts.shape[0]
    beta_speed = np.zeros(n_cells)
    beta_cos = np.zeros(n_cells)
    beta_sin = np.zeros(n_cells)
    ll0 = np.zeros(n_cells)
    ll1 = np.zeros(n_cells)

    for ci in range(n_cells):
        lam_hz = rate_maps_hz[ci, bx[idx], by[idx]]
        mu0 = np.maximum(lam_hz * dt, 1e-12)
        offset = np.log(mu0)
        y_cell = counts[ci, idx].astype(float)

        # Offset-only model has beta=0
        ll0[ci] = -_poisson_nll(np.zeros(3), X * 0.0, y_cell, offset)

        beta, ll_full = _fit_poisson_glm_with_offset(X, y_cell, offset)
        beta_speed[ci], beta_cos[ci], beta_sin[ci] = beta
        ll1[ci] = ll_full

    hd_pref = np.arctan2(beta_sin, beta_cos)
    hd_strength = np.sqrt(beta_cos**2 + beta_sin**2)

    return ModulationResult(
        cell_ids=[str(i) for i in range(n_cells)],
        beta_speed=beta_speed,
        beta_cos_hd=beta_cos,
        beta_sin_hd=beta_sin,
        hd_pref_rad=hd_pref,
        hd_strength=hd_strength,
        ll_offset_only=ll0,
        ll_full=ll1,
        delta_ll=ll1 - ll0,
    )


def run_decode_comparison(
    spikes_by_cell: Dict[str, Array],
    x: Iterable[float],
    y: Iterable[float],
    t: Iterable[float],
    hd: Iterable[float],
    speed: Iterable[float],
    cfg: DecoderConfig | None = None,
    train_fraction: float = 0.7,
) -> dict:
    """Train/evaluate spikes-only vs spikes+motion-prior decoding.

    Returns a dictionary with decode results and modulation summary.
    """
    if cfg is None:
        cfg = DecoderConfig()

    x, y, t, hd, speed, cell_ids = validate_inputs(spikes_by_cell, x, y, t, hd, speed)
    counts = binned_spike_counts(spikes_by_cell, t, cell_ids)
    train_idx, test_idx = make_train_test_split(t, train_fraction=train_fraction)

    rate_maps_hz, occupancy, x_edges, y_edges, _ = fit_rate_maps(
        x=x,
        y=y,
        t=t,
        counts=counts,
        train_idx=train_idx,
        n_x_bins=cfg.n_x_bins,
        n_y_bins=cfg.n_y_bins,
        smoothing_sigma_bins=cfg.rate_smoothing_sigma_bins,
        occupancy_epsilon_s=cfg.occupancy_epsilon_s,
        rate_floor_hz=cfg.rate_floor_hz,
    )

    spikes_only = decode_particle_filter(
        counts=counts,
        x=x,
        y=y,
        t=t,
        hd=hd,
        speed=speed,
        test_idx=test_idx,
        rate_maps_hz=rate_maps_hz,
        x_edges=x_edges,
        y_edges=y_edges,
        occupancy=occupancy,
        cfg=cfg,
        use_motion_prior=False,
    )

    spikes_plus_motion = decode_particle_filter(
        counts=counts,
        x=x,
        y=y,
        t=t,
        hd=hd,
        speed=speed,
        test_idx=test_idx,
        rate_maps_hz=rate_maps_hz,
        x_edges=x_edges,
        y_edges=y_edges,
        occupancy=occupancy,
        cfg=cfg,
        use_motion_prior=True,
    )

    modulation = analyze_hd_speed_modulation(
        counts=counts,
        x=x,
        y=y,
        t=t,
        hd=hd,
        speed=speed,
        rate_maps_hz=rate_maps_hz,
        x_edges=x_edges,
        y_edges=y_edges,
        idx=test_idx,
    )

    return {
        "cell_ids": cell_ids,
        "counts": counts,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "occupancy": occupancy,
        "rate_maps_hz": rate_maps_hz,
        "spikes_only": spikes_only,
        "spikes_plus_motion": spikes_plus_motion,
        "modulation": modulation,
    }


def summary_table(results: dict) -> str:
    a = results["spikes_only"]
    b = results["spikes_plus_motion"]
    improve_rmse = 100.0 * (a.rmse_cm - b.rmse_cm) / max(a.rmse_cm, 1e-9)
    improve_med = 100.0 * (a.median_cm - b.median_cm) / max(a.median_cm, 1e-9)

    lines = [
        "Decoder comparison:",
        f"- spikes only:      RMSE={a.rmse_cm:.2f} cm, median={a.median_cm:.2f} cm",
        f"- spikes+HD+speed:  RMSE={b.rmse_cm:.2f} cm, median={b.median_cm:.2f} cm",
        f"- improvement:      RMSE={improve_rmse:.1f}%, median={improve_med:.1f}%",
    ]

    mod = results["modulation"]
    lines += [
        "Modulation (test period):",
        f"- median |beta_speed|: {np.median(np.abs(mod.beta_speed)):.3f}",
        f"- median HD strength:  {np.median(mod.hd_strength):.3f}",
        f"- median delta LL:     {np.median(mod.delta_ll):.3f}",
    ]
    return "\n".join(lines)


def plot_particle_filter_snapshots(
    snapshots: List[ParticleSnapshot],
    x: Array,
    y: Array,
    title: str | None = None,
    particle_size: float = 4.0,
    alpha: float = 0.25,
    weight_floor: float = 1e-12,
    ellipse_std: float = 2.0,
):
    """Plot prev cloud, motion prior, and observation-weighted cloud for each snapshot."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if len(snapshots) == 0:
        raise ValueError("snapshots is empty")

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    def _norm_w(w: Array) -> Array:
        w = np.asarray(w, dtype=float)
        s = float(np.sum(w))
        if s <= 0 or not np.isfinite(s):
            return np.full_like(w, 1.0 / max(len(w), 1), dtype=float)
        return w / s

    def _mean_cov(px: Array, py: Array, w: Array) -> Tuple[float, float, Array]:
        w = _norm_w(w)
        mx = float(np.sum(w * px))
        my = float(np.sum(w * py))
        dx = px - mx
        dy = py - my
        cov_xx = float(np.sum(w * dx * dx))
        cov_xy = float(np.sum(w * dx * dy))
        cov_yy = float(np.sum(w * dy * dy))
        return mx, my, np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)

    def _add_ellipse(ax, mx: float, my: float, cov: Array):
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        if np.any(~np.isfinite(evals)) or evals[0] <= 0 or evals[1] <= 0:
            return
        angle = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
        width = 2.0 * float(ellipse_std) * float(np.sqrt(evals[0]))
        height = 2.0 * float(ellipse_std) * float(np.sqrt(evals[1]))
        ell = Ellipse(
            (mx, my),
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.add_patch(ell)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    pad = 0.02 * max(xmax - xmin, ymax - ymin)
    xlim = (xmin - pad, xmax + pad)
    ylim = (ymin - pad, ymax + pad)

    logw_all = np.concatenate([np.log(s.post_w + weight_floor) for s in snapshots])
    vmin = float(np.percentile(logw_all, 5))
    vmax = float(np.percentile(logw_all, 99))

    n = len(snapshots)
    fig, axes = plt.subplots(
        nrows=n, ncols=3, figsize=(12, 2.6 * n), squeeze=False, constrained_layout=True
    )
    if title:
        fig.suptitle(title)

    col_titles = ["Prev posterior", "Motion prior (predict)", "Observation update (weights)"]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct)

    sc = None
    for r, s in enumerate(snapshots):
        ax0, ax1, ax2 = axes[r]

        # Prev posterior (before predict)
        mx0, my0, _ = _mean_cov(s.prev_px, s.prev_py, s.prev_w)
        ax0.scatter(s.prev_px, s.prev_py, s=particle_size, alpha=alpha, color="0.4")
        ax0.scatter([mx0], [my0], s=40, color="tab:red", marker="o")
        if s.ti > 0:
            ax0.scatter([x[s.ti - 1]], [y[s.ti - 1]], s=60, color="k", marker="x")
        ax0.set_xlim(*xlim)
        ax0.set_ylim(*ylim)
        ax0.set_aspect("equal", adjustable="box")
        ax0.text(0.01, 0.99, f"ti={s.ti}, t={s.t_s:.2f}s", transform=ax0.transAxes, va="top")

        # Motion prior (after predict)
        mx1, my1, _ = _mean_cov(s.prior_px, s.prior_py, s.prev_w)
        ax1.scatter(s.prior_px, s.prior_py, s=particle_size, alpha=alpha, color="0.4")
        ax1.scatter([mx1], [my1], s=40, color="tab:blue", marker="o")
        ax1.scatter([s.true_x], [s.true_y], s=60, color="k", marker="x")
        ax1.arrow(
            mx0,
            my0,
            s.dx_motion,
            s.dy_motion,
            color="tab:blue",
            width=0.0,
            head_width=1.5,
            length_includes_head=True,
            alpha=0.9,
        )
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.set_aspect("equal", adjustable="box")
        ax1.text(
            0.01,
            0.99,
            f"dx={s.dx_motion:.2f}, dy={s.dy_motion:.2f}",
            transform=ax1.transAxes,
            va="top",
        )

        # Observation update (weights after update)
        logw = np.log(s.post_w + weight_floor)
        sc = ax2.scatter(
            s.prior_px,
            s.prior_py,
            s=particle_size,
            c=logw,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            alpha=0.9,
        )
        mx2, my2, cov2 = _mean_cov(s.prior_px, s.prior_py, s.post_w)
        ax2.scatter([s.true_x], [s.true_y], s=60, color="k", marker="x")
        ax2.scatter([s.x_hat], [s.y_hat], s=40, color="cyan", marker="o")
        _add_ellipse(ax2, mx2, my2, cov2)
        ax2.set_xlim(*xlim)
        ax2.set_ylim(*ylim)
        ax2.set_aspect("equal", adjustable="box")

        sx = float(np.sqrt(max(cov2[0, 0], 0.0)))
        sy = float(np.sqrt(max(cov2[1, 1], 0.0)))
        ax2.text(
            0.01,
            0.99,
            f"ESS={s.ess:.0f}, sd=({sx:.1f},{sy:.1f}) cm",
            transform=ax2.transAxes,
            va="top",
            color="white",
        )
        ax2.text(
            0.01,
            0.90,
            f"resample={int(s.did_resample)}",
            transform=ax2.transAxes,
            va="top",
            color="white",
        )

    if sc is not None:
        fig.colorbar(sc, ax=axes[:, 2], shrink=0.8, label="log(weight)")
    return fig, axes
