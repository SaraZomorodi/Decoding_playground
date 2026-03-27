from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Lighting = Literal["light", "dark"]
Axis = Literal["x", "y", "either", "both"]
Direction = Literal["increasing", "decreasing"]


@dataclass(frozen=True)
class Trajectory:
    t_s: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Segment:
    start_idx: int
    end_idx: int
    start_s: float
    end_s: float
    axis: Axis
    direction: Direction


def _smooth_1d(x: np.ndarray, window_samples: int) -> np.ndarray:
    window_samples = int(window_samples)
    if window_samples <= 1:
        return x
    if window_samples % 2 == 0:
        window_samples += 1
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def find_monotonic_speed_segments(
    traj: Trajectory,
    *,
    axis: Axis = "either",
    direction: Direction = "increasing",
    speed_min: float = 3.0,
    speed_max: float = 100.0,
    smooth_s: float = 0.05,
    delta_tol: float = 0.0,
    min_duration_s: float = 1.0,
    min_displacement: float = 0.0,
) -> list[Segment]:
    """
    Find contiguous segments where:
      - speed_min < speed < speed_max
      - selected coordinate(s) change monotonically (no turn-back) per-step

    Parameters
    ----------
    axis:
      "x" / "y": monotonic along that axis only
      "either": monotonic along x OR y (per-step)
      "both": monotonic along x AND y (per-step)
    direction:
      "increasing" or "decreasing"
    smooth_s:
      Moving-average smoothing (seconds) applied before checking monotonicity.
    delta_tol:
      Per-step tolerance in coordinate units (e.g. cm). Use >0 to ignore tiny jitter.
    min_displacement:
      Minimum net displacement over the segment (axis-specific for "x"/"y", Euclidean otherwise).
    """

    t = np.asarray(traj.t_s, dtype=float)
    x = np.asarray(traj.x, dtype=float)
    y = np.asarray(traj.y, dtype=float)

    if t.ndim != 1:
        raise ValueError("Expected 1D time array.")
    if not (t.shape == x.shape == y.shape):
        raise ValueError("t_s, x, y must have the same shape.")
    if len(t) < 2:
        return []

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Time must be strictly increasing.")

    dx_raw = np.diff(x)
    dy_raw = np.diff(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.hypot(dx_raw, dy_raw) / dt
    speed_ok = (speed > float(speed_min)) & (speed < float(speed_max))

    dt_median = float(np.median(dt))
    smooth_samples = int(round(float(smooth_s) / dt_median)) if smooth_s > 0 else 1
    x_s = _smooth_1d(x, smooth_samples)
    y_s = _smooth_1d(y, smooth_samples)
    dx = np.diff(x_s)
    dy = np.diff(y_s)

    tol = float(delta_tol)
    if direction == "increasing":
        x_dir = dx > tol
        y_dir = dy > tol
    else:
        x_dir = dx < -tol
        y_dir = dy < -tol

    if axis == "x":
        dir_ok = x_dir
    elif axis == "y":
        dir_ok = y_dir
    elif axis == "either":
        dir_ok = x_dir | y_dir
    elif axis == "both":
        dir_ok = x_dir & y_dir
    else:
        raise ValueError(f"Unknown axis: {axis!r}")

    ok = speed_ok & dir_ok
    if not np.any(ok):
        return []

    ok_i8 = ok.astype(np.int8)
    edges = np.diff(np.concatenate([[0], ok_i8, [0]]))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1) - 1  # inclusive, in step-index space

    segments: list[Segment] = []
    min_duration_s = float(min_duration_s)
    min_displacement = float(min_displacement)

    for s_step, e_step in zip(run_starts, run_ends, strict=True):
        start_idx = int(s_step)
        end_idx = int(e_step + 1)  # convert steps -> sample index
        start_s = float(t[start_idx])
        end_s = float(t[end_idx])
        if end_s - start_s < min_duration_s:
            continue

        if min_displacement > 0:
            if axis == "x":
                disp = float(x[end_idx] - x[start_idx])
            elif axis == "y":
                disp = float(y[end_idx] - y[start_idx])
            else:
                disp = float(np.hypot(x[end_idx] - x[start_idx], y[end_idx] - y[start_idx]))
            if direction == "increasing":
                if disp < min_displacement:
                    continue
            else:
                if disp > -min_displacement:
                    continue

        segments.append(
            Segment(
                start_idx=start_idx,
                end_idx=end_idx,
                start_s=start_s,
                end_s=end_s,
                axis=axis,
                direction=direction,
            )
        )

    return segments


def ensure_mplconfigdir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = os.path.join(tempfile.gettempdir(), "mplconfig")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache_dir


def load_rat_trajectory(rat_n: int, lighting: Lighting, *, filter_speed: bool = False) -> Trajectory:
    from utils import RAT

    rat = RAT(rat_n, lighting, filter_speed=filter_speed)
    return Trajectory(
        t_s=np.asarray(rat.T, dtype=float),
        x=np.asarray(rat.X, dtype=float),
        y=np.asarray(rat.Y, dtype=float),
    )


def clip_trajectory(traj: Trajectory, *, start_s: float = 0.0, stop_s: float | None = None) -> Trajectory:
    if traj.t_s.ndim != 1:
        raise ValueError("Expected 1D time array.")
    if not (traj.t_s.shape == traj.x.shape == traj.y.shape):
        raise ValueError("t_s, x, y must have the same shape.")

    t0 = float(traj.t_s[0])
    start_s = float(start_s)
    stop_s = float(traj.t_s[-1] if stop_s is None else stop_s)
    if start_s < t0:
        raise ValueError(f"start_s must be >= {t0:.6g}. Got {start_s}.")
    if stop_s <= start_s:
        raise ValueError(f"stop_s must be > start_s. Got start_s={start_s}, stop_s={stop_s}.")

    i0 = int(np.searchsorted(traj.t_s, start_s, side="left"))
    i1 = int(np.searchsorted(traj.t_s, stop_s, side="right"))
    i0 = max(0, min(i0, len(traj.t_s) - 1))
    i1 = max(i0 + 1, min(i1, len(traj.t_s)))
    return Trajectory(t_s=traj.t_s[i0:i1], x=traj.x[i0:i1], y=traj.y[i0:i1])


def _robust_square_limits(x: np.ndarray, y: np.ndarray, *, pad_frac: float = 0.05) -> tuple[float, float, float, float]:
    x_min, x_max = np.nanpercentile(x, [1, 99]).astype(float)
    y_min, y_max = np.nanpercentile(y, [1, 99]).astype(float)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half_span = max(x_max - x_min, y_max - y_min) / 2.0
    half_span *= 1.0 + float(pad_frac)
    if not np.isfinite(half_span) or half_span <= 0:
        half_span = 1.0
    return (cx - half_span, cx + half_span, cy - half_span, cy + half_span)


def make_trajectory_animation(
    traj: Trajectory,
    *,
    start_s: float = 0.0,
    stop_s: float | None = None,
    fps: int = 30,
    video_duration_s: float | None = 10.0,
    trail_s: float | None = 5.0,
    arena_limits: tuple[float, float, float, float] | None = None,
    title: str | None = None,
):
    """
    Returns a Matplotlib animation for a trajectory segment.

    - start_s/stop_s: time window in seconds.
    - video_duration_s: output duration; controls playback speed (segment length / video_duration_s).
    - trail_s: how many seconds of recent path to draw; set None to draw full path so far.
    """

    ensure_mplconfigdir()
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if fps <= 0:
        raise ValueError("fps must be > 0.")

    seg = clip_trajectory(traj, start_s=start_s, stop_s=stop_s)
    seg_duration_s = float(seg.t_s[-1] - seg.t_s[0])
    if seg_duration_s <= 0:
        raise ValueError("Trajectory segment has non-positive duration.")

    if video_duration_s is None:
        video_duration_s = seg_duration_s
    video_duration_s = float(video_duration_s)
    if video_duration_s <= 0:
        raise ValueError("video_duration_s must be > 0.")

    n_frames = max(2, int(np.ceil(video_duration_s * fps)))
    frame_times = np.linspace(seg.t_s[0], seg.t_s[-1], n_frames)
    frame_indices = np.searchsorted(seg.t_s, frame_times, side="left")
    frame_indices = np.clip(frame_indices, 0, len(seg.t_s) - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)

    if arena_limits is None:
        xmin, xmax, ymin, ymax = _robust_square_limits(seg.x, seg.y)
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in arena_limits)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    (trail_line,) = ax.plot([], [], lw=2, color="tab:blue")
    (dot,) = ax.plot([], [], "o", ms=6, color="tab:red")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def init():
        trail_line.set_data([], [])
        dot.set_data([], [])
        time_text.set_text("")
        return trail_line, dot, time_text

    def update(frame_i: int):
        idx = int(frame_indices[frame_i])
        t_now = float(seg.t_s[idx])

        if trail_s is None:
            j0 = 0
        else:
            j0 = int(np.searchsorted(seg.t_s, t_now - float(trail_s), side="left"))
            j0 = max(0, min(j0, idx))

        trail_line.set_data(seg.x[j0 : idx + 1], seg.y[j0 : idx + 1])
        dot.set_data([seg.x[idx]], [seg.y[idx]])
        time_text.set_text(f"t = {t_now:.2f} s")
        return trail_line, dot, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )
    return ani


def save_animation(ani, out_path: str | os.PathLike, *, fps: int = 30, dpi: int = 150) -> Path:
    out_path = Path(out_path)
    ext = out_path.suffix.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".gif":
        ani.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    elif ext == ".mp4":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Saving .mp4 requires ffmpeg to be installed and on PATH.")
        ani.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        ani.save(str(out_path), fps=fps, dpi=dpi)
    return out_path


def _parse_limits(text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Expected 4 comma-separated numbers: xmin,xmax,ymin,ymax")
    vals = tuple(float(p) for p in parts)
    return vals  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a trajectory animation from rat tracking data.")
    parser.add_argument("--rat", type=int, default=1, help="Rat id (1..5 for the included dataset).")
    parser.add_argument("--lighting", type=str, choices=["light", "dark"], default="light")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (s).")
    parser.add_argument("--stop", type=float, default=None, help="Stop time (s). Default: end of recording.")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS.")
    parser.add_argument(
        "--video-duration",
        type=float,
        default=10.0,
        help="Output duration (s). Controls playback speed.",
    )
    parser.add_argument(
        "--trail",
        type=float,
        default=5.0,
        help="Trail length (s). Use 0 for a dot-only video; use -1 to draw full path.",
    )
    parser.add_argument("--out", type=str, default="trajectory.gif", help="Output path (.gif or .mp4).")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--filter-speed", action="store_true", help="Filter tracking points by speed.")
    parser.add_argument(
        "--limits",
        type=_parse_limits,
        default=None,
        help="Axis limits as xmin,xmax,ymin,ymax (comma-separated).",
    )

    args = parser.parse_args(argv)

    ensure_mplconfigdir()
    import matplotlib

    matplotlib.use("Agg")

    traj = load_rat_trajectory(args.rat, args.lighting, filter_speed=args.filter_speed)
    trail_s = None if args.trail < 0 else float(args.trail)
    ani = make_trajectory_animation(
        traj,
        start_s=args.start,
        stop_s=args.stop,
        fps=args.fps,
        video_duration_s=args.video_duration,
        trail_s=trail_s,
        arena_limits=args.limits,
        title=args.title,
    )
    save_animation(ani, args.out, fps=args.fps, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
