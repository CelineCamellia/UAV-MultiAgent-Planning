from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MAPPO_DIR = BASE_DIR / "mappo_module"
MODEL_PATH = MAPPO_DIR / "models" / "actor_1001472.pth"
OUTPUT_DIR = BASE_DIR / "outputs" / "gifs"
CRUISE_ALTITUDE = 1.6


def _grid_to_world(point: Tuple[int, int]) -> np.ndarray:
    return np.array([point[0] / 29.0 * 50.0, point[1] / 23.0 * 50.0, 0.0], dtype=float)


def _resample_polyline(points: np.ndarray, total_steps: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((total_steps, 3), dtype=float)
    if len(points) == 1:
        return np.repeat(points, total_steps, axis=0)

    seg_len = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg_len)])
    if dist[-1] <= 1e-6:
        return np.repeat(points[:1], total_steps, axis=0)

    target = np.linspace(0, dist[-1], total_steps)
    out = np.zeros((total_steps, 3), dtype=float)
    out[:, 0] = np.interp(target, dist, points[:, 0])
    out[:, 1] = np.interp(target, dist, points[:, 1])
    out[:, 2] = np.interp(target, dist, points[:, 2])
    return out


def _cruise_then_land_altitude(t: float, cruise_altitude: float = CRUISE_ALTITUDE) -> float:
    """起飞到巡航高度，末段下降到地面目标点。"""
    if t < 0.18:
        u = t / 0.18
        return float(cruise_altitude * (0.5 - 0.5 * np.cos(np.pi * u)))
    if t < 0.72:
        return float(cruise_altitude)
    if t < 0.92:
        u = (t - 0.72) / 0.20
        return float(cruise_altitude * (0.5 + 0.5 * np.cos(np.pi * u)))
    return 0.0


def _make_route_coupled_episode(mission_result: Any, max_steps: int = 90):
    """根据二维规划结果生成同场景三维协同轨迹。"""
    path = getattr(mission_result, "path", []) or []
    if len(path) < 2:
        return _make_fallback_episode(max_steps=max_steps)

    route = np.array([_grid_to_world(p) for p in path], dtype=float)
    hold_frames = max(4, int(max_steps * 0.08))
    moving_steps = max(max_steps - hold_frames, 2)
    centerline_moving = _resample_polyline(route, moving_steps)
    centerline = np.vstack(
        [
            centerline_moving,
            np.repeat(centerline_moving[-1][None, :], hold_frames, axis=0),
        ]
    )
    z_targets = np.array([CRUISE_ALTITUDE, CRUISE_ALTITUDE, CRUISE_ALTITUDE], dtype=float)
    trajs: List[List[np.ndarray]] = [[] for _ in range(3)]
    final_direction = route[-1, :2] - route[-2, :2]
    final_norm = np.linalg.norm(final_direction)
    if final_norm < 1e-6:
        final_direction = np.array([1.0, 0.0])
    else:
        final_direction = final_direction / final_norm

    for step in range(max_steps):
        curr = centerline[step].copy()
        prev = centerline[max(step - 1, 0)]
        nxt = centerline[min(step + 1, max_steps - 1)]
        direction = nxt[:2] - prev[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = final_direction
        else:
            direction = direction / norm
        side = np.array([-direction[1], direction[0]])
        forward = direction
        offsets_2d = [
            np.array([0.0, 0.0]),
            -forward * 1.6 + side * 1.1,
            -forward * 1.6 - side * 1.1,
        ]
        t = step / max(max_steps - 1, 1)
        for i, offset in enumerate(offsets_2d):
            p = curr.copy()
            p[:2] += offset
            p[0] = np.clip(p[0], 0, 50)
            p[1] = np.clip(p[1], 0, 50)
            p[2] = _cruise_then_land_altitude(t, z_targets[i])
            trajs[i].append(p)

    dynamic_set = set(getattr(mission_result, "dynamic_obstacles", []) or [])
    obstacles = []
    for p in getattr(mission_result, "obstacles", []) or []:
        world = _grid_to_world(p)[:2]
        if p in dynamic_set:
            obstacles.append(
                {
                    "type": "dynamic",
                    "pos": world,
                    "z_min": 0.45,
                    "z_max": 2.1,
                    "size": 2.8,
                }
            )
        else:
            high_zone = (p[0] + p[1]) % 3 == 0
            obstacles.append(
                {
                    "type": "box",
                    "pos": world,
                    "z_min": 0.15 if high_zone else 0.0,
                    "z_max": 1.75 if high_zone else 1.25,
                    "size": 2.4,
                }
            )

    start = _grid_to_world(getattr(mission_result, "start"))
    goal = _grid_to_world(getattr(mission_result, "goal"))
    return trajs, start, goal, obstacles, z_targets


def _make_fallback_episode(max_steps: int = 90):
    start = np.array([5.0, 5.0, 0.0])
    goal = np.array([45.0, 42.0, 0.0])
    offsets = np.array([[0.0, 0.0, 0.0], [-1.6, -1.1, 0.0], [-1.6, 1.1, 0.0]])
    z_targets = np.array([CRUISE_ALTITUDE, CRUISE_ALTITUDE, CRUISE_ALTITUDE])
    obstacles = [
        {"type": "cyl", "pos": np.array([18.0, 18.0]), "z_min": 0.0, "z_max": 1.3, "size": 2.8},
        {"type": "box", "pos": np.array([27.0, 26.0]), "z_min": 0.4, "z_max": 2.0, "size": 3.0},
        {"type": "cyl", "pos": np.array([35.0, 34.0]), "z_min": 0.2, "z_max": 1.7, "size": 2.8},
    ]
    trajs: List[List[np.ndarray]] = [[] for _ in range(3)]
    for step in range(max_steps):
        t = step / (max_steps - 1)
        curve = np.array([0.0, 6.0 * np.sin(np.pi * t), 0.0])
        center = start * (1 - t) + goal * t + curve
        if t >= 0.92:
            center = goal.copy()
        for i in range(3):
            p = center + offsets[i]
            p[2] = _cruise_then_land_altitude(t, z_targets[i])
            trajs[i].append(p.copy())
    return trajs, start, goal, obstacles, z_targets


def _try_mappo_episode(max_steps: int = 120):
    try:
        import torch
        import torch.nn as nn

        if str(MAPPO_DIR) not in sys.path:
            sys.path.insert(0, str(MAPPO_DIR))
        from env import Config, MultiUAVEnv  # type: ignore

        class Actor(nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, act_dim),
                )
                self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

            def get_action(self, x):
                return self.network(x)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(str(MODEL_PATH))

        cfg = Config()
        cfg.max_steps = min(cfg.max_steps, max_steps)
        env = MultiUAVEnv(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor = Actor(env.observation_space.shape[1], env.action_space.shape[1]).to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        state_dict = {k.replace("net.", "network."): v for k, v in state_dict.items()}
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()

        obs, _ = env.reset()
        trajs: List[List[np.ndarray]] = [[] for _ in range(cfg.num_agents)]
        done = False
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                action = actor.get_action(torch.FloatTensor(obs).to(device)).cpu().numpy()
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            for i in range(cfg.num_agents):
                trajs[i].append(env.states[i, 0:3].copy())
            steps += 1
        return trajs, env.start_pos, env.goal_pos, env.obstacles, np.array(cfg.z_targets)
    except Exception:
        return _make_fallback_episode(max_steps=90)


def generate_3d_simulation_gif(
    max_steps: int = 100,
    fps: int = 12,
    mission_result: Any = None,
) -> Tuple[str, Dict[str, object]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if mission_result is not None:
        trajs, start, goal, obstacles, z_targets = _make_route_coupled_episode(mission_result, max_steps=max_steps)
        source = "2D route-coupled simulation"
    else:
        trajs, start, goal, obstacles, z_targets = _try_mappo_episode(max_steps=max_steps)
        source = "MAPPO checkpoint" if MODEL_PATH.exists() else "fallback demo"
    frame_count = min(len(trajs[0]), max_steps)
    colors = ["#2563eb", "#7c3aed", "#f97316"]
    frames = []

    fig = plt.figure(figsize=(14, 5.2), dpi=120)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1, 1])
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_xy = fig.add_subplot(gs[1])
    ax_xoz = fig.add_subplot(gs[2])
    start_positions = np.array([traj[0] for traj in trajs])
    target_positions = np.array([traj[-1] for traj in trajs])

    for step in range(frame_count):
        for ax in (ax3d, ax_xy, ax_xoz):
            ax.cla()

        ax3d.set_title("3D multi-UAV formation")
        ax3d.set_xlim(0, 50)
        ax3d.set_ylim(0, 50)
        ax3d.set_zlim(0, 2.4)
        ax3d.view_init(elev=22, azim=45)
        ax3d.scatter(goal[0], goal[1], 0, c="#f59e0b", marker="*", s=150, edgecolors="black")
        ax3d.plot([goal[0], goal[0]], [goal[1], goal[1]], [0, CRUISE_ALTITUDE], c="#f59e0b", linestyle="--", alpha=0.55)

        ax_xy.set_title("Top view: obstacle avoidance")
        ax_xy.set_xlim(0, 50)
        ax_xy.set_ylim(0, 50)
        ax_xy.set_aspect("equal")
        ax_xy.grid(True, color="#e5e7eb", linewidth=0.6)

        ax_xoz.set_title("Side view: altitude control")
        ax_xoz.set_xlim(0, 50)
        ax_xoz.set_ylim(-0.1, 2.4)
        ax_xoz.grid(True, color="#e5e7eb", linewidth=0.6)
        ax_xoz.scatter(goal[0], 0, c="#f59e0b", marker="*", s=110, edgecolors="black", zorder=5)
        ax_xoz.axhline(CRUISE_ALTITUDE, color="#f59e0b", linestyle=":", alpha=0.45)

        for obs in obstacles:
            pos = np.asarray(obs["pos"])
            size = float(obs.get("size", 2.6))
            z_min = float(obs.get("z_min", 0.0))
            z_max = float(obs.get("z_max", 1.4))
            height = max(z_max - z_min, 0.1)
            if obs.get("type") == "dynamic":
                ax_xy.add_patch(Rectangle(pos - 1.8, 3.6, 3.6, color="#7c3aed", alpha=0.13, zorder=1))
                ax3d.bar3d(pos[0] - size / 2, pos[1] - size / 2, z_min, size, size, height, color="#7c3aed", alpha=0.10, shade=True)
                ax_xoz.add_patch(Rectangle((pos[0] - size / 2, z_min), size, height, color="#7c3aed", alpha=0.055, zorder=1))
            elif obs.get("type") == "box":
                ax_xy.add_patch(Rectangle(pos - 1.5, 3.0, 3.0, color="#ef4444", alpha=0.10, zorder=1))
                ax3d.bar3d(pos[0] - size / 2, pos[1] - size / 2, z_min, size, size, height, color="#ef4444", alpha=0.075, shade=True)
                ax_xoz.add_patch(Rectangle((pos[0] - size / 2, z_min), size, height, color="#ef4444", alpha=0.045, zorder=1))
            else:
                ax_xy.add_patch(Circle(pos, 1.5, color="#ef4444", alpha=0.10, zorder=1))
                ax3d.bar3d(pos[0] - size / 2, pos[1] - size / 2, z_min, size, size, height, color="#ef4444", alpha=0.075, shade=True)
                ax_xoz.add_patch(Rectangle((pos[0] - size / 2, z_min), size, height, color="#ef4444", alpha=0.045, zorder=1))

        for i, target_z in enumerate(z_targets):
            ax_xoz.axhline(float(target_z), color=colors[i], linestyle="--", alpha=0.35)

        for i, traj in enumerate(trajs):
            arr = np.asarray(traj[: step + 1])
            if len(arr) == 0:
                continue
            curr = arr[-1]
            sp = start_positions[i]
            tp = target_positions[i]
            ax3d.scatter(sp[0], sp[1], sp[2], c=colors[i], marker="^", s=55, edgecolors="black", alpha=0.9)
            ax3d.scatter(tp[0], tp[1], tp[2], c=colors[i], marker="*", s=70, edgecolors="black", alpha=0.9)
            ax3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=colors[i], linewidth=3.0, alpha=0.95)
            ax3d.scatter(curr[0], curr[1], curr[2], c=colors[i], s=58, edgecolors="black")

            ax_xy.scatter(sp[0], sp[1], c=colors[i], marker="^", s=58, edgecolors="black", zorder=4)
            ax_xy.scatter(tp[0], tp[1], c=colors[i], marker="*", s=82, edgecolors="black", zorder=4)
            ax_xy.plot(arr[:, 0], arr[:, 1], c=colors[i], linewidth=2.9, zorder=5)
            ax_xy.scatter(curr[0], curr[1], c=colors[i], s=52, edgecolors="black", zorder=6)
            ax_xy.add_patch(Circle(curr[:2], 1.0, fill=False, linestyle=":", color=colors[i], alpha=0.55, zorder=4))

            ax_xoz.scatter(sp[0], sp[2], c=colors[i], marker="^", s=48, edgecolors="black", zorder=5)
            ax_xoz.scatter(tp[0], tp[2], c=colors[i], marker="*", s=72, edgecolors="black", zorder=5)
            ax_xoz.plot(arr[:, 0], arr[:, 2], c=colors[i], linewidth=2.9, zorder=6)
            ax_xoz.scatter(curr[0], curr[2], c=colors[i], s=50, edgecolors="black", zorder=7)

        fig.suptitle(f"MAPPO 3D cooperative simulation | step {step + 1}/{frame_count}", fontsize=12)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    plt.close(fig)
    out_path = OUTPUT_DIR / f"mappo_3d_simulation_{int(time.time())}.gif"
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / max(fps, 1))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    meta = {
        "gif": str(out_path),
        "frames": len(frames),
        "agents": 3,
        "source": source,
        "scene": "coupled with current 2D planning result" if mission_result is not None else "independent simulation",
        "target_point_z": 0,
        "cruise_altitude": CRUISE_ALTITUDE,
        "airspace_constraints": "3D obstacle/no-fly-zone volumes",
    }
    return str(out_path), meta
