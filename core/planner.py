from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


GridPoint = Tuple[int, int]


@dataclass
class MissionResult:
    task_text: str
    start: GridPoint
    goal: GridPoint
    obstacles: List[GridPoint]
    dynamic_obstacles: List[GridPoint]
    path: List[GridPoint]
    safe: bool
    scene_id: int
    report: Dict[str, object]
    logs: List[str]


def default_obstacles(scene_id: int = 0) -> List[GridPoint]:
    scene_id = int(scene_id) % 6
    wall1_x = 7 + (scene_id % 3)
    wall1_gap = 8 + (scene_id % 5)
    wall2_y = 6 + (scene_id % 3)
    wall2_gap_a = 15 + (scene_id % 4)
    wall2_gap_b = wall2_gap_a + 1
    wall3_x = 21 + ((scene_id + 1) % 3)
    wall3_gap = 13 + ((scene_id * 2) % 5)

    obstacles: List[GridPoint] = []
    for y in range(4, 16):
        if y != wall1_gap:
            obstacles.append((wall1_x, y))
    for x in range(12, 24):
        if x not in (wall2_gap_a, wall2_gap_b):
            obstacles.append((x, wall2_y))
    for y in range(10, 21):
        if y != wall3_gap:
            obstacles.append((wall3_x, y))
    return obstacles


def build_dynamic_no_fly_zone(base_path: List[GridPoint], start: GridPoint, goal: GridPoint) -> List[GridPoint]:
    """把动态禁飞区放在原路径中段附近，让重规划前后有明显差异。"""
    if len(base_path) < 10:
        return []

    mid = base_path[int(len(base_path) * 0.55)]
    cx, cy = mid
    candidates: List[GridPoint] = []

    # 以原路径中段为中心生成一个小块禁飞区，直接压住原最优路径。
    for dx in range(-2, 3):
        for dy in range(-1, 2):
            p = (cx + dx, cy + dy)
            if p not in (start, goal):
                candidates.append(p)

    # 再沿路径方向补几个点，保证视觉上能看出“路径被拦截”。
    for p in base_path[max(1, int(len(base_path) * 0.48)) : min(len(base_path) - 1, int(len(base_path) * 0.63))]:
        if p not in (start, goal):
            candidates.append(p)

    return sorted(set(candidates))


def parse_task(task_text: str) -> Dict[str, object]:
    text = task_text.strip() or "执行区域巡检任务，并在避开禁飞区后抵达目标点。"
    risk = "禁飞区" in text or "避障" in text or "安全" in text
    return {
        "mission_type": "区域巡检 / 协同路径规划",
        "input": text,
        "agents": 3,
        "start": [2, 2],
        "goal": [27, 20],
        "constraints": ["避开障碍物", "保持安全距离", "支持动态重规划"],
        "risk_level": "中" if risk else "低",
    }


def astar(
    start: GridPoint,
    goal: GridPoint,
    obstacles: List[GridPoint],
    width: int = 30,
    height: int = 24,
) -> List[GridPoint]:
    blocked = set(obstacles)

    def h(p: GridPoint) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_set: List[Tuple[int, int, GridPoint]] = [(h(start), 0, start)]
    came_from: Dict[GridPoint, Optional[GridPoint]] = {start: None}
    g_score: Dict[GridPoint, int] = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            path: List[GridPoint] = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in blocked:
                continue
            new_cost = cost + 1
            if nxt not in g_score or new_cost < g_score[nxt]:
                g_score[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(open_set, (new_cost + h(nxt), new_cost, nxt))
    return []


def run_mission(task_text: str, dynamic_obstacle: bool = False, scene_id: int = 0) -> MissionResult:
    logs = [
        "接收自然语言任务输入",
        "解析任务目标、起点、终点与安全约束",
        "构建二维栅格地图与障碍物集合",
    ]
    parsed = parse_task(task_text)
    start: GridPoint = tuple(parsed["start"])  # type: ignore[arg-type]
    goal: GridPoint = tuple(parsed["goal"])  # type: ignore[arg-type]
    static_obstacles = default_obstacles(scene_id=scene_id)
    base_path = astar(start, goal, static_obstacles)
    dynamic_obstacles: List[GridPoint] = []
    if dynamic_obstacle:
        dynamic_obstacles = build_dynamic_no_fly_zone(base_path, start, goal)
        logs.append("检测到动态禁飞区，原路径中段受阻，触发路径重规划")
    obstacles = sorted(set(static_obstacles + dynamic_obstacles))
    path = astar(start, goal, obstacles)
    safe = bool(path) and not any(p in set(obstacles) for p in path)
    logs.extend(
        [
            f"A* 规划完成，路径节点数：{len(path)}",
            "安全校验通过" if safe else "未找到安全路径，需要调整约束",
            "生成可解释任务报告",
        ]
    )
    return MissionResult(
        task_text=task_text,
        start=start,
        goal=goal,
        obstacles=obstacles,
        dynamic_obstacles=dynamic_obstacles,
        path=path,
        safe=safe,
        scene_id=scene_id,
        report={
            "任务类型": parsed["mission_type"],
            "无人机数量": parsed["agents"],
            "风险等级": parsed["risk_level"],
            "路径长度": len(path),
            "安全状态": "通过" if safe else "未通过",
            "是否重规划": "是" if dynamic_obstacle else "否",
            "场景编号": scene_id,
            "动态禁飞区数量": len(dynamic_obstacles),
        },
        logs=logs,
    )


def plot_mission(result: MissionResult):
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=140)
    ax.set_title("2D Mission Planning Result", fontsize=13, pad=10)
    ax.set_xlim(-1, 30)
    ax.set_ylim(-1, 24)
    ax.set_aspect("equal")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)

    dynamic_set = set(result.dynamic_obstacles)
    static_obstacles = [p for p in result.obstacles if p not in dynamic_set]

    if static_obstacles:
        ox, oy = zip(*static_obstacles)
        ax.scatter(ox, oy, c="#ef4444", marker="s", s=56, label="Obstacle / no-fly zone")

    if result.dynamic_obstacles:
        dx, dy = zip(*result.dynamic_obstacles)
        ax.scatter(dx, dy, c="#7c3aed", marker="s", s=78, label="Dynamic no-fly zone")

    if result.path:
        px, py = zip(*result.path)
        ax.plot(px, py, color="#2563eb", linewidth=2.5, label="Planned path")
        ax.scatter(px, py, color="#2563eb", s=12)

    ax.scatter([result.start[0]], [result.start[1]], c="#16a34a", s=130, marker="o", label="Start")
    ax.scatter([result.goal[0]], [result.goal[1]], c="#f59e0b", s=170, marker="*", label="Goal")
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    return fig
