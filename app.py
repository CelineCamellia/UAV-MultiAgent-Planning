from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from core.planner import plot_mission, run_mission
from mappo_adapter import generate_3d_simulation_gif


st.set_page_config(
    page_title="UAV Agent",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state():
    defaults = {
        "mission_result": None,
        "mission_fig": None,
        "gif_path": None,
        "gif_meta": None,
        "logs": [],
        "run_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_log(text: str):
    st.session_state.logs.append(text)


def show_animated_gif(gif_path: str):
    path = Path(gif_path)
    gif_bytes = path.read_bytes()
    encoded = base64.b64encode(gif_bytes).decode("utf-8")
    st.markdown(
        f"""
        <div style="width:100%; text-align:center;">
            <img
                src="data:image/gif;base64,{encoded}"
                style="width:100%; max-width:980px; border-radius:12px;"
                alt="3D cooperative simulation"
            />
            <div style="color:#64748b; font-size:1rem; margin-top:0.5rem;">
                三维多无人机编队飞行 / 避障 / 高度控制
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_state()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.75rem; padding-bottom: 2rem;}
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 16px 18px;
        color: white;
        min-height: 102px;
    }
    .metric-card h3 {font-size: 0.95rem; margin: 0 0 8px 0; color: #cbd5e1;}
    .metric-card p {font-size: 1.35rem; margin: 0; font-weight: 700;}
    .main-title {
        font-size: 3.1rem;
        line-height: 1.3;
        font-weight: 800;
        color: #2b2f3a;
        margin: 0.55rem 0 1.05rem 0;
        overflow: visible;
    }
    .main-title .prefix {
        display: block;
        white-space: nowrap;
        line-height: 1.32;
        padding-top: 0.18rem;
        margin-bottom: 0.5rem;
        overflow: visible;
    }
    .main-title .title-cn {
        display: block;
        line-height: 1.22;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title"><span class="prefix">🚁 UAV Agent:</span><span class="title-cn">无人机多智能体任务规划与三维协同仿真系统</span></div>',
    unsafe_allow_html=True,
)
st.caption("完成任务解析、路径规划、动态重规划、安全校验和三维协同仿真。")

with st.sidebar:
    st.header("控制台")
    scene_id = st.slider("场景编号", min_value=0, max_value=5, value=0, step=1)
    dynamic_obstacle = st.toggle("加入动态禁飞区并触发重规划", value=False)
    max_steps = st.slider("三维动画帧数", min_value=40, max_value=140, value=90, step=10)
    st.divider()
    if st.button("清空页面状态 / 重新运行", use_container_width=True):
        for key in ("mission_result", "mission_fig", "gif_path", "gif_meta", "logs"):
            st.session_state[key] = [] if key == "logs" else None
        st.session_state.run_count = 0
        st.rerun()

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown('<div class="metric-card"><h3>主入口</h3><p>app.py</p></div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="metric-card"><h3>核心链路</h3><p>Agent + A* + Safety</p></div>', unsafe_allow_html=True)
with col_c:
    st.markdown('<div class="metric-card"><h3>三维仿真</h3><p>MAPPO GIF</p></div>', unsafe_allow_html=True)

st.subheader("1. 任务输入")
task_text = st.text_area(
    "输入自然语言任务",
    value="三架无人机从起点出发，执行区域巡检任务，避开禁飞区并保持安全距离，最终抵达目标区域。",
    height=92,
)

run_col, sim_col, replay_col = st.columns([1.1, 1.1, 0.9])
with run_col:
    run_clicked = st.button("🚀 运行任务规划", type="primary", use_container_width=True)
with sim_col:
    sim_clicked = st.button("🎥 生成 / 刷新三维仿真GIF", use_container_width=True)
with replay_col:
    all_clicked = st.button("▶ 一键运行完整流程", use_container_width=True)

if run_clicked or all_clicked:
    with st.spinner("正在执行任务解析、路径规划与安全校验..."):
        result = run_mission(task_text, dynamic_obstacle=dynamic_obstacle, scene_id=scene_id)
        fig = plot_mission(result)
        st.session_state.mission_result = result
        st.session_state.mission_fig = fig
        st.session_state.run_count += 1
        st.session_state.logs.extend(result.logs)
        add_log(f"第 {st.session_state.run_count} 次任务规划完成")
    st.success("任务规划完成。")

if sim_clicked or all_clicked:
    with st.spinner("正在生成三维协同仿真GIF，请稍等..."):
        if sim_clicked or st.session_state.mission_result is None:
            result = run_mission(task_text, dynamic_obstacle=dynamic_obstacle, scene_id=scene_id)
            fig = plot_mission(result)
            st.session_state.mission_result = result
            st.session_state.mission_fig = fig
            st.session_state.run_count += 1
            st.session_state.logs.extend(result.logs)
            add_log(f"第 {st.session_state.run_count} 次任务规划完成")

        gif_path, meta = generate_3d_simulation_gif(
            max_steps=max_steps,
            mission_result=st.session_state.mission_result,
        )
        st.session_state.gif_path = gif_path
        st.session_state.gif_meta = meta
        add_log(f"三维协同仿真GIF已生成：{Path(gif_path).name}")
    st.success("三维仿真GIF生成完成。")

left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("2. 二维路径规划与动态重规划")
    if st.session_state.mission_result is None:
        st.info("暂无路径规划结果。")
    else:
        st.pyplot(st.session_state.mission_fig, clear_figure=False)

with right:
    st.subheader("3. 任务解析与安全报告")
    if st.session_state.mission_result is None:
        st.info("暂无任务报告。")
    else:
        result = st.session_state.mission_result
        st.json(result.report)
        if result.safe:
            st.success("安全校验通过：路径未穿越障碍物/禁飞区。")
        else:
            st.error("安全校验未通过：当前约束下未找到可行路径。")

st.subheader("4. 三维协同仿真")
gif_box, meta_box = st.columns([1.35, 0.65])
with gif_box:
    if st.session_state.gif_path:
        show_animated_gif(st.session_state.gif_path)
    else:
        st.info("尚未生成三维仿真GIF。")
with meta_box:
    st.markdown("#### 仿真状态")
    if st.session_state.gif_meta:
        st.json(st.session_state.gif_meta)
    else:
        st.write("等待生成")

st.subheader("5. 系统日志")
if st.session_state.logs:
    st.code("\n".join(f"[{i + 1:02d}] {line}" for i, line in enumerate(st.session_state.logs)), language="text")
else:
    st.info("运行任务后会显示可解释日志。")
