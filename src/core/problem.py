"""
Problem 模块 — 场景加载，组装 Context。
"""

from src.core.context import Context
from src.core.config import EMConfig
from src.simple_scene_builder import build_simple_scene


def run(ctx: Context) -> Context:
    """加载场景并验证。"""
    ctx.scene = build_simple_scene()

    # 从场景天线对象同步配置
    ant = ctx.scene.get("antenna", {})
    extra = ant.get("extra", {})
    ant_cfg = extra.get("antenna_config", {})

    if ant_cfg:
        cfg = ctx.config
        cfg.antenna_type = ant_cfg.get("type", cfg.antenna_type)
        cfg.antenna_sigma_z = ant_cfg.get("sigma_z", cfg.antenna_sigma_z)
        cfg.antenna_tilt = ant_cfg.get("tilt_angle", cfg.antenna_tilt)
        cfg.antenna_patch_hpbw = ant_cfg.get("patch_hpbw", cfg.antenna_patch_hpbw)
        cfg.antenna_horn_hpbw = ant_cfg.get("horn_hpbw", cfg.antenna_horn_hpbw)
        freq_from_cfg = ant_cfg.get("frequency")
        if freq_from_cfg:
            cfg.frequency = freq_from_cfg
        pos = extra.get("position") or ant_cfg.get("position")
        if pos:
            cfg.antenna_pos = tuple(pos)

    return ctx
