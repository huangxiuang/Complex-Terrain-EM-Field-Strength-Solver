"""
Pipeline Scheduler — 按序调度 problem → prep → solver → post。
"""

from src.core.context import Context
from src.core.config import EMConfig
from src.core import problem
from src.engine import prep, solver, post


class Scheduler:
    """管道调度器。每个模块暴露统一的 run(ctx) → ctx 接口。"""

    def __init__(self):
        self._pipeline = [problem, prep, solver, post]

    def run(self, config: EMConfig = None,
            rx_points: list = None,
            scene: dict = None) -> list:
        """
        执行完整管道。

        参数
        ----
        config : EMConfig | None
        rx_points : list[(x,y,z)]
        scene : dict | None  若提供则跳过 problem 模块

        返回
        ----
        results : list[dict]  每个接收点的计算结果
        """
        ctx = Context(
            config=config or EMConfig(),
            rx_points=rx_points or [],
        )
        if scene is not None:
            ctx.scene = scene

        for module in self._pipeline:
            if module is problem and scene is not None:
                continue  # 外部已提供场景
            ctx = module.run(ctx)

        return ctx.results
