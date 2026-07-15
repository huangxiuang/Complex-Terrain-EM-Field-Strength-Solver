# EM Solver — 柱坐标系抛物方程（CPE）电磁场求解器

2.8 GHz 频段地波传播与障碍物绕射分析工具。基于 SSFT-PE（分步傅里叶变换抛物方程）方法，支持复杂地面材料、多层介质、人工障碍物。

## 快速开始

```bash
pip install -r requirements.txt
python run.py                    # 默认测试点
python run.py --rx 5,0,3        # 自定义接收点
python run.py --freq 3e9 --rx 0,0,6  # 自定义频率
python run.py --plot            # 生成可视化图
python run.py --help            # 全部选项
```

GUI 模式：
```bash
python src/main_window.py
```

## 核心物理模型

### 抛物方程（PE）

柱坐标系下的前向抛物方程，从天线位置向外步进求解：

```
∂u/∂r = j/(2k₀) · ∂²u/∂z² + jk₀(n²-1)/2 · u
```

- **高度方向**：FFT / DMFT（离散混合傅里叶变换）谱方法
- **方位方向**：Hankel 函数展开（φ 模分解）
- **边界条件**：DMFT 处理有损地面阻抗边界，PEC 处理导体

### 传播效应

| 效应 | 模型 |
|---|---|
| 自由空间扩散 | 柱面波 1/√r 衰减 |
| 大气折射 | n_atm = 1.0003 |
| 地面反射/吸收 | DMFT（ε_r、σ → 复折射率） |
| 障碍物绕射 | Knife-edge + 全 φ 屏蔽 |
| 多层介质 | 裁剪图层 + 厚度叠加 |

### 频率依赖性

以下参数随频率变化，扫频时自动反映：

- **复折射率** `n(f)` = √(ε_r − jσ/(2πf·ε₀))
- **自由空间波数** `k₀` = 2πf/c
- **DMFT 阻抗参数** `α(f)`
- **自由空间损耗** `L_fs(f)`
- **网格步长** `Δr` ∝ λ = c/f

## 项目架构

```
Config ──→ Problem ──→ Prep ──→ Solver ──→ Post ──→ Results
  │           │          │         │          │
  └────── Context（管道对象）──────────────────────┘
```

### 模块职责

| 层 | 模块 | 职责 |
|---|---|---|
| **core/** | `config.py` | 所有参数集中定义（`EMConfig`） |
| | `context.py` | 管道对象，模块间传递状态 |
| | `problem.py` | 场景加载，配置注入 |
| | `scheduler.py` | 管道调度：problem → prep → solver → post |
| **engine/** | `prep.py` | 网格建立、材料映射、DMFT 算子、初始场 |
| | `solver.py` | PE 步进（分步 FFT + Hankel 传播） |
| | `post.py` | 场提取、路径损耗、结果打包 |
| **physics/** | `materials.py` | `material_n(f, ε_r, σ)` 复折射率 |
| | `terrain.py` | 双线性插值 + 最近邻地面采样 |
| | `dmft.py` | DMFT 基函数与变换矩阵 |
| **scene/** | `builder.py` | 场景工厂（地面、墙、天线） |
| **ui/** | GUI 对话框 | 测量点、天线、材料、扫频 |

### 统一接口

所有 engine/core 模块暴露同一签名：

```python
def run(ctx: Context) -> Context:
    """处理 Context，返回更新后的 Context。"""
```

### 调度器使用

```python
from src.core.config import EMConfig
from src.core.scheduler import Scheduler

config = EMConfig(frequency=2.8e9, antenna_pos=(-5, 0, 6))
scheduler = Scheduler()
results = scheduler.run(config=config, rx_points=[(3, 0, 2)])
# results[0] = {"rx", "L_fs_dB", "path_loss_dB", "E_rx", ...}
```

## 天线类型

| 类型 | 说明 | 可调参数 |
|---|---|---|
| `gaussian`（默认） | 高斯波束 | σ_z（波束宽度）、俯角 |
| `half_wave_dipole` | 半波偶极子（垂直） | 俯角 |
| `microstrip_patch` | 微带贴片 | E 面 HPBW、俯角 |
| `horn` | 喇叭天线 | HPBW、俯角 |
| `isotropic` | 全向点源 | 无 |

默认：`gaussian`，σ_z = 4.0m（垂直 HPBW ≈ 110°）。

## 地面材料

| 材料 | ε_r | σ (S/m) |
|---|---|---|
| 中等干燥地面 | 15.0 | 0.01 |
| 湿润地面 | 25.0 | 0.1 |
| 沙地 | 3.0 | 0.001 |
| 草地 | 12.0 | 0.005 |
| 水面（淡水） | 80.0 | 0.01 |
| 水面（海水） | 80.0 | 4.0 |
| 混凝土 | 6.0 | 0.02 |
| 金属铝 | 1.0 | 3.8×10⁷ |

σ > 10⁵ 视为理想导体（PEC 边界）。

## 默认场景

```
        天线 (-5, 0, 6)   墙 (x=0, z=0-5.5m)    Rx2 (0.78, 0, 3)
             │                  ║
             │    ──────────────╫──────
             │   /              ║
      ───────┼──/────── 地面 ───╫──────────────────
             │ /                ║
           Rx1 (-0.9, 0, 3)    ║
```

- 地面：10m × 10m 平面，含沙/草/土分层 + 水面
- 墙：x=0，y=±15m，z=0-5.5m，铝导体
- 天线：(-5, 0, 6)，默认高斯波束 σ=4m

## GUI 功能

| 菜单 | 功能 |
|---|---|
| 视图 | 俯视/正视/侧视/复位、截图 |
| 图层 | 增加图层、管理裁剪图层 |
| 参数 | 材料参数、天线设置（5 种类型 + 方向图预览） |
| 工具 | 添加/精准添加/管理测量点、Ctrl+S 扫频模式、⚡求解 |

扫频模式：多频率批量计算 → 结果表格 → 曲线图 → CSV 导出。

## 输出字段

| 字段 | 含义 | 单位 |
|---|---|---|
| `dist` | 水平距离 | m |
| `L_fs_dB` | 自由空间损耗 | dB |
| `path_loss_dB` | PE 路径损耗 | dB |
| `E_rx` | 接收电场幅度 | V/m |
| `E_rx_dbuv` | 接收电场强度 | dBμV/m |

Δ = L_pe − L_fs：正值为额外损耗（绕射/吸收），近 0 为自由空间。

## 依赖

```
PyQt5, pyvista, pyvistaqt, vtk
numpy, scipy, numba, matplotlib
rasterio  (可选，DEM 导入)
```

## 参考文献

- 汪路遥. 电波传播的柱坐标系抛物方程模型及其应用研究.
- ITU-R P.526 — Propagation by diffraction.
- Dockery, G. D., & Kuttler, J. R. (1996). An improved impedance-boundary algorithm for Fourier split-step solutions of the parabolic wave equation.
