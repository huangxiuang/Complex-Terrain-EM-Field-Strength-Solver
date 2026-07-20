# EM Solver — 柱坐标系抛物方程（CPE）电磁场求解器

2.8 GHz 频段地波传播与障碍物绕射分析工具。基于 SSFT-PE（分步傅里叶变换抛物方程）方法，支持复杂地形、多层介质、人工障碍物、扫频分析、传输损耗可视化。

## 快速开始

```bash
pip install -r requirements.txt
python run.py                    # 默认测试点
python run.py --rx 5,0,3        # 自定义接收点
python run.py --freq 3e9 --rx 0,0,6  # 自定义频率
python run.py --plot            # 生成可视化图
python run.py --help            # 全部选项
```

GUI 模式（3D 场景 + 树形导航 + 绘图）：
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

| 效应 | 模型 | 说明 |
|---|---|---|
| 自由空间扩散 | 柱面波 1/√r 衰减 + 汉克尔函数精确值（kr < 1e8） | 非远场物理细节完整保留 |
| 大气折射 | n_atm = 1.0003 | 标准大气 |
| 地面反射/吸收 | DMFT（ε_r、σ → 复折射率 → Leontovich 阻抗边界） | 有损地面自动处理 |
| 障碍物绕射 | obstacle_type=wall，按 φ 屏蔽 + 最小 3 步径向段 | Knife-edge 绕射 |
| 多层介质 | 裁剪图层 + 厚度叠加 + 精确三角单元 XY 覆盖（KDTree + ray-casting） | 替代矩形 bounds 粗判 |
| 有损介质穿透 | conj(n) 修正虚部符号，复折射衰减 | σ≤10⁵ S/m 介质穿透 |
| 横向绕射 | φ-FFT 耦合（m 模式） → 有限宽障碍物两端能量汇聚 | 完整 2D CPE 独有 |
| 地形遮挡 | 双线性插值 + 边界钳位 → z_terrain[φ,r] 二维地形图 | 地形越界安全回退 |

### 频率依赖性

- **复折射率** `n(f)` = √(ε_r − jσ/(2πf·ε₀))
- **自由空间波数** `k₀` = 2πf/c
- **DMFT 阻抗参数** `α(f)`
- **自由空间损耗** `L_fs(f)`
- 衰减常数高频趋近 `α = σ/(2c·ε₀·√ε_r)`，与 f 无关

## 项目架构

```
Config ──→ Problem ──→ Prep ──→ Solver ──→ Post ──→ Results
  │           │          │         │          │
  └────── Context（管道对象）──────────────────────┘
```

### 模块职责

| 层 | 模块 | 职责 |
|---|---|---|
| **core/** | `config.py` | `EMConfig` dataclass，所有参数集中定义 |
| | `context.py` | `Context` 管道对象，模块间传递状态 |
| | `problem.py` | 场景加载，配置注入 |
| | `scheduler.py` | 管道调度：problem → prep → solver → post |
| **engine/** | `prep.py` | 网格建立、材料映射、DMFT 算子、初始场 |
| | `solver.py` | PE 步进（分步 FFT + Hankel 传播） |
| | `post.py` | 场提取、路径损耗、结果打包 |
| **physics/** | `materials.py` | `material_n(f, ε_r, σ)` 复折射率 |
| | `terrain.py` | 双线性插值 + 最近邻地面采样 |
| | `dmft.py` | DMFT 基函数与变换矩阵 |
| **scene/** | `scenes.py` | 3 个预置场景 + 注册表 |
| | `scene_dialog.py` | 场景选择 + 属性详情对话框 |
| **ui/** | `main_window.py` | 主窗口（3D + 树形导航 + 工具栏） |
| | `field_dialog.py` | 测量点对话框（XY 等高线图） |
| | `plot_dialog.py` | 画图功能（6 种图，独立窗口） |
| | `sweep_dialog.py` | 扫频模式（批量 + 曲线 + CSV） |
| | `antenna_dialog.py` | 天线设置（5 类型 + 方向图预览） |
| | `material_params_dialog.py` | 材料参数（空气/土壤/水体…可编辑） |
| | `layer_dialog.py` | 图层管理 |

### 统一接口

所有 engine/core 模块暴露同一签名：

```python
def run(ctx: Context) -> Context:
```

### 调度器使用

```python
from src.core.config import EMConfig
from src.core.scheduler import Scheduler

config = EMConfig(frequency=2.8e9, antenna_pos=(-5, 0, 6))
scheduler = Scheduler()
results = scheduler.run(config=config, rx_points=[(3, 0, 2)])
```

## 场景

| # | 场景 | 对象 | 说明 |
|---|---|---|---|
| 1 | 金属挡板 | 平坦地面 + 铝墙 + 天线 | knife-edge 绕射 |
| 2 | 自然景观 | 高斯山丘 + 正弦河道 + 植被 + 鸟 + 树 + 天线 | 复杂地形 |
| 3 | 城市街区 | 平坦地面 + 4 栋混凝土建筑 + 天线 | 城区多径 |
| 4 | 荒原（大尺度） | 1000m×1000m 山脉 + 河道 + 湖沼 + 草地/疏林/密林 + 道路桥梁电线杆 + 雪顶/岩石/沼泽 | 大区域复杂环境传播 |

荒原场景需特殊配置：天线 200m 高 + 宽波束 σ=80m + dr_factor=8 + N_Z=1024。

## 天线类型

| 类型 | 说明 | 可调参数 |
|---|---|---|
| `gaussian`（默认） | 高斯波束，σ_z=4m（小场景）/ σ_z=80m（荒原） | σ_z、俯角 |
| `half_wave_dipole` | 半波偶极子（垂直） | 俯角 |
| `microstrip_patch` | 微带贴片 | E 面 HPBW、俯角 |
| `horn` | 喇叭天线 | HPBW、俯角 |
| `isotropic` | 全向点源 | 无 |

## 材料

| 材料 | ε_r | σ (S/m) | 厚度 |
|---|---|---|---|
| 空气 | 1.0006 | 0 | — |
| 干燥土壤 | 15.0 | 0.01 | — |
| 湿润土壤 | 25.0 | 0.1 | — |
| 沙地 | 3.0 | 0.001 | 20 cm |
| 草地 | 12.0 | 0.005 | 30 cm |
| 土地 | 5.0 | 0.005 | 60 cm |
| 水面（淡水） | 80.0 | 0.01 | 100 cm |
| 水面（海水） | 80.0 | 4.0 | 100 cm |
| 混凝土 | 6.0 | 0.02 | 20 cm |
| 金属铝 | 1.0 | 3.8×10⁷ | — |

σ > 10⁵ 视为理想导体（PEC 边界，场清零）。

## GUI 功能

### 左侧树形导航
- **📁 测量点**：右键删除，双击编辑坐标
- **⚙ 功能**：➕添加 / ⚡求解 / 📊画图（双击执行）
- **📊 Results**：求解后自动出现，展开查看每点详情，双击弹窗

### 菜单
| 菜单 | 功能 |
|---|---|
| 场景 | 场景选择（3 个预置）/ 场景属性详情 |
| 视图 | 俯视/正视/侧视/复位、截图 |
| 图层 | 增加图层、管理裁剪图层 |
| 参数 | 材料参数（含空气，可编辑）、天线设置（5 类型 + 方向图预览） |
| 工具 | 扫频模式（Ctrl+S）、管理测量点 |

### 画图（📊）
勾选图类型 → 设参数（φ/z/r）→ 生成 → 弹出独立大窗口：
1. r-z 传输损耗分布（jet 蓝→红）
2. r-z 电场分布（inferno）
3. φ-z 传输损耗分布（−90°~90°）
4. 路径损耗 vs 距离
5. 场强 vs 高度
6. TL 3D 表面

导出：PNG / PDF / SVG

### 扫频模式
多频率批量计算 → 结果表格 → 曲线图 → CSV 导出。快速模式（N_Z=1024, N_PHI=64）约 4× 加速。

## 输出字段

| 字段 | 含义 | 单位 |
|---|---|---|
| `dist` | 水平距离 | m |
| `L_fs_dB` | 自由空间损耗 | dB |
| `path_loss_dB` | PE 路径损耗 | dB |
| `E_rx` | 接收电场幅度 | V/m |
| `E_rx_dbuv` | 接收电场强度 | dBμV/m |

Δ = L_pe − L_fs：正值为额外损耗（绕射/吸收/天线方向图衰减），近 0 为自由空间。

## 依赖

```
PyQt5, pyvista, pyvistaqt, vtk
numpy, scipy, numba, matplotlib
```

## 参考文献

- 汪路遥. 电波传播的柱坐标系抛物方程模型及其应用研究.
- ITU-R P.526 — Propagation by diffraction.
- Dockery, G. D., & Kuttler, J. R. (1996). An improved impedance-boundary algorithm for Fourier split-step solutions of the parabolic wave equation.
