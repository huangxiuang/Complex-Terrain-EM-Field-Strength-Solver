# EM Solver — 柱坐标系抛物方程（CPE）电磁场求解器

2.8 GHz S 波段地波传播与障碍物绕射分析工具。基于 Feit-Fleck 宽角抛物方程 + SSFT 分步傅里叶法 + DMFT 离散混合傅里叶变换，支持复杂地形、多层介质、障碍物绕射、扫频分析、DEM/ASC 导入导出，以及完整的传输损耗可视化。

## 快速开始

```bash
pip install -r requirements.txt
python run.py                    # 默认金属挡板测试
python run.py --rx 5,0,3        # 自定义接收点
python run.py --freq 3e9 --rx 0,0,6  # 自定义频率
python run.py --plot            # 生成可视化图到 data/
python run.py --help            # 全部选项
```

GUI 模式（3D 场景 + 树形导航 + 绘图）：
```bash
python src/main_window.py
```

## 核心物理模型

### 抛物方程

柱坐标系下的前向抛物方程，以天线为原点沿径向步进：

$$\frac{\partial u}{\partial r} = \frac{j}{2k_0}\frac{\partial^2 u}{\partial z^2} + jk_0\frac{n^2-1}{2}u$$

- **高度方向**：DMFT（有损地面 Leontovich 阻抗边界）或 FFT+PEC（导体地面）
- **方位方向**：FFT → m 模式 Hankel 函数精确值展开（kr < 1e8）
- **角向耦合**：完整 2D FFT（φ×z）实现横向绕射——有限宽墙中心场强高于边缘

### 传播效应

| 效应 | 方法 | 关键机制 |
|------|------|---------|
| 自由空间扩散 | Hankel 比值 + √(r_p/r_c) | 柱面波→球面波，kr<1e8 精确汉克尔 |
| 大气折射 | exp(jk₀(n_atm−2)Δr) | n_atm = 1.0003 |
| 地面反射/吸收 | DMFT + Leontovich BC | α = jk₀√(ε_c−1)/ε_c, σ≤10⁵ 自动处理 |
| 障碍物绕射 | φ 向遮挡 + min 3 步径向段 | z_terrain[φ,i] = max(ground, wall_top) |
| 多层介质 | 裁剪图层 + 厚度叠加 | 三角单元精确 XY 覆盖替代矩形 bounds |
| 有损介质穿透 | exp(jk₀(conj(n_mat)−n_atm)Δr) | conj(n) 修正虚部确保真实衰减 |
| 横向绕射 | φ-FFT m 模式耦合 | 完整 2D CPE 独有 |

### 频率依赖性

- 复折射率 $n(f) = \sqrt{\varepsilon_r - j\sigma/(2\pi f\varepsilon_0)}$
- 自由空间波数 $k_0 = 2\pi f/c$
- DMFT 阻抗参数 $\alpha(f)$ 随频率变化
- 自由空间损耗 $L_{fs}(f,r) = 20\log_{10}(f_{MHz}) + 20\log_{10}(r_{km}) + 32.45$

## 项目架构

```
EMConfig ──→ Problem ──→ Prep ──→ Solver ──→ Post ──→ Results
  │            │          │         │          │
  └────────────── Context（管道对象）──────────────┘
```

### 模块分层

| 层 | 模块 | 职责 |
|---|---|---|
| **core/** | config | `EMConfig` dataclass——频率/天线/网格/TWPE 全部参数集中 |
| | context | 管道载具——13 字段按 Problem→Prep→Solver→Post 阶段填充 |
| | problem | 场景构建 + 配置注入验证 |
| | scheduler | 管道调度器，支持外部注入场景对象 |
| **engine/** | prep | 网格建立、2D 材料映射 z_terrain[φ,r]、DMFT 算子、初始场 |
| | solver | PE 步进：DMFT→φFFT→Hankel→iFFT+折射+地形遮蔽+上边界吸收 |
| | post | 场提取、自由空间对比、L_pe 计算、结果打包 |
| **physics/** | materials | 复折射率 n(f,ε_r,σ) + 导体判定（σ>10⁵→PEC） |
| | dmft | DMFT 正/逆变换矩阵，Leontovich 边界 φ_n(z)=cos(k_n z)+(α/k_n)sin(k_n z) |
| | terrain | 双线性插值 + nearest 回退 + 越界钳位 + numba @njit |
| **scene/** | scenes | 4 预置场景 + @register 注册表 |
| | scene_dialog | 场景选择 + 属性详情对话框 |
| **ui/** | main_window | 主窗口：3D+树形导航+工具栏+后台求解线程 |
| | field_dialog | 测量点 XY 平面选择（等高线叠加）+ 结果表格 |
| | plot_dialog | 6 种可视化（r-z TL, r-z E, φ-z TL, TL vs r, E vs z, 3D表面）+ PNG/PDF/SVG |
| | sweep_dialog | 扫频模式：批量多频率 + 结果表格 + 曲线 + CSV |
| | antenna_dialog | 5 天线类型 + 极坐标方向图预览 + 位置编辑 |
| | material_params_dialog | 12 种预置材料（可编辑，JSON 持久化） |
| | layer_dialog | 图层管理 + 多边形裁剪 + 可见性/透明度/删除 |
| | template_dialog | 预设测量点模版（每场景 3 套，一键加载） |
| **—** | io_utils | DEM(GeoTIFF/IMG) + ASC 导入导出 |
| — | antenna_types | 5 种天线方向图幅度函数 + 初始场生成 |
| — | template_points | 4 场景 × 3 模版 测量点数据 |
| — | visualizer | run.py --plot 离线可视化（6 种 publication-quality 图） |

### 统一接口

所有 engine/core 模块暴露 `def run(ctx: Context) → Context`。

## 场景

| # | key | 名称 | 规模 | 说明 |
|---|---|---|---|---|
| 1 | `metal_barrier` | 金属挡板 | 20×20 m | 平坦地面+铝墙(5.5m)+天线；knife-edge 绕射基准 |
| 2 | `classic` | 自然景观 | 20×20 m | 高斯山丘+正弦河道+沙/草/土分层+鸟+树 |
| 3 | `city_block` | 城市街区 | 20×20 m | 4 栋混凝土建筑(6–15m)+城区多径 |
| 4 | `wilderness` | 荒原 | 1000×1000 m | 9 山(最高188m)+河+湖沼+5种植被+道路桥梁+雪顶岩石沼泽 |

荒原需特殊配置：天线 200m+σ_z=80m+dr_factor=8+N_Z=1024+N_φ=64。

### 预设测量点模版（预设 → 模版测量点…）

| 场景 | 模版 1 | 模版 2 | 模版 3 |
|------|--------|--------|--------|
| 金属挡板 | 墙前墙后 (5) | 高度剖面 (6) | 远场绕射 (5) |
| 自然景观 | 山丘遮挡 (4) | 河道沿线 (4) | 全向扫描 (5) |
| 城市街区 | 建筑穿透 (5) | 街道峡谷 (5) | 多径混合 (5) |
| 荒原 | 跨地形剖面 (6) | 森林穿透 (4) | 全地形覆盖 (5) |

## 天线类型

| 类型 | 说明 | 可调参数 |
|------|------|---------|
| `gaussian`（默认） | 高斯波束 | σ_z（小场景 4m，荒原 80m）、俯角 ±30° |
| `half_wave_dipole` | 半波偶极子（垂直） | 俯角 ±30° |
| `microstrip_patch` | 微带贴片 | E 面 HPBW 30–150°、俯角 |
| `horn` | 喇叭天线 | HPBW 5–90°、俯角 |
| `isotropic` | 全向点源 | 无 |

## 材料

### 基础预置（12 种，可编辑）

| 材料 | ε_r | σ (S/m) | 厚度 |
|------|-----|---------|------|
| 空气 | 1.0006 | 0 | — |
| 干燥土壤 | 15.0 | 0.01 | — |
| 湿润土壤 | 25.0 | 0.1 | — |
| 沙地 | 3.0 | 0.001 | 20 cm |
| 草地 | 12.0 | 0.005 | 30 cm |
| 土地 | 5.0 | 0.005 | 60 cm |
| 水面（淡水） | 80.0 | 0.01 | 100 cm |
| 水面（海水） | 80.0 | 4.0 | 100 cm |
| 混凝土 | 6.0 | 0.02 | 20 cm |
| 玻璃 | 7.0 | 1×10⁻⁶ | 1.5 cm |
| 木材 | 2.0 | 1×10⁻⁴ | 8 cm |
| 金属铝 | 1.0 | 3.8×10⁷ | — |

### 荒原扩展材料（ITU P.527/P.833 校准）

| 材料 | ε_r | σ | 2.8 GHz 衰减率 | ITU |
|------|-----|-----|---------------|-----|
| 低矮植被 | 1.2 | 6×10⁻⁵ | 0.09 dB/m | P.833 |
| 森林冠层 | 1.5 | 2×10⁻⁴ | 0.27 dB/m | P.833 |
| 沥青路面 | 4.0 | 0.001 | 0.84 dB/m | P.527 |
| 沼泽湿地 | 30.0 | 0.05 | 5.95 dB/m | P.527 |
| 雪/冰层 | 2.0 | 1×10⁻⁴ | 0.10 dB/m | P.527 |
| 岩石崖面 | 7.0 | 1×10⁶ | PEC (∞) | — |

σ > 10⁵ → 理想导体（PEC 场清零）。非导体使用复折射穿透 `conj(n)` 修正虚部。

## GUI 功能

### 菜单

| 菜单 | 功能项 |
|------|--------|
| **场景** | 场景选择（4 预置）、场景属性详情 |
| **预设** | 模版测量点（12 套一键加载） |
| **视图** | 俯视/正视/侧视/复位、截图 |
| **图层** | 增加图层（多边形裁剪）、管理裁剪图层 |
| **导入/导出** | 导入 DEM/GeoTIFF、导入 ASC 高程栅格、导出地形为 ASC |
| **参数** | 材料参数（12 种可编辑+JSON）、天线设置（5 类型+方向图预览） |
| **工具** | 扫频模式 (Ctrl+S)、快速添加测量点、管理测量点 |

### 画图（6 种 + PNG/PDF/SVG 导出）

1. r-z 传输损耗分布（jet 蓝→红 + 地形叠加）
2. r-z 电场分布（inferno）
3. φ-z 传输损耗分布（−90°~90°）
4. 路径损耗 vs 距离
5. 场强 vs 高度
6. TL 3D 表面

### 扫频

批量多频率 → 表格(L_pe+E_dBμV/m+E_rx) → 曲线 → CSV。快速模式约 4× 加速。

### 导入/导出

- **DEM (GeoTIFF/IMG)**：rasterio 读取 → 自动降采样 ≤150×150 → 替换场景地形
- **ASC 栅格**：ESRI ASCII Raster 导入/导出（ncols/nrows/xllcorner/yllcorner/cellsize/NODATA_value）

## 输出字段

| 字段 | 含义 | 单位 |
|------|------|------|
| `dist` | 水平距离 | m |
| `L_fs_dB` | 自由空间损耗 | dB |
| `path_loss_dB` | PE 路径损耗 | dB |
| `E_rx` | 接收电场幅度 | V/m |
| `E_rx_dbuv` | 接收电场强度 | dBμV/m |

Δ = L_pe − L_fs：正值=额外损耗，负值=场增强，≈0=自由空间。

## CLI

```bash
python run.py --freq 2.8e9 --tx="-400,0,200" --nz=1024 --dr=8.0 --plot
```

## 依赖

PyQt5 · pyvista · pyvistaqt · vtk · numpy · scipy · numba · matplotlib · rasterio(可选)

## 文档

| 文档 | 路径 |
|------|------|
| 用户使用指南 | `docs/用户使用指南.html` |
| 工作与技术报告（企业级） | `docs/工作和技术报告.html` |
| CPE 模型详细介绍 | `docs/CPE求解器模型详细介绍.md` |
| 抛物方程完整推导 | `docs/抛物方程推导.md` |
| 标量近似详解 | `docs/标量近似详解.md` |
| 全部参数表 | `docs/全部参数表.md` |
| 荒原场景修复报告 | `docs/荒原场景修复报告.md` |

## 参考文献

- 汪路遥. 电波传播的柱坐标系抛物方程模型及其应用研究[D]. 西南交通大学, 2021.
- ITU-R P.526 — Propagation by diffraction.
- ITU-R P.527 — Electrical characteristics of the surface of the Earth.
- ITU-R P.833 — Attenuation in vegetation.
- Dockery & Kuttler (1996). An improved impedance-boundary algorithm for Fourier split-step solutions of the parabolic wave equation. IEEE TAP.
- Levy (2000). Parabolic Equation Methods for Electromagnetic Wave Propagation. IEE.
