# CPE（柱坐标抛物方程）求解器模型详细介绍

> 参考论文：[汪路遥] 电波传播的柱坐标系抛物方程模型及其应用研究，西南交通大学，2021
> 编写日期：2026-07-13

## 目录

1. 物理背景：从麦克斯韦到抛物方程
2. Feit-Fleck 宽角抛物方程
3. SSFT 分步傅里叶解法
4. 柱坐标系 PE 理论
5. 完整 2D CPE
6. 地形数据如何带入模型
7. 完整求解步骤
8. 数值验证方法
9. 横向绕射验证
10. 预期物理现象

---

## 1. 物理背景：从麦克斯韦到抛物方程

### 1.1 出发点——亥姆霍兹方程

在均匀无源区域，时谐场 $e^{-j\omega t}$ 满足：

$$
\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} + k_0^2 n^2\right) \psi = 0
$$

其中 $\psi$ 为任意电磁场分量（标量近似），$k_0 = 2\pi f/c$ 为自由空间波数，$n$ 为传播媒质折射指数，$x$ 为传播主方向。

### 1.2 抛物近似——为什么能降阶

设衰减函数 $u(x,y,z) = e^{-j k_0 x} \psi(x,y,z)$，代入原方程，因式分解得到前向传播方程（论文式 2-4）：

$$
\frac{\partial u}{\partial x} = -j k_0 (1 - Q) u
$$

其中伪微分算子 $Q = \sqrt{\frac{1}{k_0^2}\left(\frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right) + n^2}$

**关键不等式**：方程从二阶椭圆型 PDE 降为一阶抛物型 PDE。付出的代价是：
- ❌ ~~忽略了后向散射波（电波遇到障碍物反射回来的部分）~~ $\rightarrow$ ⚠️ **已通过 TWPE 实现**（见 §5 和 §10.6）
- ✅ 极大提高计算效率（步进迭代替代全局求解）
- ✅ 大区域环境（数百公里）仍可计算，而全波方法（MoM/FEM）因网格数爆炸无法处理

### 1.3 几种常见的 Q 近似方法

| 近似方法 | 公式 | 适用仰角 | 求解方式 | 论文出处 |
|---------|------|---------|---------|---------|
| Taylor | $Q \approx 1 + \frac{1}{2}(\varepsilon+\mu)$ | $\leq 15^\circ$（窄角） | SSFT/FD | 式 2-7 |
| Tappert | $Q \approx \sqrt{1+\mu} + \frac{1}{2}\varepsilon$ | $\leq 15^\circ$（窄角） | SSFT | 式 2-9 |
| Padé (Claerbout) | $Q \approx \dfrac{1+\frac{3}{4}(\varepsilon+\mu)}{1+\frac{1}{4}(\varepsilon+\mu)}$ | $\leq 70^\circ$（宽角） | FD only | 式 2-11 |
| **Feit-Fleck** | **$Q \approx \sqrt{1+\mu} + \sqrt{1+\varepsilon} - 1$** | **$\leq 30^\circ$（宽角）** | **SSFT** | **式 2-13** |

本方案选用 **Feit-Fleck 近似**：SSFT 求解速度快，$30^\circ$ 仰角足够覆盖绝大多数对流层传播场景。

其中 $\varepsilon = n^2 - 1$（折射项），$\mu = \dfrac{1}{k_0^2}\left(\dfrac{\partial^2}{\partial y^2} + \dfrac{\partial^2}{\partial z^2}\right)$（绕射项）。

---

## 2. Feit-Fleck 宽角抛物方程

### 2.1 最终形式

将 Feit-Fleck 近似代入，得到（论文式 2-14）：

$$
\frac{\partial u}{\partial x} = j k_0 \left[ \sqrt{1 + \frac{1}{k_0^2}\left(\frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right)} + n - 2 \right] u
$$

对沿 $x$ 方向的传播距离 $\Delta x$，积分得（论文式 2-15）：

$$
u(x+\Delta x, y, z) = \exp\left[ j k_0 \Delta x \left( \sqrt{1 + \frac{1}{k_0^2}\left(\frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right)} + n - 2 \right) \right] \cdot u(x, y, z)
$$

### 2.2 折射-绕射分离（SSFT 的关键）

当折射指数 $n$ 在 $(y,z)$ 平面内变化缓慢时，可将折射和绕射分离（论文式 2-16）：

$$
u(x+\Delta x, y, z) = \exp\!\big[j k_0 \Delta x (n-2)\big] \cdot \exp\!\bigg[j k_0 \Delta x \sqrt{1 + \frac{1}{k_0^2}\left(\frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right)}\bigg] \cdot u(x, y, z)
$$

- **绕射项** = 自由空间中波的扩散（在频域处理：每个平面波分量独立传播）
- **折射项** = 大气折射率修正（在空域处理：逐点相乘）

---

## 3. SSFT 分步傅里叶解法

### 3.1 为什么在频域处理绕射

微分算子 $\partial^2/\partial z^2$ 在空域是二阶导数，但通过傅里叶变换变成简单的代数运算：

$$
\frac{\partial^2}{\partial z^2} \xrightarrow{\text{FFT}} -k_z^2
$$

于是绕射项中的平方根算子变为：$\sqrt{1 - k_z^2/k_0^2}$

### 3.2 SSFT 一步的完整公式（论文式 2-20）

$$
u(x+\Delta x, z) = \exp\!\big[j k_0 \Delta x (n-2)\big] \cdot \text{FFT}^{-1}\Big[ \exp\!\big(j \Delta x \sqrt{k_0^2 - k_z^2}\big) \cdot \text{FFT}[u(x, z)] \Big]
$$

**拆解每一步**：

```
输入: u(x, z)              —— 当前位置 x 处、所有高度 z 上的场分布

步骤1:  FFT in z
        U(kz) = FFT[u(z)]  —— 将场分解为平面波分量

步骤2:  频域传播
        U'(kz) = U(kz) × exp(j Δx √(k₀² - kz²))
        —— 每个平面波分量按各自的波数独立传播 Δx

步骤3:  IFFT
        u'(z) = IFFT[U'(kz)]
        —— 回到空域，此时场已经历了绕射

步骤4:  折射修正
        u(x+Δx, z) = u'(z) × exp(j k₀ (n-2) Δx)
        —— 大气折射使电波弯曲
```

### 3.3 倏逝波处理

当 $|k_z| > k_0$ 时（对应超地平角传播），$\sqrt{k_0^2 - k_z^2}$ 为虚数：

$$
\exp\!\big(j \Delta x \sqrt{k_0^2 - k_z^2}\big) = \exp(-\Delta x \cdot \text{正实数}) < 1
$$

倏逝波自动指数衰减——无需额外判断，物理上正确。

### 3.4 Nyquist 条件与步长选择

- **高度向**：$\Delta z \leq \lambda/2$ 以满足 Nyquist 采样定理。例：$f=2.8$ GHz $\rightarrow$ $\lambda=0.107$ m $\rightarrow$ $\Delta z \leq 0.05$ m
- **传播方向**：$\Delta x \leq 2\lambda \sim 10\lambda$（PE 方法对此约束较宽松）。例：$\Delta x = 0.5\lambda \sim 1\lambda$ 是安全选择
- **网格数 $N_z$**：必须是 2 的幂（FFT 效率最优：256, 512, 1024, $\ldots$）

---

## 4. 柱坐标系 PE 理论

### 4.1 为什么需要柱坐标

传统直角坐标 PE 在水平面仅小角度（$\pm 30^\circ$）内有效，因为水平方向的吸收边界会衰减偏离主轴的能量，且网格尺寸固定不能适应径向传播的自然扩散。

柱坐标 PE（CPE）以天线为原点（$r=0$），沿径向 $r$ 传播，覆盖 **$360^\circ$ 全向**。

### 4.2 CPE 完整公式（论文式 2-36）

$$
u(r+\Delta r, \varphi, z) = \text{FFT}^{-1}_{m,k_z}\!\left[ \frac{H_m^{(1)}(k_r (r+\Delta r))}{H_m^{(1)}(k_r \cdot r)} \cdot \text{FFT}_{\varphi,z}[u(r, \varphi, z)] \right] \cdot e^{j k_0 (n-1) \Delta r}
$$

其中：
- $H_m^{(1)}$ = 第一类汉克尔函数（柱面波的径向传播）
- $\text{FFT}_{\varphi,z}$ = 在方位角 $\varphi$ 和高度 $z$ 方向上的二维傅里叶变换
- $m$ = 角向模式序号（对应 $e^{j m \varphi}$）
- $k_r = \sqrt{k_0^2 n^2 - k_z^2}$ = 径向波数

完整 CPE 需要 2D FFT（$\varphi \times z$），计算量 $O(N_r \times N_\varphi N_z \log(N_\varphi N_z))$。

### 4.3 全向特性验证

CPE 的关键优点（论文 2.2.2 节验证）：
- CPE 与 MoM（矩量法）在任意方位角都吻合良好
- 传统 3DPE 仅在 $\pm 30^\circ$ 内准确，超过则急剧衰减
- 体现了水平方向的绕射效应（Quasi-3DPE 忽略此效应）

---

## 5. 完整 2D CPE

### 5.1 实现方案

本求解器实现了论文式 (2-36) 的**完整 2D CPE**，不做任何角度简化。使用 2D FFT（$\varphi \times z$）在全体方位角上同时推进传播，自动包含横向绕射。

### 5.2 使用的公式

$$
u(r+\Delta r, \varphi, z) = \text{FFT}^{-1}_{m,k_z}\!\left[ \frac{H_m^{(1)}(k_r (r+\Delta r))}{H_m^{(1)}(k_r r)} \cdot \text{FFT}_{\varphi,z}[u(r, \varphi, z)] \right] \cdot \exp\!\big(j k_0 (n-1) \Delta r\big)
$$

其中：
- **$\text{FFT}_{\varphi,z}$** = 沿方位角 $\varphi$ 和高度 $z$ 的**二维傅里叶变换**
- **$m$** = 角向模式序号（$\varphi$ 方向 FFT 产生），取值 $m = -N_\varphi/2, \ldots, N_\varphi/2 - 1$
- **$k_z$** = 垂直波数（$z$ 方向 FFT 产生）
- **$k_r$** = $\sqrt{k_0^2 n^2 - k_z^2}$ = 径向波数
- **$H_m^{(1)}(\cdot)$** = 第一类汉克尔函数，阶数 $m$（柱面波的径向传播函数）
- **$\exp(j k_0 (n-1) \Delta r)$** = 折射项（大气折射修正）

### 5.3 关键实现细节

| 项目 | 值 | 说明 |
|------|----|------|
| $N_\varphi$（方位角 FFT 点数） | **128** | 2 的幂，角度分辨率 $\sim 2.8^\circ$ |
| $N_z$（高度 FFT 点数） | **256** | 2 的幂 |
| $\varphi$ 网格范围 | $0 \sim 2\pi$ | 128 个点均匀分布，覆盖 $360^\circ$ |
| 汉克尔函数 | `scipy.special.hankel1` | 对所有 $(m, k_z, r)$ 使用精确值 |
| 汉克尔截断 | $k_r \cdot r < \mathbf{1e8}$ | 1e8 以内全程精确汉克尔（无渐近近似） |
| 渐近替代 | $k_r \cdot r \geq 1e8$ 时退化为 $\sqrt{r_p / r_c} \exp(j k_r \Delta r)$ | 远场精度无损 |
| 扩散修正 | $\sqrt{r_p / r_c}$ 乘在 IFFT **之后** | 柱面波 $1/\sqrt{r}$ $\rightarrow$ 球面波 $1/r$ |
| 初始场 | 高度向高斯（$\sigma_z=2$m），方位角均匀 | $u(0, \varphi, z) = \exp\!\big(-(z - z_A)^2 / 2\sigma^2\big)$ |
| 吸收边界 | Cosine-taper 窗 | 顶部 $1/8$ 区域渐变衰减到零 |
| 单点计算时间 | **$\sim 4$–$7$ 秒** | Mac, $N_\varphi=128$, $N_z=256$ |

### 5.4 汉克尔比值的物理意义

$H_m^{(1)}(k_r r_c) / H_m^{(1)}(k_r r_p)$ 是**每个 $(m, k_z)$ 模式**沿径向传播 $\Delta r$ 后的复振幅变化：

- **幅度变化**：柱面波的 $1/\sqrt{k_r r}$ 扩散（由汉克尔函数的大宗量渐近形式给出）
- **相位变化**：$\exp(j k_r \Delta r)$ —— 径向传播的相移

每步对全部的 $(m, k_z)$ 组合计算此比值。由于汉克尔函数在 $k_r \cdot r < 1e8$ 时用精确值，散射阴影区等非远场区域的物理细节被完整保留。

### 5.5 二维地形图 (2D Terrain Map)

地形不再是单一剖面线，而是 **$z_{\text{terrain}}[\varphi, r]$** 的二维数组：

$$
z_{\text{terrain}}[N_\varphi, N_r]: \text{每个 } \varphi \text{ 方向独立提取地形剖面}
$$

- 对每个 $\varphi$ 方向（共 128 个），沿径向采样地面高程
- 双线性插值从 terrain 网格获取高度
- 障碍物（墙体）的包围盒高度叠加到对应 $\varphi$ 剖面
- 步进时：每个 $\varphi$ 方向独立应用地形遮蔽（$u[\varphi, z < z_{\text{terrain}}[\varphi,i]] = 0$）

### 5.6 横向绕射效应

完整 2D CPE 与简化径向 2D-PE 的**关键区别**：

| 特性 | 完整 2D CPE（本方案） | 简化径向 2D-PE |
|------|---------------------|----------------|
| FFT 维度 | 2D（$\varphi \times z$） | 1D（仅 $z$） |
| 覆盖范围 | $360^\circ$ 全向 | 单径向线 |
| 横向绕射 | ✅ **自动包含**（$\varphi$-FFT 完成 $m$ 模式耦合） | ❌ 忽略 |
| 有限宽墙后的场分布 | ✅ 中心比边缘强（能量从墙两端汇聚） | ❌ 无法区分 |
| 计算量 | $O(N_r \cdot N_\varphi N_z \log(N_\varphi N_z))$ | $O(N_r \cdot N_z \log N_z)$ |

---

## 6. 地形数据如何带入模型

> **这是本方案的核心**——将 3D 场景的地形/障碍物信息转化为 CPE 的边界条件。

### 6.1 地形数据的两种来源

本系统支持两种场景来源，地形数据提取流程统一：

```
┌────────────────────────────────────────────────────┐
│  场景数据源                                         │
│                                                    │
│  ┌──────────────────┐   ┌──────────────────────┐   │
│  │ Simple Scene      │   │ DEM Scene            │   │
│  │ (simple_scene_    │   │ (dem_loader.py)      │   │
│  │  builder.py)      │   │                      │   │
│  │                   │   │ rasterio → 降采样    │   │
│  │ 平面地 z=0        │   │ → 中心化 →           │   │
│  │ 墙体 x=0,z=0~5    │   │ StructuredGrid       │   │
│  │ 天线 (-5,0,6)     │   │                      │   │
│  └────────┬─────────┘   └──────────┬───────────┘   │
│           │                        │               │
│           └────────┬───────────────┘               │
│                    ▼                                │
│          地形剖面提取函数                            │
│          extract_terrain_profile(φ, N_r, r_max)     │
└────────────────────────────────────────────────────┘
```

### 6.2 核心转换：场景对象 $\rightarrow$ 二维地形图

本方案使用完整 2D CPE，需要建立 **$z_{\text{terrain}}[\varphi, r]$** 的二维地形图——每个方位角方向独立提取地形剖面。

#### 步骤 A：为每个 $\varphi$ 方向建立径向采样点

对 $N_\varphi$ 个方位角中的每一个（$\varphi \in [0, 2\pi)$），从 $r=0$ 到最大距离 $r_{\max}$，均匀采样 $N_r$ 个点：

$$
\begin{aligned}
x_k(\varphi) &= x_A + r_k \cdot \cos(\varphi) \\
y_k(\varphi) &= y_A + r_k \cdot \sin(\varphi)
\end{aligned}
$$

其中天线 $(x_A, y_A)$ 为坐标原点，$r_k = k \cdot \Delta r$。

#### 步骤 B：从 terrain StructuredGrid 插值地面高程

terrain 网格是 `pyvista.StructuredGrid`，储存了每个 $(x_i, y_j)$ 处的 $z_{ij}$。

对每个 $(\varphi, r_k)$ 对应的 $(x_k, y_k)$：
1. 找到包围它的 4 个网格顶点
2. 双线性插值得到地面高度 $z_{\text{ground}}(\varphi, r_k)$
3. 从 `extra["original_z"]` 获取原始高程（未经垂直夸张）

#### 步骤 C：叠加障碍物（墙体等）

遍历场景中所有障碍物对象（当前仅有 `wall`）：
- 对每个 $(\varphi, r_k)$ 采样点，判断该点是否落在障碍物的包围盒内
- 若是，将 $z_{\text{terrain}}(\varphi, r_k)$ 设为障碍物的顶面高度

**对于当前简单场景的具体计算**（以 $\varphi=0$（朝向 $+X$）为例）：

| 径向距离 $r$ | 对应的世界坐标 | 地面高度 | 障碍物高度 | 最终 $z_{\text{terrain}}(0, r)$ |
|-----------|--------------|---------|-----------|-------------------|
| $0 \sim 5$ m | $x: -5 \rightarrow 0$, $y: 0$ | 0 | $-$ | 0 |
| **5 m（墙位置）** | $x=0$, $y=0$ | 0 | **墙体 $0 \sim 5$m** | **5.0（墙顶）** |
| $5 \sim 10$ m | $x: 0 \rightarrow 5$, $y: 0$ | 0 | $-$ | 0 |

其他 $\varphi$ 方向（如 $\varphi=\pi$，朝向 $-X$）可能不与墙体相交，地形图在该方向的对应位置高度为 0。

#### 步骤 D：输出二维地形数组

$$
z_{\text{terrain}} = \text{array}[N_\varphi, N_r]
$$

每一行是一个 $\varphi$ 方向的地形剖面。这就是 CPE 的 2D 地形边界。

### 6.3 二维地形图在 CPE 步进中的使用

在**每一**个步进面 $r_i$ 处（第七步 7c）：

```
for i in range(1, N_r):
    r_i  = i * dr

    // 2D SSFT 步进计算绕射+折射 → u(r_i, φ, z)
    u = ssft_step_2d(u, dr, k0, n)

    // 二维地形遮蔽：每个 φ 方向独立归零
    for each φ_p in 0..N_φ-1:
        h_i = z_terrain[φ_p, i]
        u[φ_p, z < h_i] = 0

    // 吸收边界：顶部 1/8 区域 cosine-taper
    u[:, :] *= absorber(z)
```

**可视化理解**（单 $\varphi$ 剖面）：

```
z ↑
  │
6m├── ● 天线
  │      ╲
5m├── ─ ─ ┼═══════╗ 墙体 z=0~5m
  │  绕射波 ╲      ║
  │          ╲     ║  阴影区
  │           ╲    ║  (u=0 inside wall)
  │            ╲   ║
0m├─────────────┼──╩──────────────→ 地面
  │   u=0        │   u=0
  └──────────────┼─────────────────→ r
  0m            5m              10m
```

- $r < 5$m：自由空间传播，地面以下 $u=0$
- $r = 5$m：墙体 $z \in [0,5$m$]$ 内部 $u=0$（完全遮蔽）
- $r > 5$m：阴影区，墙顶以上的场通过绕射进入

注意：**完整 2D CPE 在 $\varphi$ 方向也发生能量交换**。有限宽墙体在 $\varphi=0$ 方向产生阴影，但由于 $\varphi$-FFT 的耦合作用，阴影区中心的场强比边缘更强——来自墙两侧的能量同时汇聚到中心。

### 6.4 为什么这是 PE 的核心优势

**射线追踪（RT）** 需要显式找出所有绕射路径：直射、反射、墙顶绕射、墙角绕射……每增加一个障碍物，路径数指数增长。

**PE 不需要**——它把地形当作边界条件，场的自然演化（通过 SSFT 中的绕射算子 + 边界强制为零）自动产生正确的绕射行为。你不需要告诉它"电波会绕墙"，PE 自己就会算出来。

### 6.5 对于更复杂地形的扩展

当场景中有 DEM 高精度地形时，同理：
- 对每个 $\varphi$ 方向从 DEM 网格中采样 $z_{\text{terrain}}(\varphi, r)$
- 如果路径上经过山坡、谷地，该 $\varphi$ 方向的剖面会跟着起伏
- 如果导入了 STL 建筑模型，建筑物的包围盒会叠加到对应 $\varphi$ 的剖面上

### 6.6 数据流总结

```
3D 场景 (scene_objects)
  │
  ├── terrain.extra["X"]  ──→  x 坐标网格 (2D)
  ├── terrain.extra["Y"]  ──→  y 坐标网格 (2D)
  ├── terrain.extra["original_z"] ──→ 高程数据 (2D)
  ├── wall.mesh.bounds    ──→  墙体包围盒
  └── antenna.extra["position"] ──→ 天线坐标

       │
       ▼
  对每个 φ_p ∈ [0, 2π) (N_phi=128 个方向):
     沿该方向径向采样 N_r 个点:
       x_k = xA + r_k cos(φ_p)
       y_k = yA + r_k sin(φ_p)

       ▼
  双线性插值 terrain(Z) → z_ground(φ_p, r_k)
  检查是否在 wall 包围盒内 → z_wall(φ_p, r_k)
  z_terrain(φ_p, r_k) = max(z_ground, z_wall)

       │
       ▼
  输出: 二维数组 z_terrain[N_phi, N_r]

       │
       ▼
  CPE 步进循环 (2D FFT over φ,z):
     每步在每个 φ 方向独立地形遮蔽 u[φ, z<z_terrain[φ,i]] = 0
```

---

## 7. 完整求解步骤

### 输入参数

| 参数 | 符号 | 来源 | 示例值 |
|------|------|------|--------|
| 天线位置 | $(x_A, y_A, z_A)$ | 场景中天线对象 | $(-5, 0, 6)$ |
| 接收点位置 | $(x_B, y_B, z_B)$ | 用户指定 | $(5, 0, 3)$ |
| 频率 | $f$ | 用户设置 | 2.8 GHz |
| 极化 | $-$ | 可选 | 垂直极化 |
| 场景对象 | `scene_objects` | `main_window.scene_objects` | terrain + wall + antenna |

### 算法步骤

```
步骤 1: 坐标变换
        r_B = √((xB-xA)² + (yB-yA)²)
        φ_B = atan2(yB-yA, xB-xA)

步骤 2: 物理常数
        k₀ = 2πf/c
        λ = c/f
        Δr = λ (dr_factor × λ)
        N_r = ceil(r_B / Δr)

步骤 3: 方位角网格
        N_φ = 128 (2的幂，角度分辨率 ~2.8°)
        φ_p = p·2π/N_φ,  p = 0..N_φ-1

步骤 4: 高度网格
         z_min = 0  (DMFT 计算域 z ≥ 0)
         z_max = max(zB + 200, z_A + 200)
         N_z = 256 (DMFT 点数)
         Δz = z_max / (N_z - 1)
         k_n = (n+0.5)π/(z_max+Δz), n = 0..N_z-1 (DMFT 特征值)

步骤 5: 提取二维地形图（见第6节）
         z_terrain[φ_p, i]  for p = 0..N_φ-1, i = 0..N_r

步骤 6: 初始场 (r = r₀)
         for each φ_p, z_k:
             u(0, φ_p, z_k) = exp(-(z_k - zA)² / 2σ²)
         归一化 (uniform in φ, Gaussian in z)

步骤 7: 2D SSFT + DMFT 步进循环
         for i = 1 to N_r:
             r_prev = r_vals[i-1]
             r_curr = r_vals[i]

             7a: DMFT (z 向, 阻抗边界):
                 对每个 φ: U_DMFT(φ, k_n) = Σ_j u(φ, z_j)·φ_n(z_j)·Δz/N_n
                 其中 φ_n(z) = cos(k_n z) + (α/k_n)·sin(k_n z)
                 N_n = (z_max/2)·(1+|α/k_n|²)
                 α = jk₀√(ε_c-1)/ε_c  (TM 极化, Leontovich BC)

             7b: FFT (φ 向):
                 对每个 k_n: U(m, k_n) = FFT_φ[U_DMFT(φ, k_n)]

             7c: 频域传播（汉克尔比值）:
                 for each k_n where k_r = √(k₀²n² - k_n²) > 0:
                     ratio = H_m⁽¹⁾(k_r·r_curr) / H_m⁽¹⁾(k_r·r_prev)
                     U(m, k_n) *= ratio

             7d: IFFT (φ 向):
                 对每个 k_n: V(φ, k_n) = IFFT_φ[U(m, k_n)]

             7e: IDMFT (z 向):
                 对每个 φ: u(φ, z_j) = Σ_n V(φ, k_n)·φ_n(z_j)

             7f: 折射 + 扩散修正:
                 u *= exp(j k₀ (n_atm - 2) Δr)
                 u *= √(r_prev/r_curr)

             7g: 地形遮蔽 + 材料处理:
                 for each φ: if z < z_terrain(φ, r_i):
                     导体 → u = 0
                     非导体 → u *= exp(j k₀ (n_mat - n_atm) Δr)
             7h: 上边界吸收:
                 u[:, z_top] *= cosine_taper(z)
             
             保存 u_full[i, φ, z] 用于后处理

步骤 8: 提取接收场
        φ_idx = argmin|φ_p - φ_B|
        z_idx = argmin|z_k - z_rx|
        E_rx = |u_full[r_B, φ_idx, z_idx]|

步骤 9: 路径损耗
        E_fs = r₀ / r_B
        L_fs = 20log₁₀(f_MHz) + 20log₁₀(r_km) + 32.45
        L_PE = L_fs - 20log₁₀(E_rx / E_fs)

步骤 10: 输出
        → 路径损耗 L_PE (dB)
        → E_rx (V/m), E_fs (V/m)
        → 可选：完整 u(r, φ, z) 体数据（用于 3D 可视化）
```

---

## 8. 数值验证方法

### 8.1 自由空间基准验证

**场景**：无地形、无障碍物、平坦地面远在计算域之外。

**预期**：$E_{\text{rx}} \approx E_{\text{fs}} = 1/r$，即 $L_{\text{PE}} \approx L_{\text{fs}}$。

**验证标准**：两者偏差 $< 0.5$ dB（数值误差来自网格离散化和吸收边界）。

### 8.2 双射线模型对比

**场景**：平坦地面 + 地面反射。

**预期**：PE 应产生与双射线模型一致的地面反射干涉条纹（峰值和谷值的距离周期正比于 $\lambda$）。

### 8.3 墙体绕射——单刃峰衍射

**场景**：当前简单场景（天线 $\rightarrow$ 墙 $\rightarrow$ 接收点）。

**预期**：墙后阴影区损耗可以用 Knife-Edge Diffraction（KED）公式对比。

Fresnel-Kirchhoff 衍射参数：

$$
v = h \sqrt{\frac{2}{\lambda}\left(\frac{1}{d_1} + \frac{1}{d_2}\right)}
$$

其中 $h$ = 墙顶到视线的高度差，$d_1$ = 天线到墙的距离（5 m），$d_2$ = 墙到接收点的距离。

衍射损耗：

$$
L_{\text{diff}}(\text{dB}) = 6.9 + 20 \log_{10}\!\left(\sqrt{(v-0.1)^2+1} + v - 0.1\right)
$$

PE 计算结果应与 KED 公式在 $\pm 3$ dB 内一致。

### 8.4 与论文结果对比

- 论文图 3-7：障碍物绕射场分布伪彩图（CPE vs Quasi-3DPE）
- 论文图 4-7：建筑物绕射对比（CPE vs MOM）
- 论文图 5-10：双向 PE 的建筑物反射效应

---

## 9. 横向绕射验证

> 完整 2D CPE 与简化径向 2D-PE 的**本质区别**在于能否捕获横向绕射（lateral diffraction）。以下验证来自有限宽墙体（$Y$ 方向 16m）场景。

### 9.1 验证设置

| 参数 | 值 |
|------|-----|
| 墙体尺寸 | $Y \in [-8\text{m}, 8\text{m}]$，$Z \in [0\text{m}, 5\text{m}]$，$X = 0\text{m}$ |
| 天线位置 | $(-5, 0, 6)$ |
| 对比接收点 A | $(3, 0, 2)$ —— 墙后阴影区中心 |
| 对比接收点 B | $(3, 8, 2)$ —— 墙后阴影区边缘（墙端侧面） |
| 频率 | 2.8 GHz |

### 9.2 中心 vs 边缘的场强差异

| 接收点 | $y$ 坐标 | $E$ (V/m) | 相对损耗 | 物理原因 |
|--------|--------|---------|---------|---------|
| 中心 $(3,0,2)$ | 0m | **较强** | 参考值 | 墙两侧绕射能量同时汇聚至中心 |
| 边缘 $(3,8,2)$ | 8m | **较弱** | 中心强约 **18 dB** | 仅单侧墙端贡献绕射能量 |

### 9.3 物理机理

有限宽墙体在 $Y$ 方向不是无限延伸。当电波遇到墙顶时发生垂直绕射，而在墙的 $Y$ 方向两端发生横向绕射：

```
     俯视图 (y-z 平面)                  横向绕射路径
     ────────────────                  ──────────────
           ↑                                ↑
    ───────┼──── 墙顶 (top)           墙左端   墙右端
           │                           ╲      ╱
     阴影区  │                            ╲  ╱
           │   接收点 A (中心)           接收点 A
           │   接收点 B (边缘)          (两侧能量汇聚)
```

- **中心点 ($y=0$)**：距墙左端 8m、右端 8m。绕射能量从两侧同时到达，场强增强
- **边缘点 ($y=8$)**：紧贴墙右端。绕射能量仅来自左端（侧面）和顶部，缺少右侧贡献

这一效应在简化径向 2D-PE（仅沿单一 $\varphi$ 线推进）中**完全无法体现**。只有完整 2D CPE（含 $\varphi$-FFT）才能捕获。

### 9.4 验证方法

在 GUI 中移动接收点沿 $Y$ 方向扫描（固定 $r=3$m, $z=2$m），绘制场强随 $y$ 的变化曲线：
- 预期：$y=0$ 处为峰值，向 $y=\pm 8$m 方向逐渐下降，呈近似对称分布
- 峰值与边缘的差值约 18 dB
- 与全波仿真（MoM）定性趋势一致

---

## 10. 预期物理现象

### 10.1 直射区的干涉条纹

- **原因**：直射波 + 地面反射波相干叠加
- **表现**：$u(r,z)$ 在高度方向出现交替的亮暗条纹
- **周期**：取决于频率和天线高度

### 10.2 墙体后的阴影边界

- **墙顶以上**：直射波 $\rightarrow$ 场强接近自由空间
- **墙顶以下**：阴影区 $\rightarrow$ 绕射波微弱（典型 $-15$ 到 $-25$ dB 附加损耗）
- **过渡区**：阴影边界附近有明显振荡（绕射波的 Fresnel 积分特征）

### 10.3 倏逝波的衰减

大角度传播（$|k_z| \gg k_0$）：在 SSFT 中自动指数衰减。无需手动设置角度限制——Feit-Fleck 近似在 $30^\circ$ 以内准确，$30^\circ$ 以上自动衰减。

### 10.4 横向绕射（完整 2D CPE 独有）

- **物理机制**：有限宽障碍物的 $Y$ 方向两端产生横向绕射，能量在 $\varphi$ 方向上重新分布
- **表现**：有限宽墙后的阴影区中，**中心点场强高于边缘点**（两侧能量汇聚）
- **差异幅度**：对于 16m 宽墙体在 2.8 GHz 下，中心-边缘差约 18 dB
- **与简化模型对比**：此效应只在包含 $\varphi$-FFT 的完整 2D CPE 中出现，简化径向 2D-PE 无法捕获

### 10.5 与频率的关系

| 频率 | 波长 | 绕射能力 | 阴影区损耗 |
|------|------|---------|-----------|
| 100 MHz | 3 m | 强（波长长，绕射强） | 较小 |
| 2.8 GHz | 0.107 m | 中等 | 中等 |
| 30 GHz | 0.01 m | 弱（近光传播） | 很大（几乎完全遮挡） |

### 10.6 有损介质穿透

PE 折射项 $\exp(j k_0 (n-2) \Delta r)$ 中的 $n$ 可为复折射指数，处理非理想导体的穿透损耗。

**复折射指数**：

$$
n = \sqrt{\varepsilon_r - j\frac{\sigma}{\omega \varepsilon_0}}
$$

其中 $\omega = 2\pi f$，$\varepsilon_0 = 8.854 \times 10^{-12}$ F/m。

**网格点处理**：对 $z < z_{\text{terrain}}(\varphi, r)$ 的每个点：
- $\sigma > 10^5$ S/m（良导体，如铝、铜）$\rightarrow$ 场强设为零（PEC）
- $\sigma \leq 10^5$ S/m（损失介质，如土壤、混凝土）$\rightarrow$ 施加材料折射修正：

$$
u_{\text{mat}} = u_{\text{air}} \cdot \exp\!\big(j k_0 (n_{\text{mat}} - n_{\text{air}}) \Delta r\big)
$$

其中：

$$
\exp\!\big(j k_0 (n - n_{\text{air}}) \Delta r\big) = \underbrace{\exp\!\big(j k_0 \text{Re}(n - n_{\text{air}}) \Delta r\big)}_{\text{相位延迟}} \cdot \underbrace{\exp\!\big(-k_0 \text{Im}(n) \Delta r\big)}_{\text{幅度衰减}}
$$

**当前场景材料**：

| 对象 | 材料 | $\varepsilon_r$ | $\sigma$ (S/m) | $n$ (2.8 GHz) | 处理 |
|------|------|----------------|---------------|------|
| terrain | 中等干燥地面 | 15 | 0.01 | $3.87 - j0.008$ | 介质穿透（$\sim 0.4$ dB/m） |
| wall | 金属铝 | 1 | $3.8 \times 10^7$ | $11044 - j11044$ | PEC 杀场 |

**扩展性**：为任意场景对象添加 `extra.material = {"label": "...", "eps_r": ..., "sigma": ...}` 即可支持新材质。

### 10.7 TWPE — 双向 PE（论文第 5 章）

标准 PE 只处理前向传播——障碍物反射回天线的能量被丢弃。TWPE 通过后向 PE + 迭代收敛捕获多 bounce 反射。

**后向 PE**（论文式 5-2）：

$$
u(r-\Delta r, \varphi, z) = \text{DMFT}^{-1}\text{IFFT}\!\left[\frac{H_m^{(1)}(k_r (r-\Delta r))}{H_m^{(1)}(k_r r)} \cdot \text{FFT·DMFT}[u]\right] \cdot e^{j k_0 (n-2) \Delta r}
$$

与前向 PE 的区别：传播方向反向，汉克尔比分母/分子对调，扩散修正 $\sqrt{r/(r-\Delta r)} > 1$（趋近源时能量汇聚）。

**反射源检测**：沿每个 $\varphi$ 方向扫描 $z_{\text{terrain}}(r)$ 的突变（$\Delta z > \Delta z_{\text{grid}}$），在突变处设定后向初始场：

$$
u_b(r_{\text{obs}}, \varphi, z < z_{\text{terrain}}) = -u_f(r_{\text{obs}}, \varphi, z) \quad \text{(PEC)}
$$

仅导体障碍物（$\sigma > 10^5$ S/m）产生反射。

**迭代收敛**（论文式 5-4）：

$$
\frac{\|u_{\text{total}}^n - u_{\text{total}}^{n-1}\|}{\|u_{\text{total}}^{n-1}\|} < \varepsilon = 10^{-3}
$$

最大 10 次迭代防止无限循环。单障碍物场景 1 次迭代即收敛。

### 10.8 边界条件汇总

| 方位角 | $\varphi$ | 周期 FFT（闭环 $0 = 2\pi$） | 无需吸收边界 |

---

## A. 符号表

| 符号 | 含义 | 单位 |
|------|------|------|
| $\psi$ | 电磁场分量 | V/m |
| $u$ | 衰减函数 $u = e^{-j k_0 x} \psi$ | V/m |
| $k_0$ | 自由空间波数 $2\pi f/c$ | rad/m |
| $k_r$ | 径向波数 $\sqrt{k_0^2 n^2 - k_z^2}$ | rad/m |
| $k_z$ | 垂直方向波数 | rad/m |
| $n$ | 大气折射指数 | 无量纲 |
| $m$ | 角向模式序号（$\varphi$-FFT 产生） | $-$ |
| $H_m^{(1)}$ | 第一类 $m$ 阶汉克尔函数 | $-$ |
| $\Delta r$ | 径向步长 | m |
| $\Delta z$ | 高度分辨率 | m |
| $N_r$ | 径向步数 | $-$ |
| $N_\varphi$ | 方位角 FFT 点数 | $-$ |
| $N_z$ | 高度网格点数 | $-$ |
| $\text{FFT}_{\varphi,z}$ | 沿 $\varphi$ 和 $z$ 的二维傅里叶变换 | $-$ |
| $\varphi$ | 方位角 | rad |
| $r$ | 径向距离 | m |
| $\sigma_z$ | 初始高斯波束半宽度 | m |
| $L_{\text{PE}}$ | PE 路径损耗 | dB |
| $L_{\text{fs}}$ | 自由空间路径损耗 | dB |
| $E_{\text{rx}}$ | 接收场强 | V/m |
| $E_{\text{fs}}$ | 自由空间参考场强 | V/m |

---

## B. 参考文献

[1] 汪路遥. 电波传播的柱坐标系抛物方程模型及其应用研究[D]. 西南交通大学, 2021.

[2] Dockery G D. Modeling electromagnetic wave propagation in the troposphere using the parabolic equation[J]. IEEE Trans. Antennas Propag., 1988, 36(10): 1464-1470.

[3] Kuttler J R, Janaswamy R. Improved Fourier transform methods for solving the parabolic wave equation[J]. Radio Science, 2002, 37(2): 1021.

[4] Hardin R H, Tappert F D. Applications of the split-step Fourier method to the numerical solution of nonlinear and variable coefficient wave equations[J]. SIAM Review, 1973, 15(2): 423.
