# DemoSpeedup 对比方法分析文档

## 目录

1. [问题一：DemoSpeedup 与其他下采样方法对比时是否控制了总轨迹长度一致](#问题一demospeedup-与其他下采样方法对比时是否控制了总轨迹长度一致)
2. [问题二：对比的下采样方法详细介绍](#问题二对比的下采样方法详细介绍)
3. [Section 7.1：与其他演示加速方法的详细对比](#section-71与其他演示加速方法的详细对比)

---

## 问题一：DemoSpeedup 与其他下采样方法对比时是否控制了总轨迹长度一致

### 结论

**没有控制总轨迹长度（动作步数）一致。** DemoSpeedup 在对比实验中有意地使用了更少的动作步数（约为原始的 50%），其核心思想是：通过熵引导的智能下采样，在减少动作步数的同时保持任务完成能力。

### 代码证据

#### 1. Chunk Size 减半

在 README.md 中明确指出：
- **代理策略训练（原始）**：`chunk_size=50`（ACT）或 `chunk_size=48`（DP）
- **加速策略训练（DemoSpeedup）**：`chunk_size=25`（ACT）或 `chunk_size=24`（DP）

```bash
# 原始策略训练
python act/imitate_episodes.py --chunk_size 50 ...

# 加速策略训练
python act/imitate_episodes.py --chunk_size 25 --speedup ...
```

#### 2. Action Sequence 和 Execution Length 减半

在 `robobase/robobase/workspace.py`（第 115-117 行）中：

```python
if cfg.speedup:
    cfg.action_sequence = cfg.action_sequence // 2
    cfg.execution_length = cfg.execution_length // 2
```

当启用 speedup 时，动作序列长度和执行长度都被直接减半。

#### 3. 高增益控制器补偿

为了在更少的动作步数下完成同样的任务，DemoSpeedup 使用了**高增益（High Gain）控制器**：

在 Aloha 环境中（`aloha/act/sim_env.py`，第 42-47 行）：
```python
if not speedup:
    xml_path = os.path.join(XML_DIR, f"bimanual_viperx_transfer_cube.xml")
else:
    xml_path = os.path.join(XML_DIR, f"bimanual_viperx_transfer_cube_high_gain.xml")
```

在 BiGym 环境中（`robobase/robobase/workspace.py`，第 92-93 行）：
```python
if cfg.speedup:
    factory.HIGH_GAIN = True
```

#### 4. 评估环境一致

虽然动作步数不同，但评估是在**相同的仿真环境**中进行的，最大时间步数（`max_timesteps`）保持一致。对比的核心指标是**任务成功率**和**平均回报**，而非轨迹长度本身。

### 总结

DemoSpeedup 的实验设计逻辑是：
- 原始方法：使用完整动作序列 + 普通控制器增益
- DemoSpeedup：使用 ~50% 的动作序列 + 高增益控制器

两者在**相同的环境和任务评估标准**下进行对比，但**动作步数（轨迹长度）并不相同**。DemoSpeedup 的核心贡献正是在于用更少的步数实现同等甚至更好的任务表现。

---

## 问题二：对比的下采样方法详细介绍

DemoSpeedup 在代码中实现了以下几种方法进行对比：

### 方法概览

| 方法 | 代码位置 | 核心思想 |
|------|---------|---------|
| 原始策略（无下采样） | `imitate_episodes.py` | 使用完整示教轨迹训练 |
| DP Waypoint Selection（恒定阈值） | `awe_entropy.py: dp_waypoint_selection()` | 基于几何距离的动态规划路标选择 |
| DemoSpeedup（熵引导 DP） | `awe_entropy.py: dp_entropy_waypoint_selection()` | 熵加权的动态规划路标选择 |
| 运行时熵引导下采样 | `act_utils.py: process_action_label()` | 基于熵标签的自适应跳帧 |

---

### 方法一：原始策略（无下采样）

最简单的基线方法——直接使用完整的示教轨迹进行策略学习，不进行任何下采样处理。

**参数设置：**
- Chunk Size：ACT=50，DP=48
- 控制器增益：普通
- 训练轮数：16000 epochs

---

### 方法二：DP Waypoint Selection（恒定阈值几何距离动态规划）

**代码位置：** `aloha/act/awe_entropy.py`，函数 `dp_waypoint_selection()`（第 10-100 行）

#### 核心思想

使用动态规划（Dynamic Programming）选择最少的路标点（waypoints），使得每段轨迹的几何重建误差都不超过给定阈值 $\epsilon$。

#### 几何距离计算

对于轨迹中的每个点 $\mathbf{p}_i$，计算它到路标点 $\mathbf{p}_j$ 和 $\mathbf{p}_k$ 连线的距离：

**位置距离（Point-to-Line Distance）：**

给定线段起点 $\mathbf{p}_j$ 和终点 $\mathbf{p}_k$，查询点 $\mathbf{p}_i$ 到线段的距离为：

$$
\mathbf{v}_{line} = \mathbf{p}_k - \mathbf{p}_j
$$

$$
\mathbf{v}_{point} = \mathbf{p}_i - \mathbf{p}_j
$$

$$
t = \text{clamp}\left(\frac{\mathbf{v}_{point} \cdot \mathbf{v}_{line}}{\|\mathbf{v}_{line}\|^2}, 0, 1\right)
$$

$$
\mathbf{proj} = \mathbf{p}_j + t \cdot \mathbf{v}_{line}
$$

$$
d_{pos}(\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k) = \|\mathbf{p}_i - \mathbf{proj}\|
$$

**组合距离（位置 + 四元数）：**

代码位置：`get_all_geometric_distance_gpu()`（第 252-304 行）

对四元数部分（欧拉角前3维）使用相同的 Point-to-Line 距离计算，最终距离为：

$$
d(\mathbf{s}_i, \mathbf{s}_j, \mathbf{s}_k) = d_{pos}(\mathbf{s}_i^{pos}, \mathbf{s}_j^{pos}, \mathbf{s}_k^{pos}) + d_{quat}(\mathbf{s}_i^{quat}, \mathbf{s}_j^{quat}, \mathbf{s}_k^{quat})
$$

其中 $\mathbf{s}^{pos}$ 为前 3 维位置，$\mathbf{s}^{quat}$ 为第 3-6 维旋转分量。

#### 轨迹重建误差

给定路标点集合 $W = \{w_0, w_1, ..., w_m\}$，轨迹重建误差为每段内所有点到对应线段距离的最大值：

$$
E(W) = \max_{i} \left( d(\mathbf{s}_i, \mathbf{s}_{w_{seg(i)}}, \mathbf{s}_{w_{seg(i)+1}}) \right)
$$

其中 $seg(i)$ 表示点 $i$ 所在线段的起始路标索引。

#### 动态规划算法

**伪代码：**

```
输入: 轨迹状态序列 S = {s_0, s_1, ..., s_{N-1}}, 误差阈值 ε
输出: 最小路标点集合 W

# 预计算所有点对点的几何距离矩阵
all_distance = compute_all_geometric_distance(S)  # 三维距离矩阵 [N, N, N]

# 初始化备忘录表
memo[0] = (0, [])
memo[1] = (1, [1])

# 自底向上动态规划
for i = 2 to N-1:
    min_waypoints = +∞
    best_waypoints = []
    
    for k = 1 to i-1:
        # 计算从 k 到 i 的轨迹重建误差
        traj_err = geometric_trajectory_error(S[k:i+1], all_distance[k:i+1])
        
        if traj_err < ε:
            # 子问题的路标数 + 当前点
            total_count = 1 + memo[k].count
            if total_count < min_waypoints:
                min_waypoints = total_count
                best_waypoints = memo[k].waypoints + [i]
    
    memo[i] = (min_waypoints, best_waypoints)

# 获取最终路标点集合
W = memo[N-1].waypoints ∪ {N-1}  # 确保最后一帧是路标
return W
```

---

### 方法三：DemoSpeedup（熵引导 DP Waypoint Selection）

**代码位置：** `aloha/act/awe_entropy.py`，函数 `dp_entropy_waypoint_selection()`（第 110-204 行）

#### 核心思想

在 DP Waypoint Selection 的基础上，引入**动作熵（action entropy）** 来动态调整每个时间步的误差阈值。高熵区域（需要精确控制）使用更严格的阈值，低熵区域（简单运动）使用更宽松的阈值。

#### 熵计算方法

##### 方法 A：核密度估计（KDE）熵

**代码位置：** `aloha/act/detr/models/entropy_utils.py`，类 `KDE` 的 `kde_entropy()` 方法（第 47-72 行）

给定策略在时间步 $t$ 的 $M$ 个动作采样 $\{a_1, a_2, ..., a_M\}$：

1. **高斯核密度估计：**

$$
K(\mathbf{a}_i, \mathbf{a}_j) = \exp\left(-\frac{\|\mathbf{a}_i - \mathbf{a}_j\|^2}{2h^2}\right)
$$

其中 $h$ 为带宽（bandwidth），代码中设为 1。

2. **密度估计：**

$$
\hat{p}(\mathbf{a}_i) = \frac{1}{M} \sum_{j=1}^{M} K(\mathbf{a}_i, \mathbf{a}_j)
$$

3. **熵估计：**

$$
H = -\frac{1}{M} \sum_{i=1}^{M} \log(\hat{p}(\mathbf{a}_i) + \epsilon)
$$

其中 $\epsilon = 10^{-8}$ 防止对数下溢。

##### 方法 B：Kozachenko-Leonenko（KL）k-NN 熵估计

**代码位置：** `aloha/act/detr/models/entropy_utils.py`，函数 `kozachenko_leonenko_entropy()`（第 18-31 行）

$$
H_{KL} = \psi(n) - \psi(k) - d \cdot \log\left(\text{mean}(\mathbf{d}_{k\text{-NN}})\right)
$$

其中：
- $\psi(\cdot)$ 为 digamma 函数
- $n$ 为样本数量
- $k$ 为 k-NN 的邻居数（默认 $k=5$）
- $d$ 为维度
- $\mathbf{d}_{k\text{-NN}}$ 为每个点到其第 $k$ 个最近邻的平均距离

#### 时序聚合（Temporal Aggregation）

**代码位置：** `aloha/act/imitate_episodes.py`（第 276-303 行）

为了获得更稳定的熵估计，使用时序聚合：

$$
\hat{H}_t = \text{KDE\_entropy}\left(\bigcup_{t'} \mathbf{S}_{t',t}\right)
$$

其中 $\mathbf{S}_{t',t}$ 为在查询时间 $t'$ 时获得的对应时间步 $t$ 的动作采样集合。

#### HDBSCAN 聚类标签生成

**代码位置：** `aloha/act/imitate_episodes.py`，函数 `hdbscan_with_custom_merge()`（第 378-422 行）

利用 HDBSCAN 聚类将时间步分为两类：
- **标签 0（精确区域）：** 低熵，需要精确控制的区域——对应下采样时跳步较少
- **标签 1（非精确区域）：** 高熵或过渡区域——对应下采样时可以跳过更多步

**伪代码：**

```
输入: 熵序列 entropy[0..N-1], 时间步序列 timesteps[0..N-1]
输出: 标签序列 labels[0..N-1], 其中 0=精确, 1=非精确

# 1. 数据标准化
entropy_norm = (entropy - mean(entropy)) / std(entropy)
timesteps_norm = (timesteps - mean(timesteps)) / std(timesteps)
data = stack(timesteps_norm, entropy_norm)  # [N, 2]

# 2. HDBSCAN 聚类
clusterer = HDBSCAN(min_cluster_size=5)
initial_labels = clusterer.fit_predict(data)

# 3. 前 50 帧设为非精确（开始阶段通常是初始运动）
initial_labels[0:50] = -1

# 4. 精炼标签：低熵聚类标记为 0（精确），其余标记为 1（非精确）
for each cluster label:
    if mean(cluster_entropy) < 0:  # 标准化后低于平均值
        refined_labels[cluster] = 0  # 精确区域
    else:
        refined_labels[cluster] = -1  # 非精确

# 5. 最终标签：取绝对值（-1 → 1, 0 → 0）
labels = |refined_labels|
return labels
```

#### 熵权重转换

**代码位置：** `aloha/act/awe_entropy.py`，函数 `calculate_weights_from_entropy()`（第 102-107 行）

$$
w_t = H_t \times 0.4
$$

#### 熵加权误差阈值

$$
\epsilon_t = \epsilon \times w_t
$$

其中 $\epsilon$ 为基础误差阈值，$w_t$ 为时间步 $t$ 的熵权重。

#### 动态规划算法（熵引导版本）

**伪代码：**

```
输入: 轨迹 S, 熵序列 entropy, 基础误差阈值 ε
输出: 最小路标点集合 W

# 1. 计算熵权重和逐步误差阈值
weights = entropy * 0.4
err_threshold_per_step = ε * weights  # 每个时间步的误差阈值

# 2. 预计算所有几何距离
all_distance = compute_all_geometric_distance(S)

# 3. 初始化备忘录表
memo[0] = (0, [])
memo[1] = (1, [1])

# 4. 自底向上动态规划
for i = 2 to N-1:
    min_waypoints = +∞
    best_waypoints = []
    
    for k = max(1, i-4) to i-1:  # 注意：限制搜索窗口为 4
        # 计算从 k 到 i 每个点的重建误差
        traj_err, err_list = geometric_trajectory_error(
            S[k:i+1], all_distance[k:i+1], return_list=True
        )
        
        # 关键区别：逐点检查误差是否满足熵加权阈值
        if ALL(err_list[j] ≤ err_threshold_per_step[k+j]) for j in [k..i]:
            total_count = 1 + memo[k].count
            if total_count < min_waypoints:
                min_waypoints = total_count
                best_waypoints = memo[k].waypoints + [i]
    
    memo[i] = (min_waypoints, best_waypoints)

W = memo[N-1].waypoints ∪ {N-1}
return W
```

**与恒定阈值 DP 的关键区别：**

1. **逐步自适应阈值：** 使用 `err_threshold_per_step[k:i+1]` 替代全局固定 `ε`
2. **逐点检查：** 使用 `ALL(err_list ≤ err_threshold_per_step)` 而非 `total_err < ε`
3. **搜索窗口限制：** `k` 的搜索范围从 `range(1, i)` 缩小到 `range(max(1, i-4), i)`，提高计算效率

---

### 方法四：运行时熵引导下采样（训练时使用）

**代码位置：** `aloha/act/act_utils.py`，函数 `process_action_label()`（第 175-214 行）

在 Aloha 环境中，这是训练时实际使用的下采样方法。

#### 核心参数

- `low_v = 2`：低熵区域（精确区域，label=0）的跳步距离
- `high_v = 4`：高熵区域（非精确区域，label=1）的跳步距离

#### 伪代码

```
输入: 动作序列 action[0..H-1], 标签序列 label[0..H-1]
输出: 下采样后的动作序列

low_v = 2   # 精确区域跳步
high_v = 4  # 非精确区域跳步
indices = []
i = -1

while i < H:
    if label[i] == 0 AND i + low_v < H:
        # 精确区域：跳 2 步
        i = i + low_v
        indices.append(i)
    
    elif label[i] == 1:
        if i + high_v < H AND ALL(label[i:i+high_v] == 1):
            # 连续非精确区域：跳 4 步
            i = i + high_v
            indices.append(i)
        else:
            # 找到下一个精确区域（label=0）的位置
            next_zero = find_next(label[i+1:] == 0)
            if next_zero exists:
                i = i + 1 + next_zero
                indices.append(i)
            else:
                break
    else:
        i = i + 1

downsampled_action = action[indices]
return downsampled_action
```

#### Robobase 版本

**代码位置：** `robobase/robobase/replay_buffer/uniform_replay_buffer.py`，函数 `downsample_action_with_labels()`（第 67-106 行）

与 Aloha 版本逻辑相似，但有细微差别：

```
输入: 动作序列 action[0..H-1], 标签序列 label[0..H-1], 目标长度 chunk_len
输出: 下采样后的动作序列

low_v = 2
high_v = 4
indices = []
i = -2

while len(indices) < chunk_len - 2 AND i < H:
    if i + high_v < H AND ALL(label[i:i+high_v] == 1):
        # 非精确区域优先：跳 4 步
        i = i + high_v
        indices.append(i)
    elif i + low_v < H:
        # 默认：跳 2 步
        i = i + low_v
        indices.append(i)
    else:
        i = i + 1

# 边界处理：确保末尾有足够的点
if indices:
    last_i = indices[-1]
    # 补充边界点以确保平滑过渡
    ...

downsampled_action = action[indices]
return downsampled_action
```

**Robobase 版本的关键区别：**
1. 起始索引为 -2（而非 -1）
2. 有明确的目标长度限制 `chunk_len - 2`
3. 优先检查高熵跳步条件
4. 包含边界处理逻辑

---

### 各方法对比总结

| 特性 | 原始策略 | DP Waypoint（恒定阈值） | DemoSpeedup（熵引导 DP） | 运行时熵下采样 |
|------|---------|------------------------|------------------------|--------------|
| 下采样方式 | 无 | 几何误差 DP | 熵加权几何误差 DP | 基于标签的跳帧 |
| 使用熵信息 | ❌ | ❌ | ✅ | ✅（间接，通过标签） |
| 自适应性 | 无 | 全局阈值 | 逐步自适应阈值 | 二级跳步 |
| Chunk Size | 50/48 | 可变 | 可变 | 25/24 |
| 控制器增益 | 普通 | 普通 | 高增益 | 高增益 |
| 计算复杂度 | O(1) | O(N²) | O(N·W)，W≤4 | O(N) |
| 应用阶段 | - | 离线标注 | 离线标注 | 在线训练 |

---

### 核心公式汇总

#### 1. 几何距离（Point-to-Line）

$$
d(\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k) = \left\|\mathbf{p}_i - \left(\mathbf{p}_j + \text{clamp}\left(\frac{(\mathbf{p}_i - \mathbf{p}_j) \cdot (\mathbf{p}_k - \mathbf{p}_j)}{\|\mathbf{p}_k - \mathbf{p}_j\|^2}, 0, 1\right) \cdot (\mathbf{p}_k - \mathbf{p}_j)\right)\right\|
$$

#### 2. KDE 熵估计

$$
H_{KDE} = -\frac{1}{M} \sum_{i=1}^{M} \log\left(\frac{1}{M}\sum_{j=1}^{M} \exp\left(-\frac{\|\mathbf{a}_i - \mathbf{a}_j\|^2}{2h^2}\right) + \epsilon\right)
$$

#### 3. Kozachenko-Leonenko 熵估计

$$
H_{KL} = \psi(n) - \psi(k) - d \cdot \log\left(\frac{1}{M}\sum_{i=1}^{M} \bar{d}_{k\text{-NN}}(i)\right)
$$

#### 4. 熵加权误差阈值

$$
\epsilon_t = \epsilon_{base} \times (H_t \times 0.4)
$$

#### 5. DP 最优子结构

$$
\text{memo}[i] = \min_{k \in [1, i)} \left(1 + \text{memo}[k].\text{count}\right) \quad \text{s.t.} \quad E(S[k:i+1]) < \epsilon
$$

#### 6. 熵引导 DP 最优子结构

$$
\text{memo}[i] = \min_{k \in [\max(1, i-4), i)} \left(1 + \text{memo}[k].\text{count}\right) \quad \text{s.t.} \quad \forall t \in [k, i]: e_t \leq \epsilon_t
$$

---

## Section 7.1：与其他演示加速方法的详细对比

本节重点分析论文 Section 7.1 中 DemoSpeedup 与四种基线下采样方法的对比。这四种方法用于替换 DemoSpeedup 中的熵引导分段加速模块，所有方法在相同的实验环境和评估标准下进行比较。

### 对比方法概览

| 方法 | 类型 | 下采样方式 | 是否需要 Oracle/先验 | 代码位置 |
|------|------|-----------|---------------------|---------|
| **Contact Oracle** | 基于接触的启发式 | 接触事件驱动的精度分区 | ✅ 需要 3D 接触信息 | `sim_env.py` 中的接触检测逻辑 |
| **AWE\*** | 动态规划 + 熵加权 | 调整阈值使轨迹缩短至 2× | ❌ | `awe_entropy.py: dp_waypoint_selection()` / `dp_entropy_waypoint_selection()` |
| **恒定 2×** | 均匀下采样 | 每隔 1 帧取样 | ❌ | `uniform_replay_buffer.py` 第 840 行（注释） |
| **恒定 3×** | 均匀下采样 | 每隔 2 帧取样 | ❌ | 类似 `action[::3]` |

---

### 基线方法一：Contact Oracle（基于接触的 Oracle 标签）

#### 方法描述

Contact Oracle 是一种基于仿真器**接触信息（contact information）**的启发式方法，用于将轨迹划分为**高精度区域**和**低精度区域**。其核心假设是：**物体接触状态发生变化的时刻（如夹爪抓取物体、物体放置到另一物体上）前后需要高精度控制**，其余时段可以使用低精度（即可大步跳帧）。

#### 划分规则

在操作场景中，系统监测所有物体对（geom pair）的接触状态。当检测到以下事件时，标记该时刻前后的固定时间窗口为**高精度区域**：

1. **新的物体对接触事件**：例如夹爪手指与目标物体首次接触
2. **物体脱离事件**：例如物体从桌面抬起（不再接触桌面）
3. **物体间接触变化**：例如目标物体与放置位置首次接触

其余时间段标记为**低精度区域**。

#### 代码中的接触检测逻辑

Contact Oracle 的核心依赖仿真器提供的接触对（contact pair）信息。在代码中，这些接触检测逻辑已在 reward 函数中实现（用于评估任务完成度），相同的逻辑可以直接复用于 Contact Oracle 标签生成。

**代码位置：** `aloha/act/sim_env.py`

##### TransferCube 任务的接触检测（第 204-234 行）

```python
def get_reward(self, physics):
    # 获取所有接触对
    all_contact_pairs = []
    for i_contact in range(physics.data.ncon):
        id_geom_1 = physics.data.contact[i_contact].geom1
        id_geom_2 = physics.data.contact[i_contact].geom2
        name_geom_1 = physics.model.id2name(id_geom_1, "geom")
        name_geom_2 = physics.model.id2name(id_geom_2, "geom")
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)

    # 检测各类接触事件
    touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
    touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
    touch_table = ("red_box", "table") in all_contact_pairs
```

##### Insertion 任务的接触检测（第 259-312 行）

```python
def get_reward(self, physics):
    # ...检测接触对...
    touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
    touch_left_gripper = (
        ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        # ...更多 socket 变体...
    )
    peg_touch_table = ("red_peg", "table") in all_contact_pairs
    peg_touch_socket = (
        ("red_peg", "socket-1") in all_contact_pairs
        or ("red_peg", "socket-2") in all_contact_pairs
        # ...更多组合...
    )
    pin_touched = ("red_peg", "pin") in all_contact_pairs
```

#### 伪代码

```
输入: 演示轨迹 D = {(s_0, a_0), ..., (s_{T-1}, a_{T-1})}, 时间窗口 Δt
输出: 标签序列 labels[0..T-1], 其中 0=高精度, 1=低精度

# 初始化所有帧为低精度
labels = [1] * T

# 记录上一时刻的接触状态集合
prev_contacts = get_contact_pairs(s_0)

for t = 1 to T-1:
    curr_contacts = get_contact_pairs(s_t)

    # 检测新增接触对（新物体接触事件）
    new_contacts = curr_contacts - prev_contacts
    # 检测消失接触对（物体脱离事件）
    lost_contacts = prev_contacts - curr_contacts

    if new_contacts ≠ ∅ OR lost_contacts ≠ ∅:
        # 在接触变化时刻前后 Δt 帧标记为高精度
        for τ = max(0, t - Δt) to min(T-1, t + Δt):
            labels[τ] = 0  # 高精度

    prev_contacts = curr_contacts

return labels
```

#### 对于 TransferCube 任务的具体接触事件

| 接触事件 | 含义 | 对应 contact pair |
|---------|------|------------------|
| 右夹爪抓取方块 | `touch_right_gripper` | `("red_box", "vx300s_right/10_right_gripper_finger")` |
| 方块离开桌面 | `NOT touch_table` | `("red_box", "table")` 消失 |
| 左夹爪接收方块 | `touch_left_gripper` | `("red_box", "vx300s_left/10_left_gripper_finger")` |
| 方块传递完成 | `touch_left_gripper AND NOT touch_table` | 组合条件 |

#### 对于 Insertion 任务的具体接触事件

| 接触事件 | 含义 | 对应 contact pair |
|---------|------|------------------|
| 双手分别抓取peg和socket | `touch_left_gripper AND touch_right_gripper` | 多个 socket/peg geom 组合 |
| 双手同时抬起 | 离开桌面 | `peg_touch_table`/`socket_touch_table` 消失 |
| peg 接触 socket | 对准阶段 | `("red_peg", "socket-*")` 出现 |
| peg 插入到 pin | 完成插入 | `("red_peg", "pin")` 出现 |

#### 局限性

- **依赖 Oracle 信息**：需要仿真器提供精确的 3D 接触对信息（`physics.data.contact`），真实世界中无法直接获取
- **依赖精确的 3D 先验**：需要预先知道所有物体的 geom 名称和可能的接触对
- **不适用于仅有 2D 相机输入的真实场景**

---

### 基线方法二：AWE*（动态规划路标选择 + 熵加权）

#### 方法描述

AWE*（Adjusted AWE with entropy weighting）基于 AWE（Action Waypoints from Entropy）方法的动态规划框架，但做了两个关键调整：

1. **调整阈值使轨迹缩短至约 2×**：原始 AWE 旨在提升成功率，其选择的路标点到达时间可能远长于原始演示。AWE* 调整了误差阈值 $\epsilon$，使得下采样后的轨迹长度大致为原始长度的一半。
2. **引入熵对近似误差进行加权**：原始 AWE 仅依赖关节角度轨迹的几何误差，AWE* 通过熵对每个时间步的误差阈值进行加权，使得高熵（低确定性）区域保留更多路标点。

#### 核心代码

AWE* 的基础实现对应 `dp_waypoint_selection()`，熵加权版本对应 `dp_entropy_waypoint_selection()`。

**代码位置：** `aloha/act/awe_entropy.py`

##### dp_waypoint_selection()（基础 AWE DP，第 10-100 行）

这是不使用熵加权的基础版本：

```python
def dp_waypoint_selection(
    env=None, actions=None, gt_states=None,
    err_threshold=None, initial_states=None,
    remove_obj=None, pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)

    num_frames = len(actions)
    initial_waypoints = [num_frames - 1]  # 最后一帧为路标

    memo = {}
    func = fast_geometric_waypoint_trajectory
    distance_func = (
        get_all_pos_only_geometric_distance_gpu if pos_only
        else get_all_geometric_distance_gpu
    )
    all_distance = distance_func(gt_states)

    # 初始化
    for i in range(num_frames):
        memo[i] = (0, [])
    memo[1] = (1, [1])

    # 自底向上 DP
    for i in range(2, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []
        for k in range(1, i):  # 搜索所有可能的分割点
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]
            total_traj_err = func(
                actions=actions[k:i+1], gt_states=gt_states[k:i+1],
                waypoints=waypoints,
                all_distance=all_distance[k:i+1, k:i+1, k:i+1]
            )
            if total_traj_err < err_threshold:  # 全局固定阈值
                subproblem_waypoints_count, subproblem_waypoints = memo[k]
                total_waypoints_count = 1 + subproblem_waypoints_count
                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]
        memo[i] = (min_waypoints_required, best_waypoints)

    _, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    waypoints = sorted(set(waypoints))
    return waypoints
```

##### dp_entropy_waypoint_selection()（熵加权 AWE*，第 110-204 行）

```python
def dp_entropy_waypoint_selection(
    env=None, actions=None, entropy=None, gt_states=None,
    err_threshold=None, initial_states=None,
    remove_obj=None, pos_only=False,
):
    if gt_states is None:
        gt_states = copy.deepcopy(actions)

    num_frames = len(actions)
    entropy_weights = calculate_weights_from_entropy(entropy)  # weights = entropy * 0.4
    all_err_threshold = err_threshold * entropy_weights  # 逐步自适应阈值

    initial_waypoints = [num_frames - 1]
    memo = {}
    func = fast_geometric_waypoint_trajectory
    distance_func = (
        get_all_pos_only_geometric_distance_gpu if pos_only
        else get_all_geometric_distance_gpu
    )
    all_distance = distance_func(gt_states)

    for i in range(num_frames):
        memo[i] = (0, [])
    memo[1] = (1, [1])

    for i in range(2, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []
        for k in range(max(1, i-4), i):  # 限制搜索窗口为 4
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]
            total_traj_err, all_traj_err = func(
                actions=actions[k:i+1], gt_states=gt_states[k:i+1],
                waypoints=waypoints,
                all_distance=all_distance[k:i+1, k:i+1, k:i+1],
                return_list=True
            )
            # 关键：逐点检查，每个时间步使用各自的熵加权阈值
            if (np.array(all_traj_err) <= all_err_threshold[k:i+1]).all():
                subproblem_waypoints_count, subproblem_waypoints = memo[k]
                total_waypoints_count = 1 + subproblem_waypoints_count
                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]
        memo[i] = (min_waypoints_required, best_waypoints)

    _, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    waypoints = sorted(set(waypoints))
    return waypoints
```

#### AWE vs AWE* 关键公式差异

##### 原始 AWE DP（`dp_waypoint_selection`）

**优化目标：**

$$
\min |W| \quad \text{s.t.} \quad \forall \text{segment } [w_j, w_{j+1}]: \max_{i \in [w_j, w_{j+1}]} d(s_i, s_{w_j}, s_{w_{j+1}}) < \epsilon
$$

其中 $\epsilon$ 为全局固定阈值。

**DP 递推：**

$$
\text{memo}[i] = \min_{k \in [1, i)} \left(1 + \text{memo}[k].\text{count}\right) \quad \text{s.t.} \quad E(S[k:i+1]) < \epsilon
$$

##### AWE*（`dp_entropy_waypoint_selection`）

**优化目标：**

$$
\min |W| \quad \text{s.t.} \quad \forall t: d(s_t, s_{w_{seg(t)}}, s_{w_{seg(t)+1}}) \leq \epsilon_t
$$

其中 $\epsilon_t = \epsilon_{base} \times (H_t \times 0.4)$ 为熵加权的逐步阈值。

**DP 递推（搜索窗口限制为 4）：**

$$
\text{memo}[i] = \min_{k \in [\max(1, i-4), i)} \left(1 + \text{memo}[k].\text{count}\right) \quad \text{s.t.} \quad \forall t \in [k, i]: e_t \leq \epsilon_t
$$

#### 伪代码：AWE* 完整流程

```
输入: 关节角度轨迹 Q = {q_0, q_1, ..., q_{N-1}},
      动作熵序列 H = {H_0, H_1, ..., H_{N-1}},
      基础误差阈值 ε_base
输出: 下采样后的路标点索引集合 W

# 步骤 1：计算熵权重
for t = 0 to N-1:
    w_t = H_t × 0.4

# 步骤 2：计算逐步误差阈值
for t = 0 to N-1:
    ε_t = ε_base × w_t

# 步骤 3：预计算几何距离矩阵
all_distance = compute_geometric_distance_gpu(Q)
# all_distance[i,j,k] = 点 i 到线段 (j,k) 的 point-to-line 距离

# 步骤 4：动态规划
memo[0] = (count=0, waypoints=[])
memo[1] = (count=1, waypoints=[1])

for i = 2 to N-1:
    best = (count=+∞, waypoints=[])

    for k = max(1, i-4) to i-1:
        # 计算 [k, i] 段内每个点的重建误差
        segment_errors = []
        for t = k to i:
            err_t = all_distance[t, k, i]  # 点 t 到线段 (k, i) 的距离
            segment_errors.append(err_t)

        # 逐点检查：每个点的误差不超过其对应的熵加权阈值
        if ALL(segment_errors[t-k] ≤ ε_{t}) for t in [k..i]:
            candidate_count = 1 + memo[k].count
            if candidate_count < best.count:
                best = (count=candidate_count, waypoints=memo[k].waypoints + [i])

    memo[i] = best

# 步骤 5：提取最终路标点
W = memo[N-1].waypoints ∪ {N-1}
W = sort(unique(W))

return W
```

#### AWE* 与 AWE 的核心区别总结

| 特性 | AWE (dp_waypoint_selection) | AWE* (dp_entropy_waypoint_selection) |
|------|---------------------------|--------------------------------------|
| 误差阈值 | 全局固定 $\epsilon$ | 逐步自适应 $\epsilon_t = \epsilon \times H_t \times 0.4$ |
| 误差检查 | 段内最大误差 < $\epsilon$ | 逐点误差 ≤ $\epsilon_t$ |
| 搜索窗口 | `range(1, i)` — 全范围搜索 | `range(max(1, i-4), i)` — 限制窗口为 4 |
| 计算复杂度 | $O(N^2)$ | $O(N \times 4) = O(N)$ |
| 阈值调整目标 | 最小化路标数 | 调整阈值使轨迹缩短至 ~2× |
| 熵加权 | ❌ | ✅ 低熵区域容忍更大误差，高熵区域更严格 |

---

### 基线方法三：恒定 2×（Constant 2×）

#### 方法描述

最简单的均匀下采样策略——对演示轨迹中的动作序列以**固定步长 2** 进行下采样，即每隔一帧取一帧，将轨迹长度缩短为原始的 $\frac{1}{2}$。

#### 公式

给定原始动作序列 $\{a_0, a_1, a_2, ..., a_{N-1}\}$：

$$
\text{downsampled} = \{a_0, a_2, a_4, ..., a_{2\lfloor(N-1)/2\rfloor}\}
$$

等价于 Python 中的 `action[::2]`。

**下采样后轨迹长度：**

$$
N' = \lceil N / 2 \rceil
$$

#### 代码位置

在 `robobase/robobase/replay_buffer/uniform_replay_buffer.py` 第 840 行，可以看到被注释掉的恒定 2× 下采样实现：

```python
# constant
# action_seq = episode[ACTION][action_start_idx:][::2][:(action_end_idx-action_start_idx)]
```

这行代码展示了恒定 2× 的实现方式：
- `episode[ACTION][action_start_idx:]`：从起始索引截取动作序列
- `[::2]`：每隔一帧取样（步长为 2）
- `[:(action_end_idx-action_start_idx)]`：截取到目标长度

#### 伪代码

```
输入: 动作序列 action[0..N-1]
输出: 下采样后的动作序列

stride = 2
indices = [0, 2, 4, 6, ..., 2*⌊(N-1)/2⌋]
downsampled_action = action[indices]

return downsampled_action
```

#### 对于 Aloha 的实现方式

在 Aloha 环境中，恒定 2× 下采样通过 `--constant_waypoint 2` 参数指定：

**命令行参数定义**（`aloha/act/imitate_episodes.py`，第 870-875 行）：
```python
parser.add_argument(
    "--constant_waypoint",
    action="store",
    type=int,
    help="constant_waypoint",
    required=False,
)
```

**参数传递**（第 47 行和第 154 行）：
```python
constant_waypoint = args["constant_waypoint"]
# ...
train_dataloader, val_dataloader, stats, _ = load_data(
    dataset_dir, num_episodes, camera_names,
    batch_size_train, batch_size_val,
    speedup, constant_waypoint, policy_class,
)
```

**数据集中的使用**（`aloha/act/act_utils.py`，第 224、234 行）：
```python
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, ..., constant_waypoint=None, ...):
        self.constant_waypoint = constant_waypoint
```

#### 特点

- **无需任何先验信息**：不需要熵、接触信息或策略
- **确定性**：结果完全可预测
- **均匀丢失信息**：在所有区域（无论复杂度高低）均匀丢失相同比例的帧
- **可能丢失关键帧**：精确操作时刻（如抓取、放置）可能恰好被跳过

---

### 基线方法四：恒定 3×（Constant 3×）

#### 方法描述

与恒定 2× 类似，但以**固定步长 3** 进行下采样，即每 3 帧取 1 帧，将轨迹长度缩短为原始的 $\frac{1}{3}$。

#### 公式

给定原始动作序列 $\{a_0, a_1, a_2, ..., a_{N-1}\}$：

$$
\text{downsampled} = \{a_0, a_3, a_6, ..., a_{3\lfloor(N-1)/3\rfloor}\}
$$

等价于 Python 中的 `action[::3]`。

**下采样后轨迹长度：**

$$
N' = \lceil N / 3 \rceil
$$

#### 伪代码

```
输入: 动作序列 action[0..N-1]
输出: 下采样后的动作序列

stride = 3
indices = [0, 3, 6, 9, ..., 3*⌊(N-1)/3⌋]
downsampled_action = action[indices]

return downsampled_action
```

#### 代码实现

类似恒定 2× 的实现方式，修改步长为 3：

```python
# 恒定 3× 下采样
action_seq = episode[ACTION][action_start_idx:][::3][:(action_end_idx-action_start_idx)]
```

或通过 `--constant_waypoint 3` 参数指定。

#### 特点

- 下采样更激进，信息损失更大
- 轨迹长度缩短至原始的 1/3
- **更容易丢失关键帧**：跳过的帧更多，精确操作时刻被跳过的概率更高
- 用于对比验证均匀下采样的极限

---

### Section 7.1 四种方法的完整对比

| 特性 | Contact Oracle | AWE* | 恒定 2× | 恒定 3× |
|------|---------------|------|---------|---------|
| **下采样方式** | 接触事件驱动分区 | DP + 熵加权阈值 | 均匀步长 2 | 均匀步长 3 |
| **需要 Oracle/先验** | ✅ 3D 接触信息 | ❌ | ❌ | ❌ |
| **需要策略熵** | ❌ | ✅（用于加权） | ❌ | ❌ |
| **自适应性** | 按接触事件自适应 | 按熵逐步自适应 | 无（均匀） | 无（均匀） |
| **目标加速比** | ~2× | ~2× | 2× | 3× |
| **信息保留策略** | 保留接触变化时刻 | 保留高误差/高熵区域 | 均匀丢失 | 均匀丢失 |
| **计算复杂度** | O(N)（在线检测） | O(N)（DP 搜索窗口=4） | O(1) | O(1) |
| **真实世界适用性** | ❌（需 3D 信息） | ✅ | ✅ | ✅ |

#### 核心公式对比

| 方法 | 下采样索引计算 |
|------|--------------|
| **Contact Oracle** | $\text{indices} = \begin{cases} \text{dense sampling} & \text{if } \|\text{contacts}_t - \text{contacts}_{t-1}\| > 0 \text{ (within Δt)} \\ \text{sparse sampling} & \text{otherwise} \end{cases}$ |
| **AWE\*** | $\text{indices} = \text{DP}(\min\|W\| \text{ s.t. } \forall t: d_t \leq \epsilon \times H_t \times 0.4)$ |
| **恒定 2×** | $\text{indices} = \{0, 2, 4, ..., 2\lfloor(N-1)/2\rfloor\}$ |
| **恒定 3×** | $\text{indices} = \{0, 3, 6, ..., 3\lfloor(N-1)/3\rfloor\}$ |

---

### DemoSpeedup 相对于四种基线的优势

1. **相对于 Contact Oracle**：不需要 Oracle 信息和 3D 先验，可直接从 2D 相机输入中通过策略熵估计获取精度标签
2. **相对于 AWE\***：DemoSpeedup 使用相同的熵加权 DP 框架，但结合了运行时自适应下采样（`process_action_label`），在训练阶段动态调整跳帧策略
3. **相对于恒定 2×/3×**：DemoSpeedup 根据轨迹各段的复杂度自适应调整下采样率，在简单运动区域跳更多帧（4步），在精确操作区域保留更多帧（2步），避免均匀下采样导致的关键帧丢失
