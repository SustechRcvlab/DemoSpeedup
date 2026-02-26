# DemoSpeedup 对比方法分析文档

## 目录

1. [问题一：DemoSpeedup 与其他下采样方法对比时是否控制了总轨迹长度一致](#问题一demospeedup-与其他下采样方法对比时是否控制了总轨迹长度一致)
2. [问题二：对比的下采样方法详细介绍](#问题二对比的下采样方法详细介绍)

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
