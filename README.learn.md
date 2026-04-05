# 完全读懂这个项目：从零理解 Brandes 介数中心性 MPI+CUDA 实现

> 本文档是这个项目的**学习指南**，目标是让你在读完之后，能够独立理解每一行代码的用途。  
> 不需要提前掌握 CUDA 或 MPI，只需要懂基础 C++ 和图论概念即可。

---

## 目录

1. [这个项目在解决什么问题？](#1-这个项目在解决什么问题)
2. [什么是介数中心性？](#2-什么是介数中心性)
3. [暴力算法为什么太慢？](#3-暴力算法为什么太慢)
4. [Brandes 算法的思路](#4-brandes-算法的思路)
5. [图怎么存储在内存里？——CSR 格式](#5-图怎么存储在内存里csr-格式)
6. [并行化的三层结构：MPI / CUDA Block / CUDA Thread](#6-并行化的三层结构mpi--cuda-block--cuda-thread)
7. [逐文件读代码](#7-逐文件读代码)
   - [graph.h](#71-graphh)
   - [brandes_gpu.cu — BFS 核函数](#72-brandes_gpucu--bfs-核函数)
   - [brandes_gpu.cu — 反向传播核函数](#73-brandes_gpucu--反向传播核函数)
   - [brandes_gpu.cu — computeBC 接口函数](#74-brandes_gpucu--computebc-接口函数)
   - [main.cpp](#75-maincpp)
8. [数据在整个系统中如何流动？](#8-数据在整个系统中如何流动)
9. [几个容易误解的细节](#9-几个容易误解的细节)
10. [建议的学习路径](#10-建议的学习路径)

---

## 1. 这个项目在解决什么问题？

想象你是一个城市规划师，手里有一张地图——城市里有很多路口（节点）和道路（边）。  
你想知道：**哪个路口是最关键的？** 也就是说，如果封闭哪个路口，会导致最多的人绕路。

这个"关键程度"在图论里就叫做**介数中心性（Betweenness Centrality，BC）**。

本项目实现了一个**高性能版本**的介数中心性计算：
- 用 **MPI** 把任务分配到多台服务器
- 在每台服务器上用 **GPU（CUDA）** 做大量并行计算
- 最后汇总结果

目标硬件是 IBM Polus 集群，每个节点有 2 块 NVIDIA Tesla P100 GPU。

---

## 2. 什么是介数中心性？

### 直觉理解

下面这张图里，节点 **C** 是不是最重要的？

```
A --- B --- C --- D --- E
                 |
                 F --- G
```

从 A 到 D、A 到 E、A 到 F、A 到 G，所有路径都**必须经过 C**。  
所以 C 的介数中心性最高——它是整张图的"咽喉要道"。

### 数学定义

对于节点 v，它的介数中心性定义为：

```
BC(v) = Σ  σ(s, t | v)
       s≠v≠t  ─────────
               σ(s, t)
```

逐项解释：
- **σ(s, t)**：从节点 s 到节点 t 一共有多少条**最短路径**
- **σ(s, t | v)**：其中经过节点 v 的最短路径有多少条
- **对所有 s≠v, t≠v 的节点对求和**

换句话说：对每一对出发点和目的地，算出"走最短路时有多大概率会经过 v"，把所有节点对的这个概率加起来，就是 v 的介数中心性。

### 一个小例子

图：`0 - 1 - 2`（线形，三个节点）

- 节点对 (0,2)：唯一最短路是 0→1→2，经过节点 1。σ(0,2|1)/σ(0,2) = 1/1 = 1
- 节点对 (0,1)：路径 0→1，不经过其他节点
- 节点对 (1,2)：路径 1→2，不经过其他节点

所以 BC(1) = 1，BC(0) = BC(2) = 0。节点 1 是中间桥接点，BC 最高。✓

---

## 3. 暴力算法为什么太慢？

最直接的做法：

```
对每一对节点 (s, t)：
    找出所有最短路径
    统计哪些节点在路径上
```

如果图有 n 个节点，节点对有 O(n²) 个。  
对每对节点找最短路径用 BFS 需要 O(n + m) 时间。  
总复杂度：**O(n² × (n + m))**

对于有 100 万节点的图（社交网络、交通网络很常见）：  
100万² × ... = 根本算不完。

---

## 4. Brandes 算法的思路

Brandes（2001年）发现了一个关键洞察：可以**反过来思考**。

不要对每对(s,t)计算"v 在不在路上"，而是**固定源节点 s**，一次性算出 s 对**所有**其他节点的 BC 贡献。

### 两个阶段

**阶段一：正向 BFS**（以 s 为源点做广度优先搜索）

记录两样东西：
- `dist[v]`：s 到 v 的最短路径长度（BFS 层次）
- `sigma[v]`：s 到 v 的最短路径**数量**

`sigma` 的计算规则很简单：
```
sigma[s] = 1（源点到自身只有1条"路"）
sigma[v] = Σ sigma[u]   （对所有满足 dist[u] = dist[v]-1 的邻居 u 求和）
           u是v的前驱
```

意思是：到达 v 的最短路径数 = 所有能"一步走到 v"的上层节点的路径数之和。

**阶段二：反向传播**（从 BFS 最深层倒推回源点）

定义"依赖值" `delta[v]`：源点 s 通过 v 能"帮助"其他节点的程度。

```
delta[v] = Σ  sigma[v]  × (1 + delta[w])
          w是v的子节点  ──────────
                         sigma[w]
```

从最深层开始，逐层往上累积，最后：

```
BC[v] += delta[v]    （对所有源节点 s 累加）
```

### 复杂度

对**每个**源节点做一次 BFS：O(n + m)  
总共 n 个源节点：**O(n × (n + m))**

比暴力快了 O(n) 倍，而且非常适合并行——不同源节点的计算**完全独立**！

---

## 5. 图怎么存储在内存里？——CSR 格式

### 为什么不用邻接矩阵？

100万节点的邻接矩阵需要 100万 × 100万 × 4 字节 = **4TB**，完全放不下。

### CSR（压缩稀疏行）格式

只存真正有边的地方，用两个数组表示：

```
edges:  0-1, 1-2, 2-3

offset = [0, 1, 3, 4]   ← 每个节点的出边从哪里开始
dest   = [1, 0, 2, 1, 3, 2]  ← 所有出边的目标（无向图双向存）
          ↑  ↑──↑  ↑──↑  ↑
         节点0 节点1 节点2 节点3
         的边  的边  的边  的边
```

**访问节点 v 的所有邻居**：遍历 `dest[offset[v] .. offset[v+1]-1]`

这是 GPU 计算图算法的标准格式，因为访问模式规则，有利于内存合并（Coalesced Memory Access）。

---

## 6. 并行化的三层结构：MPI / CUDA Block / CUDA Thread

本项目用了**三层并行**，从大到小理解：

```
┌─────────────────────────────────────────────────┐
│  MPI 层（跨节点）                                │
│  进程 0：负责源节点 0 ~ n/2                      │
│  进程 1：负责源节点 n/2 ~ n-1                    │
│  各进程计算完后 MPI_Reduce 把 BC 值加在一起       │
├─────────────────────────────────────────────────┤
│  CUDA Block 层（同一 GPU 内）                    │
│  Block 0：负责源节点 0 的 BFS                   │
│  Block 1：负责源节点 1 的 BFS                   │
│  Block 2：负责源节点 2 的 BFS   ← 同时运行！    │
│  ...                                            │
├─────────────────────────────────────────────────┤
│  CUDA Thread 层（同一 Block 内）                 │
│  Thread 0, 1, ..., 255：                        │
│  并行处理当前 BFS 层的所有出边                   │
└─────────────────────────────────────────────────┘
```

**关键思想**：不同源节点的计算完全独立 → 一个 block 专门处理一个源节点，多个 block 同时跑。

---

## 7. 逐文件读代码

### 7.1 `graph.h`

这是最简单的文件，只做一件事：**定义图的存储结构并提供构建函数**。

```cpp
struct CSRGraph {
    int  n;       // 节点总数
    int  m;       // 有向边总数（无向图双向存，所以 m = 2×无向边数）
    int* offset;  // 长度 n+1，offset[v] 到 offset[v+1]-1 是 v 的出边范围
    int* dest;    // 长度 m，所有出边的目标节点
};
```

**`buildCSR` 函数做了什么？**

分三步：

```
第1步：统计每个节点的出度（有几条出边）
       edges = [(0,1), (1,2)]
       deg = [1, 2, 1]   ← 无向图，0有1条，1有2条，2有1条

第2步：用出度做前缀和，得到 offset 数组
       offset = [0, 1, 3, 4]
                 ↑     ↑
                 0号从0开始，1号从1开始，2号从3开始，结尾是4

第3步：填充 dest 数组（正向+反向各一条）
       dest = [1,  0,2,  1]
               ↑   ↑↑   ↑
              节点0 节点1 节点2
```

---

### 7.2 `brandes_gpu.cu` — BFS 核函数

这是整个项目最核心的部分，要慢慢读。

#### 函数签名

```cuda
__global__ void bfs_kernel(
    int n, const int* offset, const int* dest,  // CSR 图（只读）
    const int* sources,  // 本批次源节点列表
    int*   dist,         // 输出：BFS 距离
    float* sigma,        // 输出：最短路径数
    int*   stack,        // 输出：BFS 访问顺序（给反向传播用）
    int*   s_count       // 输出：实际访问了几个节点
)
```

`__global__` 表示这是在 GPU 上运行的核函数，由 CPU 调用，GPU 执行。

#### 每个 block 独立处理一个源节点

```cuda
int bid = blockIdx.x;   // 这个 block 是第几个（对应第几个源节点）
int src = sources[bid]; // 取出对应的源节点编号
int tid = threadIdx.x;  // 我是这个 block 里的第几号线程（0~255）
int bdim = blockDim.x;  // 这个 block 里总共有多少线程（256）
```

每个 block 有自己专属的数据区，通过偏移量隔开：

```cuda
int* my_dist  = dist  + (long long)bid * n;
//              ↑ 指针偏移：第 bid 个 block 的数据从 bid*n 处开始
```

为什么用 `long long`？防止 `bid * n` 整数溢出（大图时 bid 和 n 都很大）。

#### 初始化

```cuda
for (int v = tid; v < n; v += bdim) {
    my_dist[v]  = -1;    // -1 表示"未访问"
    my_sigma[v] = 0.0f;
}
```

256 个线程分工初始化 n 个节点：
- 线程 0 负责节点 0, 256, 512, ...
- 线程 1 负责节点 1, 257, 513, ...
- 以此类推

这叫**步长循环（stride loop）**，是 GPU 编程的标准惯用法。

```cuda
__syncthreads();
// ↑ 屏障：等所有线程都完成初始化，再继续
```

`__syncthreads()` 是 block 内的路障，**所有线程必须都到达这里，才能一起继续**。这是 GPU 编程中最重要的同步原语。

然后只让线程 0 初始化源节点（只需要一个线程做这件事）：

```cuda
if (tid == 0) {
    my_dist[src]  = 0;      // 源到自身距离为 0
    my_sigma[src] = 1.0f;   // 到自身只有 1 条路
    my_stack[0]   = src;    // 源节点入队
}
```

#### 共享内存维护队列边界

```cuda
__shared__ int s_front, s_back;
//  ↑ __shared__ 表示共享内存——block 内所有线程共享，比全局内存快 100 倍
```

`s_front` 和 `s_back` 就像一个队列的头和尾指针：
- `[s_front, s_back)` 是**当前 BFS 层**的节点范围（在 stack 数组里）
- 新发现的节点追加到 `s_back` 之后

#### BFS 主循环

```
初始状态：front=0, back=1, stack=[src]
           当前层：[src]

第1次循环：
  处理 stack[0..0] = [src]
  找 src 的所有邻居 w：
    - 若 w 未访问（dist[w]==-1）：设 dist[w]=1，追加到 stack
    - 若 dist[w]==1：sigma[w] += sigma[src]
  结束后 front=1，back=（发现的邻居数+1）
  当前层变为第 1 层的节点

第2次循环：
  处理第 1 层所有节点...
  以此类推直到队列为空
```

关键的原子操作：

```cuda
if (atomicCAS(&my_dist[w], -1, dv + 1) == -1) {
```

`atomicCAS(地址, 期望旧值, 新值)`：
- 如果 `*地址 == -1`（未访问），则把它改为 `dv+1`，并**返回旧值 -1**
- 如果 `*地址 != -1`（已访问），不修改，**返回实际值**

返回值 == -1 就意味着"是我率先访问了这个节点"。这保证了多个线程同时发现节点 w 时，只有一个线程把它加入队列。

```cuda
if (my_dist[w] == dv + 1) {
    atomicAdd(&my_sigma[w], my_sigma[v]);
}
```

这行在"w 在下一层"时执行，把 v 的路径数累加到 w。  
两个线程都可以通过 `atomicCAS`（一个成功，一个失败但仍然看到 `dist[w]==dv+1`），都会给 sigma[w] 累加自己的 sigma，结果是正确的。

---

### 7.3 `brandes_gpu.cu` — 反向传播核函数

BFS 完成后，`stack` 里按 BFS 顺序记录了所有访问的节点。  
反向传播就是**倒着走这个 stack**。

#### 为什么要倒着走？

因为 delta 的计算有依赖关系：

```
delta[v] 依赖 delta[w]，其中 w 是 v 的子节点（更深层的节点）
```

所以必须**先算深层的**，再算浅层的，即从 stack 末尾往前处理。

#### 核心计算

```cuda
for (int i = cnt - 1; i >= 1; i--) {
    int w = my_stack[i];
    float coef = (1.0f + my_delta[w]) / my_sigma[w];
    //            ↑ 1 加上 w 自己的依赖值
    //                           ↑ 除以到达 w 的路径数

    for (int e = offset[w] + tid; e < offset[w+1]; e += bdim) {
        int v = dest[e];
        if (my_dist[v] == my_dist[w] - 1)  // v 是 w 的前驱（上一层）
            atomicAdd(&my_delta[v], my_sigma[v] * coef);
    }
    __syncthreads();  // ← 必须等本轮所有更新完成，再处理下一个节点
}
```

用图示理解 `coef`：

```
从 s 出发，经过 v 到达 w 的路径数 = sigma[v]
从 s 出发，到达 w 的全部路径数   = sigma[w]

v 对于路过 w 的"贡献比例" = sigma[v] / sigma[w]

w 及其子树中的节点，都能被 v 通过 w 来"代理"
这部分额外贡献 = sigma[v]/sigma[w] × (1 + delta[w])
```

`(1 + delta[w])` 里的 1 代表 w 本身（s→...→v→w 这条路），  
`delta[w]` 代表 w 的子孙节点（s→...→v→w→...→t）。

#### 最后写入全局 BC

```cuda
for (int v = tid; v < n; v += bdim) {
    if (v != src)
        atomicAdd(&bc[v], (double)my_delta[v]);
}
```

把这个源节点 s 对所有其他节点 v 的贡献，累加到全局 BC 数组。  
对所有源节点都处理完之后，BC[v] 就是最终结果。

---

### 7.4 `brandes_gpu.cu` — `computeBC` 接口函数

这是 CPU 端调用的函数，负责：

**① 上传图数据到 GPU**

```cpp
cudaMalloc(&d_offset, (n+1) * sizeof(int));   // 在 GPU 显存里分配空间
cudaMemcpy(d_offset, g->offset, ..., cudaMemcpyHostToDevice);  // 从 CPU 内存拷到 GPU
```

`d_` 前缀是约定俗成的命名，表示"device（GPU）端"的变量。

**② 批量处理源节点**

```cpp
int batch = (n_src < BATCH_SIZE) ? n_src : BATCH_SIZE;
```

如果源节点很多（比如 10万个），一次性为全部源节点分配工作内存会耗尽显存。  
所以分批处理，每批最多 256 个源节点，处理完一批再处理下一批。

```cpp
// 工作数组大小 = batch × n
// 每个源节点占用 n 个槽位，batch 个源节点并行，总共需要 batch×n
cudaMalloc(&d_dist, (long long)batch * n * sizeof(int));
```

**③ 启动 CUDA 核函数**

```cpp
bfs_kernel<<<cur, BLOCK_SIZE>>>(...);
//           ↑     ↑
//    启动 cur 个 block，每个 block 256 个线程
```

`<<<grid, block>>>` 是 CUDA 特有的语法，称为**执行配置**：
- 第一个参数：启动多少个 block（这里等于本批次源节点数）
- 第二个参数：每个 block 多少个线程

**④ 等待 GPU 完成并取回结果**

```cpp
cudaDeviceSynchronize();  // CPU 等待 GPU 上所有核函数执行完毕
cudaMemcpy(h_tmp, d_bc, ..., cudaMemcpyDeviceToHost);  // 把结果拷回 CPU
```

---

### 7.5 `main.cpp`

主程序是 MPI 程序的骨架，按顺序做 7 件事：

#### Step 1：rank 0 读图，广播给所有进程

```
              磁盘
               ↓ fopen/fgets
           rank 0 读入边列表
               ↓ MPI_Bcast
    ┌──────────┴────────────┐
  rank 0                 rank 1
  拿到完整图             拿到完整图
```

为什么每个进程都需要完整的图？  
因为每个进程的 GPU 要独立做 BFS，BFS 需要访问任意节点的邻居，所以必须有完整图。

```cpp
MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//              ↑          ↑  ↑
//         发送1个整数   从rank0广播  广播给所有人
```

#### Step 2：均分源节点

```cpp
int per_proc  = n / nprocs;     // 每个进程基础分配量
int remainder = n % nprocs;     // 除不尽时多出来的
int my_start  = rank * per_proc + (rank < remainder ? rank : remainder);
int my_count  = per_proc + (rank < remainder ? 1 : 0);
```

举例：n=15，nprocs=2：
- rank 0：my_start=0，my_count=8（处理节点 0~7）
- rank 1：my_start=8，my_count=7（处理节点 8~14）

#### Step 3：绑定 GPU

```cpp
cudaSetDevice(rank % 2);
// rank 0 → GPU 0，rank 1 → GPU 1
// 如果有更多进程，% 2 保证在 2 块 GPU 间轮流分配
```

#### Step 4~5：计算 + 汇总

```
rank 0 的 GPU 计算 local_bc[0..n-1]（只包含节点 0~7 作为源的贡献）
rank 1 的 GPU 计算 local_bc[0..n-1]（只包含节点 8~14 作为源的贡献）
            ↓ MPI_Reduce(MPI_SUM)
rank 0 拿到 global_bc = local_bc(rank0) + local_bc(rank1)
```

`MPI_Reduce` 把所有进程的数组按元素相加，结果发给 rank 0。

#### Step 6：除以 2，打印结果

```cpp
printf("%d %.6f\n", v, global_bc[v] / 2.0);
```

**为什么除以 2？**  
Brandes 算法对每个节点 s 做一次 BFS，计算 s 对其他所有节点的贡献。  
对无向图来说，路径 s→...→t 和 t→...→s 是同一条路，但被算了两次（s 作为源一次，t 作为源一次）。  
所以最终结果需要除以 2，才和标准定义一致（NetworkX 也是这样处理的）。

---

## 8. 数据在整个系统中如何流动？

```
磁盘（test.txt）
    │
    │ fopen / fgets（只有 rank 0）
    ▼
CPU 内存（rank 0）
  edges = [(0,1),(1,2),...]
    │
    │ MPI_Bcast（广播给所有进程）
    ▼
CPU 内存（所有进程）
  CSRGraph g（完整图）
    │
    │ cudaMemcpy（HostToDevice）
    ▼
GPU 显存
  d_offset[], d_dest[]（图结构）
  d_dist[]  [batch×n]（BFS 距离）
  d_sigma[] [batch×n]（最短路径数）
  d_stack[] [batch×n]（BFS 访问顺序）
  d_delta[] [batch×n]（依赖值）
  d_bc[]    [n]      （BC 累加结果）
    │
    │ bfs_kernel（GPU 并行）
    │ back_prop_kernel（GPU 并行）
    │
    │ cudaMemcpy（DeviceToHost）
    ▼
CPU 内存
  local_bc[n]（本进程计算的部分）
    │
    │ MPI_Reduce（跨进程求和）
    ▼
CPU 内存（rank 0）
  global_bc[n]（最终结果）
    │
    │ printf
    ▼
屏幕输出 / out.txt
```

---

## 9. 几个容易误解的细节

### Q1：sigma 用 float 而不是 double，精度够吗？

对于节点数较少的图（如测试图 15 节点），float（~7位有效数字）完全够用。  
对于超大图，最短路径数可能达到 10^30 以上，float 会溢出。如需处理大图，sigma 改 double 即可。

### Q2：`atomicCAS` 之后为什么还要检查 `dist[w] == dv+1`？

```cuda
if (atomicCAS(&my_dist[w], -1, dv+1) == -1) {
    // 只有"我"首先访问了 w，才执行这里
    int pos = atomicAdd(&s_back, 1);
    my_stack[pos] = w;
}
if (my_dist[w] == dv+1) {
    // 所有 w 的前驱都在这里累积 sigma
    atomicAdd(&my_sigma[w], my_sigma[v]);
}
```

这两个 if 是分开的。第一个只有一个线程进入（入队操作）。  
第二个所有"w 的前驱"都会进入（sigma 累积操作），包括那个 atomicCAS 失败的线程。  
这样才能把所有前驱的路径数都累加进来。

### Q3：反向传播的 `__syncthreads()` 为什么在循环里？

```cuda
for (int i = cnt-1; i >= 1; i--) {
    // 处理节点 stack[i]，更新其前驱的 delta
    ...
    __syncthreads();  // ← 必须在这里等待
}
```

假设 stack[5] 的前驱是 stack[3]。  
处理 stack[5] 时会修改 delta[stack[3]]。  
必须等 stack[5] 的所有修改完成，才能处理 stack[3]（因为处理 stack[3] 时要读 delta[stack[3]]）。  
`__syncthreads()` 就是这个等待点。

### Q4：无向图为什么需要双向存储边？

BFS 是"沿着边走"，对无向图 0-1，从 0 出发能走到 1，从 1 出发也能走到 0。  
CSR 只存"出边"，所以无向边 (0,1) 必须同时存 0→1 和 1→0 两条有向边。

---

## 10. 建议的学习路径

按以下顺序读代码，循序渐进：

```
第 1 步：读 graph.h
         理解 CSRGraph 结构体和 buildCSR 函数
         动手在纸上画 offset 和 dest 数组

第 2 步：读 verify_stdlib.py
         这是纯 Python 实现的 Brandes 算法，没有任何并行化
         逻辑最清晰，先彻底理解算法本身
         重点看 BFS 部分和 while stack: 的反向传播部分

第 3 步：读 main.cpp（只读 MPI 部分）
         跳过 computeBC 调用，专注理解：
         - 图如何被读入和广播
         - 源节点如何被均分
         - MPI_Reduce 如何汇总

第 4 步：读 brandes_gpu.cu 的 computeBC 函数
         理解 CPU 端如何调用 GPU：
         cudaMalloc / cudaMemcpy / <<<>>> / cudaDeviceSynchronize

第 5 步：读 bfs_kernel
         对照 verify_stdlib.py 里的 BFS 代码，逐行找对应关系
         重点理解：__syncthreads() 在哪里、为什么在那里

第 6 步：读 back_prop_kernel
         对照 verify_stdlib.py 里的 while stack: 部分
         理解 coef 的含义和 __syncthreads() 的必要性
```

### 快速验证理解的方法

修改 `test.txt`，改成最简单的三节点图，手动算出期望结果，再运行程序验证：

```
# 三节点路径图：0-1-2
0 1
1 2
```

期望输出：
```
0 0.000000   ← 端点，BC=0
1 1.000000   ← 中间节点，是所有(0,2)路径的必经之路
2 0.000000   ← 端点，BC=0
```

如果输出一致，说明你的理解是正确的。

---

*本项目实测结果：在 IBM Polus 集群（2块 P100 GPU）上，15节点图计算时间 0.0591 秒，通信时间 0.0002 秒，所有节点 BC 值误差 < 1e-6。*
