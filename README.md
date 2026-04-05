# Brandes 介数中心性 · MPI + CUDA 并行实现

> [Читать на русском](README.ru.md)

基于 **Brandes 算法**的分布式 GPU 实现，面向 IBM Polus 超算集群（ppc64le 架构）。  
利用 **MPI** 跨节点分发源节点任务，利用 **CUDA** 在单节点内并行执行 BFS 与反向传播。

---

## 运行环境

| 项目 | 规格 |
|------|------|
| 集群 | IBM Polus (ppc64le) |
| GPU | NVIDIA Tesla P100 16GB × 2（NVLink 互联） |
| CUDA | 10.2，架构 sm\_60 |
| MPI | `mpi/openmpi3-ppc64le` |
| 编译器 | `nvcc` + `mpicxx` |

---

## 文件说明

```
.
├── graph.h            # CSR 格式图结构体 + 构建函数
├── brandes_gpu.cu     # CUDA 核函数（bfs_kernel、back_prop_kernel）+ 接口
├── main.cpp           # MPI 主程序（读图、分发、汇总、计时）
├── Makefile           # 编译脚本
├── test.txt           # 15 节点测试图（边列表格式）
├── verify.py          # 验证脚本（依赖 networkx）
└── verify_stdlib.py   # 验证脚本（纯标准库，无需安装依赖）
```

---

## 快速开始

```bash
# 1. 加载 MPI 模块
module load mpi/openmpi3-ppc64le

# 2. 编译
make

# 3. 运行（2 个 MPI 进程，各绑定 1 块 P100）
mpirun -np 2 ./brandes test.txt > out.txt

# 4. 验证结果（二选一）
python3 verify.py test.txt out.txt          # 需要 networkx
python3 verify_stdlib.py test.txt out.txt   # 纯标准库，无需安装

# 或者管道方式一步完成
mpirun -np 2 ./brandes test.txt | python3 verify_stdlib.py test.txt
```

---

## 输入格式

图文件为边列表，每行一条无向边，支持 `#` 开头的注释：

```
# 注释行
0 1
1 2
2 3
```

节点编号从 `0` 开始，程序自动推断节点总数。

---

## 输出格式

标准输出（stdout）每行打印一个节点的介数中心性值：

```
0 0.000000
1 3.500000
2 5.000000
...
```

计时信息打印到标准错误（stderr），不干扰结果管道：

```
计算时间（GPU BFS+反向传播）: 0.0591 秒
通信时间（MPI_Reduce）:        0.0002 秒
```

---

## 算法原理

### 整体流程

```
rank 0 读图 → MPI_Bcast 广播 → 各进程均分源节点
    ↓
每个进程调用 computeBC()
    ↓
  [GPU] bfs_kernel      — 正向 BFS：计算 dist[], sigma[], stack[]
  [GPU] back_prop_kernel — 反向传播：计算 delta[]，累加到 bc[]
    ↓
MPI_Reduce 汇总 → rank 0 打印结果
```

### CUDA 并行策略

- **一个 block = 一个源节点**，block 内线程并行处理出边
- BFS 层间用 `__syncthreads()` 隔离，保证 sigma 读取时已稳定
- 反向传播不存前驱列表，改为扫边重建（`dist[v] == dist[w]-1`），节省显存
- `atomicCAS` 防止节点重复入队，`atomicAdd` 无锁累积路径数和依赖值

### BC 公式

$$BC(v) = \sum_{s \neq v} \sum_{t \neq s,v} \frac{\sigma(s,t \mid v)}{\sigma(s,t)}$$

无向图中每对 $(s,t)$ 被计算两次，最终结果除以 2。

---

## 编译选项

```bash
# 若 CUDA 安装路径不是默认的 /usr/local/cuda-10.2
make CUDA_HOME=/path/to/cuda

# 清理编译产物
make clean
```

---

## 测试结果

在 IBM Polus 集群上，使用 2 个 MPI 进程（各绑定 1 块 P100）运行 15 节点测试图：

```
计算时间（GPU BFS+反向传播）: 0.0591 秒
通信时间（MPI_Reduce）:        0.0002 秒
PASS - 所有 15 个节点 BC 值正确（误差 < 1e-06）
```

通信开销仅占计算时间的 **0.3%**，GPU 加速效果显著。

---

## 注意事项

- 不使用 NCCL，各 GPU 独立工作，仅通过 MPI 通信
- 代码不含任何 x86 intrinsics，完全兼容 ppc64le
- `BATCH_SIZE`（默认 256）控制每批处理的源节点数，可按显存调整
