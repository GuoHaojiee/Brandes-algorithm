/*
 * brandes_gpu.cu - CUDA 实现 Brandes 介数中心性算法
 * 硬件: NVIDIA Tesla P100 (sm_60), CUDA 10.2, ppc64le
 * 策略: 每个 CUDA block 负责一个源节点的完整 BFS + 反向传播
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "graph.h"

// 每个 block 的线程数（处理一个源节点）
#define BLOCK_SIZE 256
// 每批处理的源节点数（控制显存占用，可按需调整）
#define BATCH_SIZE 256

// CUDA 错误检查宏
#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA错误 %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

/*
 * 正向 BFS 核函数 —— 每个 block 处理一个源节点
 *
 * 逐层 BFS：利用共享内存维护队列边界 [s_front, s_back)。
 * 每层内线程并行处理出边，通过 atomicCAS 避免节点重复访问，
 * 通过 atomicAdd 累积最短路径数 sigma。
 *
 * 输入:
 *   n, offset, dest  - CSR 图
 *   sources[bid]     - 本 block 对应的源节点编号
 * 输出 (均按 bid*n 偏移存储，每个源节点独占 n 个槽位):
 *   dist[bid*n+v]    - v 到源的 BFS 距离，-1 表示不可达
 *   sigma[bid*n+v]   - 从源到 v 的最短路径数
 *   stack[bid*n+i]   - 第 i 个被 BFS 访问的节点（层序，用于反向传播）
 *   s_count[bid]     - 本源节点 BFS 实际访问的节点数
 */
__global__ void bfs_kernel(
    int n,
    const int* __restrict__ offset,
    const int* __restrict__ dest,
    const int* __restrict__ sources,
    int*   dist,
    float* sigma,
    int*   stack,
    int*   s_count
) {
    int bid = blockIdx.x;   // 本 block 在当前批次中的索引
    int src = sources[bid]; // 源节点编号
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // 指向本源节点专属数据区（每个源节点占用 n 个连续槽位）
    int*   my_dist  = dist  + (long long)bid * n;
    float* my_sigma = sigma + (long long)bid * n;
    int*   my_stack = stack + (long long)bid * n;

    // 初始化：所有节点未访问，路径数为 0
    for (int v = tid; v < n; v += bdim) {
        my_dist[v]  = -1;
        my_sigma[v] = 0.0f;
    }
    __syncthreads();

    // 初始化源节点：距离 0，路径数 1，入队
    if (tid == 0) {
        my_dist[src]  = 0;
        my_sigma[src] = 1.0f;
        my_stack[0]   = src;
    }

    // 共享内存维护 BFS 队列边界（当前层 [front, back)，新节点追加到 back）
    __shared__ int s_front, s_back;
    if (tid == 0) { s_front = 0; s_back = 1; }
    __syncthreads();

    // BFS 主循环：逐层扩展直到队列为空
    while (true) {
        int front = s_front, back = s_back;
        if (front >= back) break; // 队列为空，BFS 结束

        // 线程并行处理当前层所有节点的出边
        for (int qi = front + tid; qi < back; qi += bdim) {
            int v  = my_stack[qi];
            int dv = my_dist[v];
            for (int e = offset[v]; e < offset[v + 1]; e++) {
                int w = dest[e];
                // 若 w 未访问，原子设置距离并追加到队列
                if (atomicCAS(&my_dist[w], -1, dv + 1) == -1) {
                    int pos = atomicAdd(&s_back, 1);
                    my_stack[pos] = w;
                }
                // 若 w 恰好在下一层，则 v 是 w 的前驱，累积最短路径数
                // 注：sigma[v] 在上一轮 __syncthreads() 后已稳定，可安全读取
                if (my_dist[w] == dv + 1) {
                    atomicAdd(&my_sigma[w], my_sigma[v]);
                }
            }
        }
        __syncthreads(); // 等待本层所有出边处理完毕，保证全局内存可见

        // 将队列前指针推进到下一层的起始位置
        if (tid == 0) s_front = back;
        __syncthreads();
    }

    // 记录本次 BFS 访问的节点总数（供反向传播使用）
    if (tid == 0) s_count[bid] = s_back;
}

/*
 * 反向累积核函数 —— 按 BFS 逆序计算节点依赖值 delta
 *
 * 对 stack 中每个节点 w（从最深层逆向到根），遍历 w 的邻居找前驱 v
 * （满足 dist[v] == dist[w] - 1），然后累积：
 *     delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
 * 最后将 delta 累加到全局 BC 数组（排除源节点自身）。
 *
 * 注: P100 (sm_60) 原生支持 double 类型的 atomicAdd。
 */
__global__ void back_prop_kernel(
    int n,
    const int* __restrict__ offset,
    const int* __restrict__ dest,
    const int* __restrict__ sources,
    const int*   __restrict__ dist,
    const float* __restrict__ sigma,
    float* delta,             // 工作数组: [batch*n]，本函数负责初始化
    const int* __restrict__ stack,
    const int* __restrict__ s_count,
    double* bc                // 全局 BC 累加数组: [n]（跨批次累加）
) {
    int bid = blockIdx.x;
    int src = sources[bid];
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int cnt = s_count[bid]; // 本源节点 BFS 访问的节点总数

    const int*   my_dist  = dist  + (long long)bid * n;
    const float* my_sigma = sigma + (long long)bid * n;
    float*       my_delta = delta + (long long)bid * n;
    const int*   my_stack = stack + (long long)bid * n;

    // 初始化依赖值全为 0
    for (int v = tid; v < n; v += bdim) my_delta[v] = 0.0f;
    __syncthreads();

    // 按 BFS 逆序处理（i=0 对应源节点，跳过；从 cnt-1 倒推到 1）
    // 每次迭代处理一个节点 w，线程并行处理 w 的所有出边
    for (int i = cnt - 1; i >= 1; i--) {
        int w = my_stack[i];
        // w 处理时，所有比 w 更深的节点的 delta 已稳定（由之前的 __syncthreads 保证）
        float coef = (1.0f + my_delta[w]) / my_sigma[w];

        // 并行扫描 w 的邻居，找前驱节点 v（dist[v] == dist[w]-1）
        for (int e = offset[w] + tid; e < offset[w + 1]; e += bdim) {
            int v = dest[e];
            if (my_dist[v] == my_dist[w] - 1)
                atomicAdd(&my_delta[v], my_sigma[v] * coef);
        }
        // 确保本轮所有 delta 更新完成，再处理 w 的前驱节点（下一次迭代）
        __syncthreads();
    }

    // 将本源节点的贡献累加到全局 BC（P100 sm_60 支持 double atomicAdd）
    for (int v = tid; v < n; v += bdim) {
        if (v != src)
            atomicAdd(&bc[v], (double)my_delta[v]);
    }
}

/*
 * 主计算接口：对指定源节点列表计算 BC 贡献，累加到 bc 数组
 *
 * 参数:
 *   g       - 图的 CSR 结构（主机端）
 *   sources - 源节点编号数组（主机端）
 *   n_src   - 源节点数量
 *   bc      - 输入/输出: 各节点 BC 值（主机端），函数将结果累加到此数组
 */
void computeBC(const CSRGraph* g, const int* sources, int n_src, double* bc) {
    int n = g->n, m = g->m;

    // 上传图数据到 GPU
    int *d_offset, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_offset, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dest,   m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_offset, g->offset, (n+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dest,   g->dest,   m*sizeof(int),     cudaMemcpyHostToDevice));

    // GPU 端全局 BC 累加数组（全零初始化）
    double* d_bc;
    CUDA_CHECK(cudaMalloc(&d_bc, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_bc, 0, n * sizeof(double)));

    // 按批次分配工作数组，避免一次性占用过多显存
    int batch = (n_src < BATCH_SIZE) ? n_src : BATCH_SIZE;
    int    *d_sources, *d_dist, *d_stack, *d_s_count;
    float  *d_sigma, *d_delta;
    CUDA_CHECK(cudaMalloc(&d_sources, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist,    (long long)batch * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sigma,   (long long)batch * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta,   (long long)batch * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stack,   (long long)batch * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s_count, batch * sizeof(int)));

    // 分批执行：每批 batch 个源节点
    for (int start = 0; start < n_src; start += batch) {
        int cur = ((start + batch) <= n_src) ? batch : (n_src - start);

        // 上传本批次源节点列表
        CUDA_CHECK(cudaMemcpy(d_sources, sources + start,
                              cur * sizeof(int), cudaMemcpyHostToDevice));

        // 第一步：正向 BFS（每个 block 处理一个源节点）
        bfs_kernel<<<cur, BLOCK_SIZE>>>(
            n, d_offset, d_dest, d_sources,
            d_dist, d_sigma, d_stack, d_s_count);
        CUDA_CHECK(cudaGetLastError());

        // 第二步：反向累积依赖值并写入全局 BC
        back_prop_kernel<<<cur, BLOCK_SIZE>>>(
            n, d_offset, d_dest, d_sources,
            d_dist, d_sigma, d_delta, d_stack, d_s_count, d_bc);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将 GPU 端结果拷回并累加到主机 bc 数组
    double* h_tmp = new double[n];
    CUDA_CHECK(cudaMemcpy(h_tmp, d_bc, n * sizeof(double), cudaMemcpyDeviceToHost));
    for (int v = 0; v < n; v++) bc[v] += h_tmp[v];
    delete[] h_tmp;

    // 释放 GPU 内存
    cudaFree(d_offset);  cudaFree(d_dest);    cudaFree(d_bc);
    cudaFree(d_sources); cudaFree(d_dist);    cudaFree(d_sigma);
    cudaFree(d_delta);   cudaFree(d_stack);   cudaFree(d_s_count);
}
