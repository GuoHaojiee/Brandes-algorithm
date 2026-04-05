/*
 * main.cpp - MPI 主程序：分布式 Brandes 介数中心性计算
 * 运行方式: mpirun -np 2 ./brandes <图文件>
 * 图文件格式: 每行 "u v" 表示一条无向边（支持以 # 开头的注释行）
 */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "graph.h"

// 声明 GPU 计算函数（实现在 brandes_gpu.cu，编译为同一 C++ 命名空间）
void computeBC(const CSRGraph* g, const int* sources, int n_src, double* bc);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "用法: %s <图文件>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // ========== Step 1: rank0 读取图文件，广播给所有进程 ==========
    int n = 0, m = 0;
    std::vector<int> edge_buf; // 边列表扁平化存储: [u0,v0, u1,v1, ...]

    if (rank == 0) {
        FILE* fp = fopen(argv[1], "r");
        if (!fp) {
            fprintf(stderr, "错误: 无法打开文件 %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        char line[256];
        int max_id = -1;
        // 逐行读取，跳过注释行（以 # 开头）
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
            int u, v;
            if (sscanf(line, "%d %d", &u, &v) == 2) {
                edge_buf.push_back(u);
                edge_buf.push_back(v);
                if (u > max_id) max_id = u;
                if (v > max_id) max_id = v;
            }
        }
        fclose(fp);
        n = max_id + 1;     // 节点编号从 0 开始
        m = (int)edge_buf.size() / 2;
    }

    // 广播图规模（节点数 n，边数 m）
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 广播边列表（非 rank0 进程先分配空间）
    if (rank != 0) edge_buf.resize(m * 2);
    MPI_Bcast(edge_buf.data(), m * 2, MPI_INT, 0, MPI_COMM_WORLD);

    // 所有进程各自在本地构建完整 CSR 图（GPU 计算需要完整图结构）
    std::vector<std::pair<int,int>> edges(m);
    for (int i = 0; i < m; i++)
        edges[i] = {edge_buf[2*i], edge_buf[2*i+1]};
    CSRGraph g = buildCSR(n, edges);

    // ========== Step 2: 按 rank 均分源节点列表 ==========
    // 总源节点为 0..n-1，前 remainder 个进程各多分配一个
    int per_proc  = n / nprocs;
    int remainder = n % nprocs;
    int my_start  = rank * per_proc + (rank < remainder ? rank : remainder);
    int my_count  = per_proc + (rank < remainder ? 1 : 0);

    std::vector<int> my_sources(my_count);
    for (int i = 0; i < my_count; i++)
        my_sources[i] = my_start + i;

    // ========== Step 3: 绑定本节点 GPU（每节点 2 块 P100，按 rank%2） ==========
    cudaSetDevice(rank % 2);

    // ========== Step 4: CUDA 计算本进程负责的源节点贡献 ==========
    std::vector<double> local_bc(n, 0.0);

    double t_comp_start = MPI_Wtime();
    if (my_count > 0)
        computeBC(&g, my_sources.data(), my_count, local_bc.data());
    double compute_time = MPI_Wtime() - t_comp_start;

    // ========== Step 5: MPI_Reduce 汇总 BC 值到 rank0 ==========
    double t_comm_start = MPI_Wtime();
    std::vector<double> global_bc(n, 0.0);
    MPI_Reduce(local_bc.data(), global_bc.data(),
               n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double comm_time = MPI_Wtime() - t_comm_start;

    // ========== Step 6: rank0 打印结果到 stdout ==========
    if (rank == 0) {
        for (int v = 0; v < n; v++) {
            // 无向图中每对 (s,t) 被计算了两次（正向和反向），除以 2 得标准 BC
            printf("%d %.6f\n", v, global_bc[v] / 2.0);
        }
    }

    // ========== Step 7: 打印计时统计（取各进程最大值） ==========
    double max_compute, max_comm;
    MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time,    &max_comm,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(stderr, "计算时间（GPU BFS+反向传播）: %.4f 秒\n", max_compute);
        fprintf(stderr, "通信时间（MPI_Reduce）:        %.4f 秒\n", max_comm);
    }

    freeCSR(g);
    MPI_Finalize();
    return 0;
}
