#pragma once
#include <vector>
#include <algorithm>

/*
 * CSR（压缩稀疏行）格式图结构体
 * 无向图双向存储：每条无向边 (u,v) 存为 u->v 和 v->u 两条有向边
 */
struct CSRGraph {
    int  n;       // 节点总数（节点编号 0 ~ n-1）
    int  m;       // 有向边总数（无向图双向存储，m = 2 * 无向边数）
    int* offset;  // offset[v]..offset[v+1]-1 为节点v的出边在dest中的范围
    int* dest;    // 边的目标节点列表，长度为m
};

/*
 * 从无向边列表构建CSR格式图（自动添加反向边）
 * 参数:
 *   n     - 节点总数
 *   edges - 无向边列表，每条边只需存一次
 * 返回:
 *   构建好的CSRGraph（使用new分配内存，需调用freeCSR释放）
 */
inline CSRGraph buildCSR(int n, const std::vector<std::pair<int,int>>& edges) {
    CSRGraph g;
    g.n = n;
    g.m = (int)edges.size() * 2; // 双向存储

    // 统计每个节点的出度
    std::vector<int> deg(n, 0);
    for (const auto& e : edges) {
        deg[e.first]++;
        deg[e.second]++;
    }

    // 构建前缀和偏移数组 offset
    g.offset = new int[n + 1];
    g.offset[0] = 0;
    for (int v = 0; v < n; v++)
        g.offset[v + 1] = g.offset[v] + deg[v];

    // 填充目标节点数组（正向边和反向边各一条）
    g.dest = new int[g.m];
    std::vector<int> cur(g.offset, g.offset + n); // 每个节点当前填充位置
    for (const auto& e : edges) {
        g.dest[cur[e.first]++]  = e.second;
        g.dest[cur[e.second]++] = e.first;
    }
    return g;
}

/*
 * 释放CSR图占用的内存
 */
inline void freeCSR(CSRGraph& g) {
    delete[] g.offset;
    delete[] g.dest;
    g.offset = nullptr;
    g.dest   = nullptr;
}
