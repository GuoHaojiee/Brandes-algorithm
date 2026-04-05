#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_stdlib.py - 不依赖 networkx，用纯标准库实现 Brandes 算法作参考
适用于集群无法安装第三方包的情况

用法:
  python3 verify_stdlib.py test.txt out.txt
  mpirun -np 2 ./brandes test.txt | python3 verify_stdlib.py test.txt
"""

import sys
from collections import deque, defaultdict

TOLERANCE = 1e-6


def load_graph(fname):
    """从边列表文件加载无向图，返回邻接表和节点集合"""
    adj = defaultdict(set)
    nodes = set()
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                adj[u].add(v)
                adj[v].add(u)
                nodes.add(u)
                nodes.add(v)
    return adj, sorted(nodes)


def brandes_bc(adj, nodes):
    """
    纯 Python 实现 Brandes 介数中心性算法（参考实现）
    无向图结果除以 2（与 NetworkX normalized=False 一致）
    """
    bc = {v: 0.0 for v in nodes}

    for s in nodes:
        # BFS 初始化
        stack   = []                          # BFS 访问顺序（用于逆向传播）
        pred    = {v: [] for v in nodes}      # 前驱节点列表
        sigma   = {v: 0.0 for v in nodes}    # 最短路径数
        dist    = {v: -1  for v in nodes}    # BFS 距离
        sigma[s] = 1.0
        dist[s]  = 0
        queue    = deque([s])

        # 正向 BFS
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] == -1:             # w 首次访问
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:   # v 是 w 的前驱
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # 反向累积依赖值
        delta = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    # 无向图：每对 (s,t) 被计算两次，除以 2
    for v in nodes:
        bc[v] /= 2.0

    return bc


def load_program_output(fname=None):
    """读取程序输出，格式: 节点id BC值"""
    result = {}
    src = open(fname) if fname else sys.stdin
    try:
        for line in src:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    result[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
    finally:
        if fname:
            src.close()
    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python3 verify_stdlib.py <图文件> [程序输出文件]")
        sys.exit(1)

    graph_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    adj, nodes = load_graph(graph_file)
    ref_bc      = brandes_bc(adj, nodes)
    prog_bc     = load_program_output(output_file)

    failed = []
    for node in nodes:
        expected = ref_bc.get(node, 0.0)
        got      = prog_bc.get(node, 0.0)
        if abs(expected - got) > TOLERANCE:
            failed.append((node, expected, got))

    if not failed:
        print(f"PASS - 所有 {len(nodes)} 个节点 BC 值正确（误差 < {TOLERANCE}）")
    else:
        print(f"FAIL - {len(failed)} / {len(nodes)} 个节点 BC 值不符:")
        print(f"{'节点':>6}  {'期望值(参考)':>16}  {'实际值(程序)':>14}  {'误差':>12}")
        for node, expected, got in failed:
            print(f"{node:6d}  {expected:16.6f}  {got:14.6f}  {abs(expected-got):12.2e}")


if __name__ == '__main__':
    main()
