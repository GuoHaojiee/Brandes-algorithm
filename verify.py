#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py - 验证 Brandes BC 程序输出的正确性
使用 NetworkX 作为参考实现，允许误差 1e-6

用法（二选一）:
  # 方式1: 程序输出重定向到文件后验证
  mpirun -np 2 ./brandes test.txt > out.txt
  python3 verify.py test.txt out.txt

  # 方式2: 管道直接验证（程序 BC 输出走 stdout，计时走 stderr 不干扰）
  mpirun -np 2 ./brandes test.txt | python3 verify.py test.txt
"""

import sys
import networkx as nx

TOLERANCE = 1e-6  # 允许的最大绝对误差


def load_graph(fname):
    """
    从边列表文件加载无向图
    支持以 '#' 开头的注释行，空行自动跳过
    """
    G = nx.Graph()
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
    return G


def load_program_output(fname=None):
    """
    读取程序输出（从文件或 stdin）
    格式: 每行 "节点id BC值"
    返回: dict {节点id: BC值}
    """
    result = {}
    src = open(fname) if fname else sys.stdin
    try:
        for line in src:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    node_id = int(parts[0])
                    bc_val  = float(parts[1])
                    result[node_id] = bc_val
                except ValueError:
                    pass  # 跳过无法解析的行（如计时输出）
    finally:
        if fname:
            src.close()
    return result


def main():
    if len(sys.argv) < 2:
        print("用法: python3 verify.py <图文件> [程序输出文件]")
        sys.exit(1)

    graph_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # 用 NetworkX 计算参考 BC 值
    # normalized=False: 不归一化（绝对路径计数）
    # NetworkX 内部对无向图自动除以 2，与本程序输出一致
    G = load_graph(graph_file)
    nx_bc = nx.betweenness_centrality(G, normalized=False)

    # 读取程序输出
    prog_bc = load_program_output(output_file)

    # 逐节点比较
    failed = []
    for node in sorted(G.nodes()):
        expected = nx_bc.get(node, 0.0)
        got      = prog_bc.get(node, 0.0)
        if abs(expected - got) > TOLERANCE:
            failed.append((node, expected, got))

    # 输出验证结果
    if not failed:
        print(f"PASS - 所有 {G.number_of_nodes()} 个节点 BC 值正确（误差 < {TOLERANCE}）")
    else:
        print(f"FAIL - {len(failed)} / {G.number_of_nodes()} 个节点 BC 值不符:")
        print(f"{'节点':>6}  {'期望值(NetworkX)':>18}  {'实际值(程序)':>14}  {'误差':>12}")
        for node, expected, got in failed:
            print(f"{node:6d}  {expected:18.6f}  {got:14.6f}  {abs(expected-got):12.2e}")


if __name__ == '__main__':
    main()
