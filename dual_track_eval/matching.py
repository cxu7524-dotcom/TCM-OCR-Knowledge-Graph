# -*- coding: utf-8 -*-
"""组级匹配：最大基数 → 最大总 IoU → 完全并列时字典序裁定。

为什么不用单次带权二分图匹配
----------------------------
把「基数」塞进权重（例如 w = BIG + IoU）在浮点下不可靠：BIG 需要大于
所有 IoU 之和的差，规模一变就要重调；而且一旦权重相等，「多配一组」和
「少配一组但 IoU 更高」会被混在一起比较。

这里的做法是**结构上分离**：
1. 先求最大流（流值 k 就是最大基数），这一步与 IoU 无关；
2. 在「恰好 k 条边」的约束下最小化总代价，等价于最大化 ΣIoU；
3. 只有 (k, ΣIoU) 完全相等时才进入字典序搜索。

代价函数为什么是 −IoU 而不是 M − IoU
------------------------------------
上一版用 `cost = M - IoU`（M = 有效边数+1），目的是让代价恒正、方便最短路。
但 M 一大，`M - 0.5` 与 `M - 0.5000000000000001` 在双精度下会相减成同一个数
（M=3 时两者都是 2.5），于是「更优一点点」的边被当成等价，回退路径选错边。
现在直接用 `cost = -IoU`：负代价在 SPFA（Bellman-Ford 类）里本来就能处理，
二分图的初始残量网络无环、无负环，SSP 每次沿最短路增广后仍保持无负环，
所以既正确又不损失精度。

规模限制只影响第三目标
----------------------
`lex_search_limits` 只约束「完全并列时取字典序最小」这一步。基数与总 IoU
由最小费用流保证，规模再大也不受影响；超过限制时结果里显式标
`tie_break_status=skipped_scale_limit`，并在 scope 里写明后果。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 字典序兜底搜索的规模上限：超过就退回最小费用流的确定解，并在结果里显式标注
LEX_LIMIT_NODES = 600   # n_pred * n_gt
LEX_LIMIT_K = 12        # 最大基数
TIME_BUDGET_S = 20.0    # 兜底搜索的时间上限，超时同样退回并标注


class _MCMF:
    """最小费用最大流（SPFA 增广），二分图规模很小，够用且无浮点累积误差。"""

    def __init__(self, n: int):
        self.n = n
        self.g: List[List[List[Any]]] = [[] for _ in range(n)]

    def add(self, u: int, v: int, cap: int, cost: float) -> int:
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])
        return len(self.g[u]) - 1

    def flow(self, s: int, t: int) -> Tuple[int, float]:
        total_flow, total_cost = 0, 0.0
        while True:
            dist = [float('inf')] * self.n
            inq = [False] * self.n
            prevv = [-1] * self.n
            preve = [-1] * self.n
            dist[s] = 0.0
            queue = [s]
            inq[s] = True
            while queue:
                u = queue.pop(0)
                inq[u] = False
                for i, (v, cap, cost, _rev) in enumerate(self.g[u]):
                    # 严格比较：不留固定容差，否则会跳过「只优一点点」的路径
                    # （那正是被漏掉的最优解）。浮点累加顺序固定，结果可复现。
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prevv[v], preve[v] = u, i
                        if not inq[v]:
                            queue.append(v)
                            inq[v] = True
            if math.isinf(dist[t]):
                break
            # 每条边 cap 都是 1 的链，一次增广 1 单位
            v = t
            while v != s:
                u, i = prevv[v], preve[v]
                self.g[u][i][1] -= 1
                rev = self.g[u][i][3]
                self.g[v][rev][1] += 1
                v = u
            total_flow += 1
            total_cost += dist[t]
        return total_flow, total_cost


def _edges(matrix: Sequence[Sequence[float]], thr: float) -> List[Tuple[int, int, float]]:
    out = []
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v >= thr:        # 有效边：原值比较，无 epsilon
                out.append((i, j, float(v)))
    return out


def solve_min_cost_max_flow(matrix: Sequence[Sequence[float]], thr: float) -> Dict[str, Any]:
    """返回 {cardinality, total_iou, pairs, method}；pairs 按预测序号升序。"""
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    edges = _edges(matrix, thr)
    if not edges or n == 0 or m == 0:
        return {'cardinality': 0, 'total_iou': 0.0, 'pairs': [], 'method': 'no_edge'}

    s, t = 0, n + m + 1
    net = _MCMF(n + m + 2)
    for i in range(n):
        net.add(s, 1 + i, 1, 0.0)
    for j in range(m):
        net.add(1 + n + j, t, 1, 0.0)
    idx = {}
    for i, j, v in edges:
        # cost = -IoU：避免 M - IoU 在 M 较大时把小差异抹平
        idx[(i, j)] = net.add(1 + i, 1 + n + j, 1, -v)

    k, _cost = net.flow(s, t)

    pairs = []
    for (i, j), e in idx.items():
        # 正向边被用掉时 cap=0、反向边 cap=1
        if net.g[1 + i][e][1] == 0:
            pairs.append((i, j))
    pairs.sort()
    total = _sum_pairs(matrix, pairs)
    return {'cardinality': k, 'total_iou': total,
            'pairs': [[i, j] for i, j in pairs], 'method': 'min_cost_max_flow'}


def _best_exact_search(matrix: Sequence[Sequence[float]], thr: float, k: int,
                       seed_total: float,
                       time_budget_s: float = TIME_BUDGET_S) -> Dict[str, Any]:
    """在固定基数 k 下，精确求「最大总 IoU，且该值下字典序最小」。

    为什么不能只用「等于 seed_total 才算并列」：
      seed_total 来自最小费用流，它自己可能因浮点累加被算小一点点。若只接受
      == seed_total，就会把一个真实更高分的解当成非法而丢掉。所以这里把第二目标
      也一起精确求解，最后再与 seed_total 对账。

    并列判定的口径：**总 IoU 用浮点精确相等判断**（不用 1e-9 之类的容差）。
      - 同一组配对在两条路径上的累加顺序一致（都按预测升序），所以相等是可复现的；
      - 用容差会把 0.5000000001 与 0.5000000002 当成并列，于是选中低分那条——
        这正是本轮要修的缺陷。
    """
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    row_max = [max([v for v in row if v >= thr], default=0.0) for row in matrix]
    suffix_max = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + row_max[i]

    used = [False] * m
    cur: List[Tuple[int, int]] = []
    best = {'total': -1.0, 'pairs': []}
    nodes = {'count': 0}
    deadline = time.monotonic() + float(time_budget_s)
    timed_out = {'hit': False}

    def dfs(i: int, need: int, acc: float) -> None:
        nodes['count'] += 1
        if nodes['count'] % 4096 == 0 and time.monotonic() > deadline:
            timed_out['hit'] = True
            return
        if timed_out['hit']:
            return
        if need == 0:
            if acc > best['total']:          # 严格大于：并列时保留先找到的（字典序更小）
                best['total'] = acc
                best['pairs'] = list(cur)
            return
        if i >= n or n - i < need:
            return
        # 上界剪枝：剩余预测各自最大可用之和都补不上，就不用往下走
        if acc + suffix_max[i] < best['total']:
            return
        for j in range(m):
            v = float(matrix[i][j])
            if not used[j] and v >= thr:
                used[j] = True
                cur.append((i, j))
                dfs(i + 1, need - 1, acc + v)
                cur.pop()
                used[j] = False
        dfs(i + 1, need, acc)

    dfs(0, k, 0.0)
    if timed_out['hit']:
        return {'status': 'timeout', 'best_total': None, 'pairs': None,
                'nodes': nodes['count'], 'time_budget_s': time_budget_s}
    if best['total'] < 0:
        return {'status': 'no_solution', 'best_total': None, 'pairs': None,
                'nodes': nodes['count']}
    return {'status': 'exact', 'best_total': best['total'], 'pairs': best['pairs'],
            'nodes': nodes['count'], 'seed_total': seed_total,
            'improved_over_seed': best['total'] > seed_total}



def solve_exhaustive(matrix: Sequence[Sequence[float]], thr: float) -> Dict[str, Any]:
    """完全枚举所有匹配，按 (基数, 总 IoU, 字典序) 取最优；只用于测试对照。"""
    import itertools
    n = len(matrix)
    m = len(matrix[0]) if n else 0
    best_key = None
    best_pairs: List[Tuple[int, int]] = []
    for k in range(min(n, m), 0, -1):
        found = False
        for preds in itertools.combinations(range(n), k):
            for gts in itertools.permutations(range(m), k):
                pairs = list(zip(preds, gts))
                if any(matrix[p][g] < thr for p, g in pairs):
                    continue
                total = float(sum(matrix[p][g] for p, g in pairs))
                key = (k, total)
                if best_key is None or key > best_key or (key == best_key and
                                                          tuple(pairs) < tuple(best_pairs)):
                    best_key, best_pairs = key, pairs
                found = True
        if found:
            break
    if best_key is None:
        return {'cardinality': 0, 'total_iou': 0.0, 'pairs': [], 'method': 'exhaustive'}
    return {'cardinality': best_key[0], 'total_iou': best_key[1],
            'pairs': [[i, j] for i, j in best_pairs], 'method': 'exhaustive'}


def resolve_limits(limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = {'max_nodes': LEX_LIMIT_NODES, 'max_k': LEX_LIMIT_K,
            'time_budget_s': TIME_BUDGET_S}
    if limits:
        for k in base:
            if k in limits and limits[k] is not None:
                base[k] = limits[k]
    return base


def match_groups(matrix: Sequence[Sequence[float]], thr: float,
                 lex_tiebreak: bool = True,
                 limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对外入口：最大基数 → 最大总 IoU → 完全并列时字典序。

    返回的 total_iou **永远**由最终 pairs 重算，不沿用求解过程中的中间值
    （否则会出现在缺陷版本里那样「pairs 是低分解、total 却是高分解」）。
    """
    base = solve_min_cost_max_flow(matrix, thr)
    base['threshold'] = thr
    base['tie_break_applied'] = False
    base['tie_break_status'] = 'not_needed'
    base['total_iou'] = _sum_pairs(matrix, base['pairs'])

    if base['cardinality'] == 0 or not lex_tiebreak:
        if base['cardinality'] == 0 and lex_tiebreak:
            base['tie_break_status'] = 'no_match'
        return base

    n = len(matrix)
    m = len(matrix[0]) if n else 0
    k = base['cardinality']
    limit = resolve_limits(limits)
    base['limits_used'] = limit
    if n * m > limit['max_nodes'] or k > limit['max_k']:
        base.update(tie_break_status='skipped_scale_limit',
                    tie_break_scope={'applied': False, 'limits': limit,
                                     'problem_size': {'preds': n, 'gts': m, 'k': k},
                                     'consequence': '基数与总 IoU 仍是最优（最小费用流保证），'
                                                    '但「完全并列时取字典序最小」这一步没有执行；'
                                                    '不得声称第三目标已满足'})
        return base

    exact = _best_exact_search(matrix, thr, k, base['total_iou'],
                               time_budget_s=limit['time_budget_s'])
    base['tie_break_scope'] = {'applied': True, 'limits': limit,
                               'problem_size': {'preds': n, 'gts': m, 'k': k},
                               'nodes_visited': exact.get('nodes')}
    if exact['status'] != 'exact':
        base.update(tie_break_status=f"{exact['status']}_fallback_mcmf",
                    tie_break_scope={**base['tie_break_scope'], 'applied': False,
                                     'consequence': '兜底搜索未完成，已退回最小费用流的确定解'})
        return base

    pairs = [[i, j] for i, j in exact['pairs']]
    base['mcmf_pairs'] = base['pairs']
    base['mcmf_total_iou'] = base['total_iou']
    base['pairs'] = pairs
    base['total_iou'] = _sum_pairs(matrix, pairs)
    base['tie_break_status'] = 'exact'
    if exact['improved_over_seed']:
        # 说明最小费用流的浮点累加把最优值算小了一点，以精确搜索为准
        base['tie_break_status'] = 'exact_improved_over_mcmf'
        base['method'] = 'min_cost_max_flow+exact_search'
    elif pairs != base['mcmf_pairs']:
        base['tie_break_applied'] = True
        base['method'] = 'min_cost_max_flow+lex_tiebreak'
    return base


def _sum_pairs(matrix: Sequence[Sequence[float]], pairs: Sequence[Sequence[int]]) -> float:
    """按预测升序累加，和 DFS 的累加顺序一致，保证可比较。"""
    return float(sum(float(matrix[i][j]) for i, j in sorted(pairs)))

