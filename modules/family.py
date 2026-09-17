"""
aiq-geometric-forensics.modules.family — 家族族谱（跨模型指纹聚类谱系）
=====================================================================
把基准库/任意模型集合的几何指纹做亲缘关系分析，输出"模型家族族谱"：
  * 两两指纹距离矩阵（欧氏，64 维剖面 + 5 维曲率口径）
  * 亲缘分级：同族（SFT 级，距离 < FAM_ATTR_DIST）/ 近亲（同家族扩参变体，
    < FAM_AFFINITY_DIST）/ 远亲（其余）
  * 自组织凝聚：单链接 agglomerative 聚类（纯 numpy，不依赖 scipy），
    输出聚合族系（largest-cluster 标签 + 每模型归属）

典型调用：
  import modules.family as F
  rep = F.family_tree()                     # 用 baselines/ 全部模型
  print(F.format_family(rep))
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import schema
from .data import Metrics, find_baseline_models
from .forensics import fingerprint

# 亲缘阈值统一取自 schema（单一事实源）
_ATTR = schema.FAM_ATTR_DIST          # 同族（SFT 级亲缘）
_AFFIN = schema.FAM_AFFINITY_DIST     # 近亲（扩参/变体）


def profile_mad(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """两两指纹**剖面段 MAD** 距离矩阵（对称，含对角线=0）。

    口径说明（承接 schema.FAM_* 注释）：只取指纹前 PROFILE_K 维
    （重采样 Gamma 剖面），不用完整欧氏——标量段 K%≈45 量级会把
    剖面差异淹没（实测完整欧氏 5 档全部 '远亲'、零凝聚，属口径失真）。
    """
    K = int(schema.PROFILE_K)
    names = list(vectors)
    out: dict[str, dict[str, float]] = {n: {} for n in names}
    for i, a in enumerate(names):
        pa = vectors[a][:K]
        for b in names[i:]:                          # 只算上三角，镜像填充
            mad = float(np.mean(np.abs(pa - vectors[b][:K])))
            out[a][b] = out[b][a] = mad
    return out


def relate(dist: float) -> str:
    """由两模型指纹距离给出亲缘标签：同族/近亲/远亲。"""
    if dist < _ATTR:
        return "同族(SFT级)"
    if dist < _AFFIN:
        return "近亲(扩参/变体)"
    return "远亲"


def agglomerate(vectors: dict[str, np.ndarray],
                threshold: float = _AFFIN) -> dict[str, Any]:
    """单链接 agglomerative 聚类（纯 numpy）。

    参数：
      vectors: {模型名: 指纹向量}
      threshold: 合并距离上限（聚类只合并距离 < 阈值 的簇）
    返回：{'labels': {模型: 簇id}, 'clusters': {簇id: [模型...]}, 'n_clusters': int}
    """
    names = list(vectors)
    n = len(names)
    K = int(schema.PROFILE_K)                         # 剖面向量维数（MAD 口径）
    # 距离矩阵（剖面段 MAD，与 profile_mad 同口径）
    dist = np.zeros((n, n))
    for i in range(n):
        pi = vectors[names[i]][:K]
        for j in range(i + 1, n):
            d = float(np.mean(np.abs(pi - vectors[names[j]][:K])))
            dist[i, j] = dist[j, i] = d

    clusters = {i: [i] for i in range(n)}             # 初始：每点一簇
    # 单链接：簇间距离 = 两簇任一点对最小距离
    def cluster_dist(c1: list[int], c2: list[int]) -> float:
        return float(min(dist[i, j] for i in c1 for j in c2))

    merged: list[dict[str, Any]] = []                 # 合并历史（谱系记录）
    while len(clusters) > 1:
        keys = list(clusters)
        best, best_d = None, float("inf")
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                d = cluster_dist(clusters[keys[x]], clusters[keys[y]])
                if d < best_d:
                    best, best_d = (keys[x], keys[y]), d
        if best is None or best_d >= threshold:       # 无可合并（或超过阈值）
            break
        cid1, cid2 = best
        merged.append({"clusters": [cid1, cid2], "distance": best_d,
                       "size": len(clusters[cid1]) + len(clusters[cid2])})
        clusters[cid1] = clusters[cid1] + clusters[cid2]   # 合并（保留小 id 腾挪）
        del clusters[cid2]

    labels: dict[str, int] = {}
    clusters_out: dict[int, list[str]] = {}
    for nid, members in clusters.items():
        clusters_out[nid] = [names[i] for i in members]
        for i in members:
            labels[names[i]] = nid
    return {"labels": labels, "clusters": clusters_out,
            "n_clusters": len(clusters_out), "merged": merged}


# ---------------------------------------------------------------- 主入口
def family_tree(models: list[str] | None = None) -> dict[str, Any]:
    """家族族谱 —— 对给定模型集合（默认 baselines/ 全部）做指纹亲缘聚类。

    返回：距离矩阵 + 亲缘表 + 凝聚族系。
    """
    names = models or find_baseline_models()          # 默认全部基准
    vectors = {n: fingerprint(Metrics(n))["vector"] for n in names}
    dist = profile_mad(vectors)                          # 剖面 MAD 距离（口径见函数 docstring）
    # 聚类用**保守**阈值 _ATTR（仅 SFT 级同族凝聚），避免单链接 chaining——
    # 若用 _AFFIN，TinyLlama 会充当"桥梁"把 Qwen 与 BLOOM 连成 1 簇（误导跨家族）。
    # 亲缘表（kinship）保留完整近亲/远亲视角，两视角互补。
    agg = agglomerate(vectors, threshold=_ATTR)

    # 亲缘关系表（每模型 → 其最近邻 + 亲缘标签）
    kinship: dict[str, dict[str, Any]] = {}
    for a in names:
        # 最近邻：除自身外距离最小的模型（排除对角线 0）
        nearest = min(dist[a], key=lambda b: dist[a][b] if b != a else float("inf"))
        kinship[a] = {"nearest": nearest,
                      "nn_distance": dist[a][nearest],
                      "relation": relate(dist[a][nearest]),
                      "cluster": agg["labels"].get(a)}

    return {"models": names, "distance_matrix": dist,
            "kinship": kinship, "clusters": agg["clusters"],
            "n_clusters": agg["n_clusters"], "merges": agg["merged"],
            "thresholds": {"同族(SFT级)": _ATTR, "近亲(扩参/变体)": _AFFIN}}


# ---------------------------------------------------------------- 格式化输出
def format_family(rep: dict[str, Any]) -> str:
    """把族谱报告渲染成可读文本（距离表 + 亲缘 + 簇）。"""
    lines = ["=" * 72, "家族族谱（指纹亲缘聚类）", "=" * 72,
             f"  模型数: {len(rep['models'])}  |  "
             f"凝聚簇: {rep['n_clusters']}",
             f"  亲缘阈值: 同族<{rep['thresholds'].get('同族(SFT级)')}  "
             f"近亲<{rep['thresholds'].get('近亲(扩参/变体)')}",
             "", "  模型最近亲缘:"]
    for name, k in rep["kinship"].items():
        lines.append(f"    {name:<32} → {k['nearest']:<28} "
                     f"d={k['nn_distance']:.4f} [{k['relation']}]")
    if rep["merges"]:
        lines.append("")                                 # 空行分隔
        lines.append("  凝聚顺序（簇合并历史）:")
        for i, m in enumerate(rep["merges"], 1):
            lines.append(f"    {i:>2}. 合并簇{m['clusters']} "
                         f"距离={m['distance']:.4f} → 规模 {m['size']}")
    lines.append("")                                   # 空行分隔
    lines.append("  凝聚后各簇成员:")
    for cid, members in rep["clusters"].items():
        lines.append(f"    簇{cid}: {', '.join(members)}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 直接运行
if __name__ == "__main__":
    print(format_family(family_tree()))
