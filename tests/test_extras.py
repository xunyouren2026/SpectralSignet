"""
aiq-geometric-forensics.tests.test_extras — 追加模块回归（stability / family）
=====================================================================
覆盖 2.5.0 新增能力：
  * stability（溯源鲁棒性）：扰动函数边界、归属存活判据、可复现
  * family（家族族谱）：剖面 MAD 距离性质、SFT 同族识别、保守聚类

用法（本包根目录下）：
  python tests/test_extras.py
退出码：全过 0，任一失败 1。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules import schema  # noqa: E402
from modules.data import Metrics, find_baseline_models  # noqa: E402
from modules.family import family_tree, profile_mad  # noqa: E402
from modules.forensics import fingerprint  # noqa: E402
from modules.stability import perturb_descriptor, stability  # noqa: E402

_N_FAIL = [0]


def check(ok: bool, msg: str) -> None:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        _N_FAIL[0] += 1
    print(f"  [{mark}] {msg}")


def _desc_rng() -> tuple[np.ndarray, np.random.Generator]:
    """构造可复现的测试剖面与随机源（模拟 64 维重采样 Gamma 剖面）。"""
    rng = np.random.default_rng(0)
    desc = 0.3 + 0.3 * np.abs(rng.standard_normal(schema.PROFILE_K))
    return desc.astype(np.float64), np.random.default_rng(0)


def test_perturb_descriptor() -> None:
    print("\n[stability] 扰动函数边界防御")
    desc, rng = _desc_rng()
    # 1) 未知类型报错
    try:
        perturb_descriptor(desc, "bogus", 0.1, rng)
        check(False, "未知扰动类型应抛 ValueError")
    except ValueError:
        check(True, "未知扰动类型 → ValueError")
    # 2) 强度越界防御
    try:
        perturb_descriptor(desc, "sft", 1.5, rng)
        check(False, "强度>1 应被拒绝")
    except AssertionError:
        check(True, "强度>1 → AssertionError")
    # 3) 空输入防御
    try:
        perturb_descriptor(np.array([]), "sft", 0.1, rng)
        check(False, "空剖面应被拒绝")
    except AssertionError:
        check(True, "空剖面 → AssertionError")
    # 4) sft 扰动后形状不变、有界
    p = perturb_descriptor(desc, "sft", 0.05, rng)
    check(p.shape == desc.shape and np.isfinite(p).all(),
          f"sft 扰动形状保持且有限 ({p.shape})")
    # 5) trunc 丢弃比例≈amount（均值填充位点比例）
    rng2 = np.random.default_rng(1)
    t = perturb_descriptor(desc, "trunc", 0.5, rng2)
    mean_fill = float(np.mean(np.abs(t - desc.mean()) < 1e-12))
    check(mean_fill >= 0.45, f"trunc 0.5 均值填充比例≈{mean_fill:.2f}（≥0.45）")


def test_stability_survival() -> None:
    print("\n[stability] 归属存活判据（扰动后认回自身）")
    models = find_baseline_models()
    if "Qwen2.5-0.5B" not in models:
        check(False, "基准库缺 Qwen2.5-0.5B，跳过")
        return
    rep = stability("Qwen2.5-0.5B")
    o = rep["overall"]
    check(0.0 <= o["survival_rate"] <= 1.0,
          f"归属存活率∈[0,1]（实测 {o['survival_rate']:.3f}）")
    # 科学事实（v0 判据误报后实测确认）：剖面形状不变性 → 强扰动仍认回自身
    check(rep["perturbations"]["sft"]["overall"] == 1.0,
          "SFT 微扰 → 100% 归属存活（微调不改身份）")
    check(rep["perturbations"]["scale"]["overall"] == 1.0,
          "幅度改写 → 100% 归属存活（形状不变性）")
    check(o["max_displacement"] > 0.0,
          f"位移监控记录强度边界（max={o['max_displacement']:.4f}）")
    # 可复现性：同 seed 两次运行结果一致
    rep2 = stability("Qwen2.5-0.5B")
    check(np.isclose(o["survival_rate"], rep2["overall"]["survival_rate"]),
          "同 seed 两次运行归属率一致（可复现）")


def test_family_profile_mad() -> None:
    print("\n[family] 剖面 MAD 距离性质")
    models = find_baseline_models()
    vectors = {n: fingerprint(Metrics(n))["vector"] for n in models}
    d = profile_mad(vectors)
    # 对称性 + 对角为 0 + 非负
    ok = all(abs(d[a][b] - d[b][a]) < 1e-12 and d[a][a] == 0.0
             and d[a][b] >= 0.0
             for a in models for b in models)
    check(ok, "距离矩阵对称 / 对角=0 / 非负")
    # SFT 同族识别：Qwen base↔Instruct 剖面 MAD 极小（实测 0.0031）
    if "Qwen2.5-0.5B" in models and "Qwen2.5-0.5B-Instruct" in models:
        d_bi = d["Qwen2.5-0.5B"]["Qwen2.5-0.5B-Instruct"]
        check(d_bi < schema.FAM_ATTR_DIST,
              f"Qwen base↔Instruct 剖面 MAD={d_bi:.4f} < 同族阈值 "
              f"{schema.FAM_ATTR_DIST}（SFT 不改身份）")


def test_family_conservative_cluster() -> None:
    print("\n[family] 保守聚类（仅 SFT 级同族凝聚，避免 chaining）")
    models = find_baseline_models()
    rep = family_tree()
    check(rep["n_clusters"] >= 2,
          f"5 模型保守聚类 ≥2 簇（实测 {rep['n_clusters']}，避免跨家族误合）")
    if {"Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"} <= set(models):
        la, lb = (rep["kinship"]["Qwen2.5-0.5B"]["cluster"],
                  rep["kinship"]["Qwen2.5-0.5B-Instruct"]["cluster"])
        check(la == lb, "Qwen base 与 Instruct 同簇（SFT 级同族凝聚）")
        check(rep["kinship"]["Qwen2.5-0.5B"]["relation"] == "同族(SFT级)",
              "Qwen base 最近邻亲缘标签 = 同族(SFT级)")
    # 亲缘表最近邻关系有限且非 NaN
    ok = all(np.isfinite(k["nn_distance"]) for k in rep["kinship"].values())
    check(ok, "全部最近邻距离有限")


def main() -> int:
    print("=" * 60)
    print("追加模块回归（stability / family）")
    print("=" * 60)
    test_perturb_descriptor()
    test_stability_survival()
    test_family_profile_mad()
    test_family_conservative_cluster()
    print("-" * 60)
    ok = _N_FAIL[0] == 0
    print(f"追加回归汇总: FAIL={_N_FAIL[0]}  →  "
          + ("全部通过" if ok else "存在失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
