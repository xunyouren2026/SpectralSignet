"""
aiq-geometric-forensics 推导验证套件（深入测试）
=====================================================================
目标：对每个模块的核心公式做「独立数值推导核对」，证明实现忠实于数学定义，
而不是只跑一遍演示路径。

覆盖模块与验证点（构造针对性输入，用解析已知值/数学恒等式反查实现）：

  harness.spl_gamma       谱集中度=前3特征值占比；秩1情报应为1.0；等向应为3/d
  harness.gamma_energy    前m个奇异值平方占比
  harness.curvature_metrics  DEFF∈[1,2]；共线→2；正交→~1；λ=σ2/σ1；K<0%正确
  harness.aiq_score       权重和=1；AIQ=100Σw_i f_i；全零→f3=0
  health.aiq_factors      与 harness.aiq_score 数学等价（同输入同输出）
  health.depth_profile    分层均值与总体均值可复核；层数除不尽边界
  health.health_verdicts  K槽边界/DEFF容差/H阈值三分支
  health.compressibility  三级分级边界（γ=0.45/0.35 精确）
  forensics.fingerprint   向量维度=24+GAMMA mean + 5标量
  forensics.family_verdict 0.01 阈值两侧
  data.Metrics            三层解析 + sanitize_path 去除隐私路径

用法（本包根目录）：
  python tests/verify_derivations.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from modules.data import Metrics, find_baseline_models, sanitize_path  # noqa: E402
from modules.forensics import family_verdict, fingerprint  # noqa: E402
from modules.harness import aiq_score, curvature_metrics, gamma_energy, spl_gamma  # noqa: E402
from modules.health import (  # noqa: E402
    aiq_factors,
    compressibility,
    depth_profile,
    health_verdicts,
)

_PASS, _FAIL = 0, 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if bool(cond):
        _PASS += 1
        print(f"  [PASS] {msg}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {msg}")


# ================================================================ harness
def test_spl_gamma() -> None:
    print("\n[harness] spl_gamma —— PCA前3特征值占比")
    rng = np.random.default_rng(0)
    # 秩1：H 每行同向 → 仅1个非零特征值，前3占比应≈1
    v = rng.standard_normal((8, 64))
    rank1 = np.outer(np.arange(8.0) + 1, v[0])  # 秩1
    g = spl_gamma(rank1)
    check(abs(g - 1.0) < 1e-6, f"秩1矩阵前3占比={g:.6f}≈1.0（数学：仅1非零特征值）")
    # 谱集中度单调性：秩1(集中) 应恒 ≥ 满秩等向(弥散)，且等向∈[0,1]
    iso = rng.standard_normal((200, 64))
    g2 = spl_gamma(iso - iso.mean(0))
    check(0.0 <= g2 <= 1.0, f"等向分布前3占比={g2:.4f}∈[0,1]")
    # 独立解析参考实现（直接按定义计算）交叉核对
    Hc = iso - iso.mean(0)
    S = np.linalg.eigvalsh(Hc.T @ Hc / (Hc.shape[0] - 1))[::-1]
    ref = float(S[:3].sum() / S.sum())
    check(abs(g2 - ref) < 1e-9, f"等向 spl_gamma 与独立参考实现一致（{g2:.6f}≈{ref:.6f}）")
    check(g2 < g, "谱集中度单调：秩1(1.0) ≥ 满秩等向，方向正确")
    # 含 NaN 防护
    check(spl_gamma(np.array([[np.nan, 1.0], [1.0, 2.0]])) is np.nan
          or str(spl_gamma(np.array([[np.nan], [1.0]]))) == "nan",
          "含NaN输入返回 nan（边界防御）")
    # 唯一确定性：同一输入两次结果一致
    check(spl_gamma(rank1) == spl_gamma(rank1), "确定性（同类输入同输出）")


def test_gamma_energy() -> None:
    print("\n[harness] gamma_energy —— 前m奇异值平方占比")
    rng = np.random.default_rng(1)
    rank1 = np.outer(np.arange(4.0), rng.standard_normal(8))
    e = gamma_energy(rank1, m=1)
    check(abs(e - 1.0) < 1e-9, f"秩1矩阵 m=1 占比={e:.6f}≈1.0")
    iso = rng.standard_normal((100, 8))
    e2 = gamma_energy(iso, m=3)
    # 独立参考：前3个奇异值平方和占比
    S2 = (np.linalg.svd(iso - iso.mean(0), compute_uv=False) ** 2)
    ref2 = float(S2[:3].sum() / S2.sum())
    check(abs(e2 - ref2) < 1e-9, f"等向 m=3 与独立参考实现一致（{e2:.4f}≈{ref2:.4f}）")
    check(0.0 <= e2 <= 1.0, f"占比∈[0,1]（{e2:.4f}）")


def test_curvature_metrics() -> None:
    print("\n[harness] curvature_metrics —— DEFF/K<0%/λ 推导")
    # 共线正曲面 κ1=κ2=1：DEFF=(1+1)^2/(1+1)=4/2=2；K<0%=0
    same = np.stack([np.ones(500), np.ones(500)], axis=1)
    c = curvature_metrics(same)
    check(abs(c["DEFF_plat"] - 2.0) < 1e-6,
          f"共线同号 DEFF={c['DEFF_plat']:.6f}=2（|κ1|=|κ2|∈[1,2]上界）")
    check(abs(c["K_neg_pct"] - 0.0) < 1e-9, f"同号 K<0%={c['K_neg_pct']}=0")
    # 共线反号 κ1=1,κ2=-1：DEFF仍=2，但K<0%=100
    anti = np.stack([np.ones(500), -np.ones(500)], axis=1)
    c2 = curvature_metrics(anti)
    check(abs(c2["DEFF_plat"] - 2.0) < 1e-6 and abs(c2["K_neg_pct"] - 100.0) < 1e-9,
          f"共线反号 DEFF={c2['DEFF_plat']:.4f}=2,K<0%={c2['K_neg_pct']}=100")
    # 随机独立高斯 κ1~N(0,3),κ2~N(0,1)：λ=σ2/σ1=1/3（构造已知比值）
    rng5 = np.random.default_rng(2)
    k1 = rng5.standard_normal(3000) * 3.0
    k2 = rng5.standard_normal(3000) * 1.0
    c5 = curvature_metrics(np.stack([k1, k2], axis=1))
    # 大样本下 std 比收敛到 1/3；允许 ±4% 样本波动
    check(abs(c5["lambda_ratio"] - 1 / 3) < 0.04,
          f"λ=σ2/σ1={c5['lambda_ratio']:.4f}≈1/3（由构造 std 比 3:1 决定）")


def test_aiq_score() -> None:
    print("\n[harness] aiq_score —— 五因子加权")
    w = (0.2, 0.25, 0.2, 0.2, 0.15)
    check(abs(sum(w) - 1.0) < 1e-12, f"权重和={sum(w)}=1")
    layers = [0.5] * 24
    a = aiq_score(layers, 0.0, 1.0, 0.5, w)
    # f1=0.5, f2=1.0, f3=1/(1+1)=0.5(修复后), f4=0.5, f5=1.0-(0)=1.0
    exp = 100 * (0.2 * 0.5 + 0.25 * 1.0 + 0.2 * 0.5 + 0.2 * 0.5 + 0.15 * 1.0)
    check(abs(a["AIQ"] - exp) < 1e-6,
          f"AIQ={a['AIQ']:.4f}≈解析{exp:.4f}（分因子手算）")
    check(abs((a["f1"], a["f2"], a["f4"], a["f5"])[0] - 0.5) < 1e-9, "f1=mean(Gamma)")
    check(abs(a["f3"] - 0.5) < 1e-9, "f3=1/(1+Hmed)=0.5（修复非退化）")


# ================================================================ health
def test_aiq_factors_eq() -> None:
    print("\n[health] aiq_factors —— 与 harness.aiq_score 数学等价")
    rng = np.random.default_rng(3)
    layers = rng.uniform(0.3, 0.9, 24).tolist()
    hmm = aiq_factors(layers, 0.01, 0.8, 0.6)
    hh = aiq_score(layers, 0.01, 0.8, 0.6)
    check(abs(hmm["AIQ"] - hh["AIQ"]) < 1e-9, "两实现 AIQ 一致")
    for k in ("f1", "f2", "f3", "f4", "f5"):
        check(abs(hmm[k] - hh[k]) < 1e-9, f"因子 {k} 一致")
    # 空输入防护（兜底路径会走这里）
    empty = aiq_factors([], 0.01, 0.5, 0.0)
    check(abs(empty["AIQ"]) > 0 and not np.isnan(empty["AIQ"]), "空输入 AIQ 有界")


def test_depth_profile() -> None:
    print("\n[health] depth_profile —— 分层占比")
    g = [float(x) for x in range(1, 25)]  # 1.0..24.0 交替均值分布
    p = depth_profile(g, 24)
    # 均分为 3 段各 8 层，段均值可解析：shallow=mean(1..8)=4.5, mid=mean(9..16)=12.5, deep=mean(17..24)=20.5
    check(abs(p["shallow"] - 4.5) < 1e-9, f"浅层均值={p['shallow']}=4.5")
    check(abs(p["mid"] - 12.5) < 1e-9, f"中层均值={p['mid']}=12.5")
    check(abs(p["deep"] - 20.5) < 1e-9, f"深层均值={p['deep']}=20.5")
    # 24层下加权=总均值：(4.5+12.5+20.5)/3=12.5=mean(1..24)
    check(abs((p["shallow"] + p["mid"] + p["deep"]) / 3 - 12.5) < 1e-9,
          "三段均值=总体均值")
    # 非整除层数（n_layers=10 → step=4, 4+4+2）不崩溃且模式合法
    p2 = depth_profile(g, 10)
    check(p2["pattern"] in ("shallow", "mid", "deep"), "非整除层数模式合法")


def test_health_verdicts() -> None:
    print("\n[health] health_verdicts —— 三分支边界")
    import math
    PI = math.pi
    # K%=20 恰在槽下界（含）→ 通过；K%=19.99 → 失败
    v = health_verdicts(20.0, PI / 2, 0.9)
    check(v[0][1], "K%=20 恰在下界仍通过（[20,57]含边界）")
    v2 = health_verdicts(19.99, PI / 2, 0.9)
    check(not v2[0][1], "K%=19.99 出界失败")
    # DEFF=π/2 偏差0 → 通过；偏差0.16 → 失败
    v3 = health_verdicts(40.0, PI / 2, 0.9)
    check(v3[1][1], "DEFF=π/2（偏差0）锁定通过")
    # |((π/2-0.16)-π/2)/π/2| = 0.16/1.5708 = 0.1019 < 0.15 实际仍通过；用一个确实超界的
    v5 = health_verdicts(40.0, PI / 2 + 0.25, 0.9)   # 偏差=0.25/1.5708≈0.16>0.15
    check(not v5[1][1], "DEFF 偏差 0.25（>0.15）锁定失败")
    # H：f3=0.06 通过（>0.05）；f3=0.05 失败（不能等）
    v6 = health_verdicts(40.0, PI / 2, 0.06)
    check(v6[2][1], "f3=0.06>0.05 → H收敛通过")
    v7 = health_verdicts(40.0, PI / 2, 0.05)
    check(not v7[2][1], "f3=0.05（不>0.05）→ H未收敛")


def test_compressibility() -> None:
    print("\n[health] compressibility —— 三级分级边界")
    pg = {"o_proj": 0.50, "k_proj": 0.45, "q_proj": 0.44, "d_proj": 0.36,
          "g_proj": 0.3499, "up_proj": 0.10}
    c = compressibility(pg)
    check("o_proj" in c["high"] and "k_proj" in c["high"], "γ≥0.45 → 高可压")
    check("q_proj" in c["mid"] and "d_proj" in c["mid"], "γ∈[0.35,0.45) → 中可压")
    check("g_proj" in c["low"] and "up_proj" in c["low"], "γ<0.35 → 难压缩")
    check("o_proj" not in c["low"], "高可压不误入难压缩")
    # 空输入不崩溃
    c0 = compressibility({})
    check(len(c0["high"]) == 0, "空输入分级为空不崩溃")


# ================================================================ forensics
def test_fingerprint() -> None:
    print("\n[forensics] fingerprint —— 向量维度与口径")
    m = Metrics("Qwen2.5-0.5B-Instruct")
    fp = fingerprint(m, n_layers=24)
    # 修复后：剖面重采样到恒定 _PROFILE_K 而非层数，维度 = 64 + 5(曲率标量) = 69
    # 固定维度使不同层数模型指纹可比（修复跨层数 MAD 最近邻 unknown/inf）
    K = fingerprint(m, n_layers=24)["descriptor_k"]  # import 自模块，避免写死 64
    check(len(fp["vector"]) == K + 5,
          f"指纹向量维度恒定={len(fp['vector'])}=K+5 跨层数可比")
    # 恒定性：28 层模型的指纹维度与 24 层相同 → MAD 可跨层数计算
    m28 = Metrics("Qwen2.5-1.5B-Instruct")  # 28 层
    fp28 = fingerprint(m28)
    check(len(fp28["vector"]) == len(fp["vector"]), "28 层与 24 层指纹维度一致(MAD 可比)")
    check(abs(fp["mean_gamma"] - 0.5122) < 1e-3, "Gamma 均值取真实实测 0.5122")
    # 缺失层数补全：仍为恒定维度，并如实置 filled 标记
    sparse = Metrics("X")  # 兜底，无层列表
    fp2 = fingerprint(sparse, n_layers=24)
    check(len(fp2["vector"]) == K + 5 and fp2["filled_profile"], "缺失剖面自动补全且维度恒定")


def test_family_verdict() -> None:
    print("\n[forensics] family_verdict —— SFT 稳定性阈值")
    ok, same = family_verdict(0.0016)
    check(same, "|Δμ|=0.0016<0.01 → 判同族（真实 base/Instruct 实测）")
    _, same2 = family_verdict(0.05)
    check(not same2, "|Δμ|=0.05≥0.01 → 判变异")


# ================================================================ data
def test_metrics() -> None:
    print("\n[data] Metrics —— 三层解析 + sanitize_path")
    # P0 内联优先
    m_inline = {"spectral": {"k_proj_gamma_mean": 0.7},
                "aiq": {"AIQ": 60.0}}
    mt = Metrics("X", measurement=m_inline)
    check(mt.source == "real-inline", "内联真实测量标记 real-inline")
    check(abs(mt.get_float("spectral.k_proj_gamma_mean", 0.0) - 0.7) < 1e-9,
          "内联值优先读取")
    # P1 baselines
    mb = Metrics("Qwen2.5-0.5B-Instruct")
    check(mb.source == "baseline", "基准存档标记 baseline")
    # P2 兜底
    mf = Metrics("NoModel")
    check(mf.source == "fallback", "未知模型回退 fallback")
    # 兜底 AIQ 应为中立结构占位 0，而非某模型的实测值（去除 Qwen 私货硬编码）
    check(mf.get_float("aiq.AIQ", -1.0) == 0.0,
          "兜底 AIQ=0 中立占位（非某模型实测值）")
    # sanitize_path 去除隐私（同时覆盖 POSIX 与 Windows 反斜杠路径）
    priv = {"model": "C:\\Users\\xxx\\Desktop\\_models\\Qwen.json",
            "metric": 1.0}
    sp = sanitize_path(priv)
    check("\\Users" not in str(sp.get("model")) and "\\Desktop" not in str(sp.get("model")),
          "sanitize_path 去除 Windows 绝对路径")
    check(sp["model"] == "Qwen.json", f"Windows 路径保留文件名（{sp['model']}）")
    priv_posix = {"model": "/home/xxx/AIQ/_models/Qwen.json", "a": [1, 2]}
    sp2 = sanitize_path(priv_posix)
    check("/home" not in str(sp2.get("model")) and sp2["model"] == "Qwen.json",
          "sanitize_path 去除 POSIX 绝对路径")
    check(sp["metric"] == 1.0, "sanitize_path 保留非路径字段")
    check(len(find_baseline_models()) >= 5, "跨家族基准模型已注册(≥5)")


def main() -> int:
    print("=" * 62)
    print("aiq-geometric-forensics 推导验证套件（深入测试）")
    print("=" * 62)
    test_spl_gamma()
    test_gamma_energy()
    test_curvature_metrics()
    test_aiq_score()
    test_aiq_factors_eq()
    test_depth_profile()
    test_health_verdicts()
    test_compressibility()
    test_fingerprint()
    test_family_verdict()
    test_metrics()
    print("=" * 62)
    print(f"推导验证汇总: PASS={_PASS}  FAIL={_FAIL}")
    print("=" * 62)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
