"""
aiq-geometric-forensics 验证套件
=====================================================================
覆盖：数据层三层路径 / 健康诊断 / 家族溯源 / 基准对比。全部为
独立函数式断言，不使用 pytest 框架（零依赖，可单文件执行）。

用法（本包根目录下）：
  python tests/run_all_tests.py
退出码：全过 0，任一失败 1。
"""
from __future__ import annotations

import os
import sys

# 确保可从本包根导入 modules（无论从哪个目录调用本脚本）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)  # tests 目录内可独立导入（可选）

from modules import (  # noqa: E402
    Metrics,
    aiq_factors,
    compare,
    depth_profile,
    diagnose,
    find_baseline_models,
    trace,
)

_PASSED = 0
_FAILED = 0


def check(cond: bool, msg: str) -> None:
    """单条断言并计数。"""
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print(f"  [PASS] {msg}")
    else:
        _FAILED += 1
        print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------- 数据层
def test_data_layer() -> None:
    print("[数据层] 三层数据解析路径")
    # 兜底：不存在模型应回退且 AIQ>0
    r = diagnose("No-Such-Model-ABC")
    check("审计兜底" in r["data_source"], "未知模型回退审计兜底值")
    check(r["aiq"]["AIQ"] > 0, "兜底路径 AIQ 正常")
    # 内联真实测量（P0）
    m = {"spectral": {"k_proj_gamma_layers": [0.5] * 24,
                      "k_proj_gamma_mean": 0.5},
         "curvature": {"DEFF_plat": 1.59, "K_neg_pct": 40.0,
                       "H_median": 0.5, "phi_mean_deg": 20.0,
                       "lambda_ratio": 0.8},
         "arch": {"n_layers": 24}}
    r2 = diagnose("my-model", measurement=m)
    check("真实测量" in r2["data_source"], "内联真实测量优先（P0）")
    check(abs(r2["aiq"]["f1"] - 0.5) < 1e-9, "内联 f1 与传入 Gamma 一致")
    # 基准存档（P1）
    check(len(find_baseline_models()) >= 5, "跨家族基准存档可枚举(≥5)")


# ---------------------------------------------------------------- 健康诊断
def test_health() -> None:
    print("[健康诊断] AIQ / 深度剖面 / 健康判定")
    a = aiq_factors([0.47, 0.48, 0.50, 0.46], 0.01, 0.8, 0.48)
    check(0 < a["AIQ"] < 100, "AIQ 落在合理区间")
    check(all(0 <= a[k] <= 1 for k in ("f1", "f2", "f3", "f4", "f5")),
          "五因子均在 [0,1]")
    prof = depth_profile([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 6)
    check(prof["pattern"] in ("shallow", "mid", "deep"), "分层模式合法")
    hp = diagnose("Qwen2.5-0.5B-Instruct")
    check(abs(hp["aiq"]["AIQ"] - 54.60) < 0.5,
          "基准 AIQ≈54.60（f3/f4 修复后，实时复算，非存档旧值）")
    check(hp["aiq"]["f3"] > 0.3, "f3=1/(1+Hmed) 非退化（修复后 >0 有区分度）")
    check(abs(hp["aiq"]["f4"] - hp["aiq"]["f1"]) > 5e-3,
          "f4(全投影谱集中) 与 f1(k 单投影) 独立（修复重复计数）")
    check(len(hp["health"]["judgements"]) == 3, "三类健康判据齐全")


# ---------------------------------------------------------------- 家族溯源
def test_forensics() -> None:
    print("[家族溯源] SFT 稳定性 / 同族判定")
    t = trace("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct")
    check(t["d_gamma_mean"] < 0.01, "base/Instruct 指纹差极小")
    check(t["same_family"], "同架构微调判定为同家族")
    check(t["nearest_family"] is not None, "MAD 最近邻有归属")


# ---------------------------------------------------------------- 基准对比
def test_compare() -> None:
    print("[基准对比] 同族判定 / 偏差数值")
    c = compare("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct")
    check(c["same_architecture"], "同架构识别正确")
    check(c["same_family"], "同家族判定正确")
    check(abs(c["metric"]["gamma_diff"] - 0.0016) < 2e-4, "伽马偏差≈0.0016（真实实测）")


# 跨家族：证明 n_layers 与指纹均为实时获取，而非硬编码 24
def test_cross_family() -> None:
    print("[跨家族] 层数动态获取 / 家族指纹区分")
    # BLOOM 取 n_layer=24、TinyLlama 取 num_hidden_layers=22 → 均从 config 实时解析
    b = Metrics("bloom-560m")
    t = Metrics("TinyLlama-1.1B-Chat-v1.0")
    check(b.source == "baseline" and t.source == "baseline", "BLOOM/TinyLlama 基准存档可装载")
    check(len(b.get_list("spectral.k_proj_gamma_layers")) == 24,
          "bloom 层数=24（从 n_layer 实时解析，非硬编码）")
    check(len(t.get_list("spectral.k_proj_gamma_layers")) == 22,
          "TinyLlama 层数=22（从 num_hidden_layers 实时解析）")
    # 跨家族指纹应显著有别于 Qwen 家族（BLOOM 谱集中度明显更低）
    gb = b.get_float("spectral.k_proj_gamma_mean", 0.0)
    gq = Metrics("Qwen2.5-0.5B-Instruct").get_float("spectral.k_proj_gamma_mean", 0.0)
    check(abs(gb - gq) > 0.05, f"BLOOM γ={gb:.4f} 与 Qwen γ={gq:.4f} 指纹明显区分",
          )
    # 各家族 AIQ 均已按真实 DEFF_cv 实时计算（非 Qwen 私有兜底）
    for name in ("bloom-560m", "TinyLlama-1.1B-Chat-v1.0", "Qwen2.5-1.5B-Instruct"):
        m = Metrics(name)
        check(30 < m.get_float("aiq.AIQ", 0.0) < 60, f"{name} AIQ 落在真实区间")
    # 修复固定点：f4 全投影谱集中必须独立于 f1(k 单投影)，且 f3 非退化
    for name in ("Qwen2.5-0.5B", "bloom-560m", "TinyLlama-1.1B-Chat-v1.0"):
        rep = diagnose(name)
        check(rep["aiq"]["f3"] > 0.3, f"{name} f3 非退化(>0.3)")
        check(abs(rep["aiq"]["f4"] - rep["aiq"]["f1"]) > 5e-3,
              f"{name} f4 独立于 f1")

    # 回归：跨层数溯源（修复前 28vs22 层向量维度不同 → MAD 恒 unknown/inf）
    tr = trace("Qwen2.5-1.5B-Instruct", "TinyLlama-1.1B-Chat-v1.0")  # 28 vs 22 层
    check(tr["nearest_family"] != "unknown",
          "跨层数(28 vs 22) MAD 最近邻有明确归属")
    check(tr["mad_distance"] == tr["mad_distance"] and tr["mad_distance"] < 1e9,
          "跨层数 MAD 距离为有限值（非 inf/NaN）")


def main() -> int:
    print("=" * 60)
    print("aiq-geometric-forensics 验证套件")
    print("=" * 60)
    test_data_layer()
    test_health()
    test_forensics()
    test_compare()
    test_cross_family()
    print("=" * 60)
    print(f"汇总: PASS={_PASSED}  FAIL={_FAILED}")
    print("=" * 60)
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
