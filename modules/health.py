"""
aiq-geometric-forensics.modules.health — 几何健康度诊断
=====================================================================
对给定模型（或真实测量）输出单向度"几何健康画像"：
  * AIQ 五因子综合商（0-100）
  * Gamma 深度剖面 + 分层（浅/中/深层集中度分布）
  * 曲率身份（DEFF 平台 / K<0% / Hmed / φmean / λ）
  * 退化-过深-短链未收敛三大健康判定
  * 可压缩性评估（Gamma 谱 → KV 压缩建议）

典型调用：
  import modules.health as H
  rep = H.diagnose(name="Qwen2.5-0.5B-Instruct")     # 基于基准存档
  rep  -> 诊断数据字典（含全部指标与判定）
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import schema
from .data import Metrics, find_baseline_models

# 判据常量统一取自 schema 单一事实源（避免分散魔法数漂移）
PI_HALF = schema.PI_HALF                    # DEFF 平台理论锚点 π/2（theory）
DEFF_TOL = schema.DEFF_TOL                  # 平台锁定容差（heuristic）
K_SLOT = schema.K_SLOT                      # K<0% 健康槽区间（heuristic）
H_CONV_THR = schema.H_CONV_THR              # H 收敛判定阈值（f3 下界，heuristic）
WEIGHTS = schema.WEIGHTS                    # AIQ 五因子权重（design）
# 可压缩性分级阈值 + 配套剪枝建议（统一取自 schema，避免散落硬编码）
_COMPRESS_HIGH = schema.COMPRESS_HIGH
_COMPRESS_MID = schema.COMPRESS_MID
_COMPRESS_LOW = schema.COMPRESS_LOW
_RATIO_HIGH = schema.COMPRESS_HIGH_RATIO    # 高可压 keep_ratio 建议区间
_RATIO_MID = schema.COMPRESS_MID_RATIO      # 中可压 keep_ratio 建议区间
_RATIO_LOW = schema.COMPRESS_LOW_RATIO      # 难压缩 keep_ratio 建议区间


# ---------------------------------------------------------------- AIQ 五因子
def aiq_factors(gamma_layers: list[float], deff_cv: float,
                h_median: float, spl: float) -> dict[str, Any]:
    """计算 AIQ 五因子及其加权综合商。

    f1 芯坍缩   = mean(Gamma)                 -> 谱集中均值（k 投影）
    f2 DEFF稳定 = 1 - DEFF 变异系数            -> 平台稳定性
    f3 H收敛    = 1/(1+Hmed)                   -> 曲率尺度收敛指数
    f4 SPL集中  = 全投影谱集中均值             -> 跨 token 全投影集中（独立于 f1）
    f5 低模纯度 = 1 - (max-min)Gamma           -> 深度均匀性
    AIQ = 100 · Σ(w_i · f_i)

    说明：
      f3 修复：旧 `1 - min(1, Hmed)` 对真实模型 Hmed∈[1.24,1.38] 恒为 0、无区分度。
          改为有界单调衰减 `1/(1+Hmed)`（Hmed=0→1，Hmed→∞→0），天然 ∈[0,1]、无阈值饱和。
      f4 修复：旧调用把 spl 传成 f1(伽马均值)，两因子重复计数。现 f4 由调用方传
          入「全投影谱集中均值」（kT/q/v/o/gate/up/down 各投影 spl_gamma 的均值），
          是与 f1（k 投影单投影）不同的实测聚合对象 → 真正独立。缺数据时回退 f1。
    """
    f1 = float(np.mean(gamma_layers)) if gamma_layers else 0.0
    f2 = 1.0 - float(deff_cv)
    f3 = 1.0 / (1.0 + max(float(h_median), 0.0))
    f4 = float(spl) if gamma_layers else f1
    f5 = (1.0 - (float(np.max(gamma_layers)) - float(np.min(gamma_layers)))
          if gamma_layers else 1.0)
    aiq = 100.0 * (WEIGHTS[0] * f1 + WEIGHTS[1] * f2 + WEIGHTS[2] * f3
                   + WEIGHTS[3] * f4 + WEIGHTS[4] * f5)
    return {"AIQ": float(aiq), "f1": f1, "f2": f2, "f3": f3, "f4": f4,
            "f5": f5, "weights": list(WEIGHTS)}


# ---------------------------------------------------------------- 深度剖面
def depth_profile(gamma_layers: list[float],
                  n_layers: int = 24) -> dict[str, Any]:
    """按深度分三层（浅/中/深，每层 1/3）统计谱集中度分布。"""
    g = np.asarray(gamma_layers, dtype=float)
    if g.size == 0:
        return {"shallow": 0.0, "mid": 0.0, "deep": 0.0, "pattern": "unknown",
                "mean": 0.0, "min": 0.0, "max": 0.0}
    # 用 array_split 严格等分三层，保证浅/中/深层数差 ≤1（修复 ceil(step)
    # 对非 3 倍层数分组不均：旧法 22 层会分成 8/8/6、28 层 10/10/8）
    parts = np.array_split(g, 3)
    shallow = float(parts[0].mean())
    mid = float(parts[1].mean())
    deep = float(parts[2].mean())
    pattern = "shallow" if (shallow >= mid and shallow >= deep) else \
              "mid" if mid >= deep else "deep"
    return {"shallow": shallow, "mid": mid, "deep": deep, "pattern": pattern,
            "mean": float(g.mean()), "min": float(g.min()), "max": float(g.max())}


# ---------------------------------------------------------------- 健康判定
def calibrate_deff_band() -> tuple[float, float]:
    """从基准家族实时标定 DEFF 平台带 [lo, hi]（区间+误差带口径）。

    用 `baselines/` 全部模型的实际 DEFF_plat 观测支撑集作为带宽：
      低=家族最小 DEFF_plat，高=家族最大 DEFF_plat。
    SKILL.md / 使用指南 主张 DEFF 按"区间 + 误差带"表述（非单一锁定 π/2 点值），
    本函数把该口径落到代码：落点落在家族标定带内即视为"平台一致"的旁证，
    同时绝不掩盖对 π/2 的系统性正偏（仍另报相对偏差）。
    少于 2 个模型时回退理论带 [π/2−DEFF_TOL, π/2+DEFF_TOL]。
    """
    vals = [Metrics(n).get_float("curvature.DEFF_plat", 0.0)
            for n in find_baseline_models()]
    vals = sorted(v for v in vals if v > 0.0)
    if len(vals) < 2:
        return (PI_HALF - DEFF_TOL, PI_HALF + DEFF_TOL)
    return (vals[0], vals[-1])


def health_verdicts(k_neg_pct: float, deff_plat: float,
                    f3: float,
                    deff_band: tuple[float, float] | None = None
                    ) -> list[tuple[str, bool, str]]:
    """生成三大健康判定（判据, 是否通过, 说明）。

    deff_band 为可选家族标定带（calibrate_deff_band 产出）；传入时在
    "DEFF平台锁定" 说明后追加"家族标定带内/出带"走向，作为区间口径的公示，
    不改动原 DEFF_TOL 布尔判据（保持既有测试与外部契约稳定）。
    """
    out: list[tuple[str, bool, str]] = []
    k_ok = K_SLOT[0] <= k_neg_pct <= K_SLOT[1]
    out.append(("K%槽内(鞍形主导)", k_ok,
                f"K<0%={k_neg_pct:.2f} 槽[{K_SLOT[0]},{K_SLOT[1]}]"))
    deff_ok = abs(deff_plat - PI_HALF) < DEFF_TOL
    band_txt = ""
    if deff_band and deff_band[0] < deff_band[1]:
        inb = deff_band[0] <= deff_plat <= deff_band[1]
        band_txt = (f"  家族标定带[{deff_band[0]:.4f},{deff_band[1]:.4f}]: "
                    f"{'在带内' if inb else '出带'}")
    out.append(("DEFF平台锁定", deff_ok,
                f"DEFF={deff_plat:.4f} vs π/2偏差 "
                f"{(deff_plat-PI_HALF)/PI_HALF*100:+.2f}%{band_txt}"))
    h_ok = f3 > H_CONV_THR
    out.append(("H收敛(链长)", h_ok,
                f"f3={f3:.3f} 阈值>{H_CONV_THR:.2f}"))
    return out


# ---------------------------------------------------------------- 可压缩性
# 可压缩性分级阈值 + 建议 keep_ratio 已统一取自 schema（见文件头 _COMPRESS_* / _RATIO_*）；
# 其证据级为"推测"（见 schema.EVIDENCE），报告/文档须显式标注，不得当实测引用；
# 待压缩-精度对照实验标定升级为 measured（协议见 docs/可压缩性阈值标定协议.md）。


def compressibility(proj_gamma: dict[str, float]) -> dict[str, Any]:
    """按各投影 Gamma 分级，生成 KV/权重压缩建议。

    分级依据：谱集中度越高 → 冗余越大 → 越可压。
      高可压 o/k（注意力合成侧） -> 激进剪枝 keep_ratio 0.25-0.4
      中可压 q/gate/down          -> 温和压缩 keep_ratio 0.5-0.6
      难压缩 up/v（前馈扩展）     -> 保持全精度
    """
    high, mid, low = {}, {}, {}
    if proj_gamma:
        for k, v in proj_gamma.items():
            try:
                v = float(v)
                if not np.isfinite(v):
                    continue
            except (TypeError, ValueError):
                continue                                   # 无实测值 → 跳过，不崩
            if v >= _COMPRESS_HIGH:
                high[k] = float(v)
            elif v >= _COMPRESS_MID:
                mid[k] = float(v)
            else:
                low[k] = float(v)
    return {
        "high": dict(sorted(high.items(), key=lambda kv: -kv[1])),
        "mid": dict(sorted(mid.items(), key=lambda kv: -kv[1])),
        "low": dict(sorted(low.items(), key=lambda kv: -kv[1])),
        "thresholds": {"high": _COMPRESS_HIGH, "mid": _COMPRESS_MID,
                       "low": _COMPRESS_LOW},
    }


# ---------------------------------------------------------------- 主入口
def diagnose(name: str = "Qwen2.5-0.5B-Instruct",
             measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    """几何健康度诊断 —— 对基准存档或真实测量生成完整健康画像。

    参数：
      name: 模型名（用于定位 baselines/ 基准存档）
      measurement: 显式真实测量 dict（P0 优先，缺省用存档/兜底）
    返回：结构化诊断字典（可直接打印或序列化）。
    """
    m = Metrics(name, measurement)

    gamma_layers = m.get_list("spectral.k_proj_gamma_layers")
    proj_gamma = m.get("spectral.proj_gamma", {}) or {}
    # 测量值实时获取：优先 P0/P1 存档；缺失时用占位（0/LN2），不伪造某模型实测
    deff = m.get_float("curvature.DEFF_plat", PI_HALF)
    k_neg = m.get_float("curvature.K_neg_pct", 0.0)
    h_med = m.get_float("curvature.H_median", 0.0)
    phi = m.get_float("curvature.phi_mean_deg", 0.0)
    lam = m.get_float("curvature.lambda_ratio", 0.0)
    # DEFF 变异系数：优先取真实测量值；缺失时用结构占位 0（绝不用某模型实测值）
    deff_cv = m.get_float("curvature.DEFF_cv", 0.0)

    n_layers = m.arch().get("n_layers", 24)
    profile = depth_profile(gamma_layers, n_layers)

    # ---- AIQ 实时复算（不信任存档旧值，杜绝 f3/f4 退化旧式数据被沿用）----
    # f4 的真实独立 SPL = 全投影谱集中均值（与 f1 的 k 单投影不同源）；
    # 无全投影数据时回退 k 投影均值（并如实标注 placeholder）。
    proj_vals = [float(v) for v in proj_gamma.values()
                 if v is not None and isinstance(v, (int, float))
                 and np.isfinite(v)]
    spl = float(np.mean(proj_vals)) if proj_vals else profile["mean"]
    aiq = aiq_factors(gamma_layers, deff_cv, h_med, spl)

    proj = proj_gamma if proj_gamma else {"k_proj": profile["mean"]}
    deff_band = calibrate_deff_band()
    verdicts = health_verdicts(k_neg, deff, float(aiq.get("f3", 0.0)),
                               deff_band=deff_band)
    compr = compressibility(proj)

    return {
        "model": name,
        "data_source": m.report(),
        "aiq": aiq,
        "gamma": {"mean": float(aiq.get("f1", profile["mean"])),
                  "min": profile["min"], "max": profile["max"],
                  "layers": list(gamma_layers)},
        "depth": profile,
        "curvature": {"DEFF_plat": deff, "K_neg_pct": k_neg,
                      "H_median": h_med, "phi_mean_deg": phi,
                      "lambda_ratio": lam,
                      "deff_dev_pct": (deff - PI_HALF) / PI_HALF * 100,
                      "deff_band": list(deff_band)},
        "health": {"judgements": verdicts,
                   "passed_all": all(ok for _, ok, _ in verdicts)},
        "compressibility": compr,
    }


# ---------------------------------------------------------------- 格式化输出
def format_health(rep: dict[str, Any]) -> str:
    """把健康诊断 dict 渲染成可读文本报告。"""
    aiq = rep["aiq"]
    lines = ["=" * 72, f"几何健康度诊断: {rep['model']}", "=" * 72,
             f"  {rep['data_source']}",
             f"\n  ★ AIQ 几何智能商 = {aiq['AIQ']:.2f} / 100",
             f"     f1 芯坍缩={aiq['f1']:.3f}  f2 DEFF稳定={aiq['f2']:.3f}  "
             f"f3 H收敛={aiq['f3']:.3f}  f4 SPL集中={aiq['f4']:.3f}  "
             f"f5 低模纯度={aiq['f5']:.3f}",
             f"\n  Gamma 深度剖面: mean={rep['gamma']['mean']:.4f} "
             f"(min={rep['gamma']['min']:.4f} max={rep['gamma']['max']:.4f})"]
    d = rep["depth"]
    lines.append(f"  分层: 浅层={d['shallow']:.3f} 中层={d['mid']:.3f} "
                 f"深层={d['deep']:.3f} → {'浅层最高' if d['pattern']=='shallow' else d['pattern']}")
    c = rep["curvature"]
    lines.append(f"\n  曲率身份: DEFF={c['DEFF_plat']:.4f} "
                 f"(偏差 {c['deff_dev_pct']:.2f}%) | K<0%={c['K_neg_pct']:.2f}% "
                 f"| Hmed={c['H_median']:.4f} | φmean={c['phi_mean_deg']:.2f}° "
                 f"| λ={c['lambda_ratio']:.4f}")
    lines.append("\n  # 健康判定:")
    for tag, ok, detail in rep["health"]["judgements"]:
        mark = "✓" if ok else "✗"
        lines.append(f"    [{mark}] {tag}: {detail}")

    compr = rep["compressibility"]
    if compr["high"] or compr["mid"] or compr["low"]:
        def fmt(dct):
            return ", ".join(f"{k}({v:.3f})" for k, v in dct.items())
        # 建议档位取自 schema（_RATIO_*），集中管理而非散落魔法数
        lines.append("\n  # 可压缩性评估:")
        lines.append(f"    ★ 高可压(γ≥{compr['thresholds']['high']}): "
                     f"{fmt(compr['high']) or '无'} → "
                     f"激进剪枝 keep_ratio {_RATIO_HIGH[0]:.2f}-{_RATIO_HIGH[1]:.2f}")
        lines.append(f"    ○ 中可压: {fmt(compr['mid']) or '无'} → "
                     f"温和压缩 keep_ratio {_RATIO_MID[0]:.2f}-{_RATIO_MID[1]:.2f}")
        lines.append(f"    ✗ 难压缩(γ<{compr['thresholds']['low']}): "
                     f"{fmt(compr['low']) or '无'} → 保持全精度 keep_ratio "
                     f"{_RATIO_LOW[0]:.2f}·驱逐最后动")
    return "\n".join(lines)


# ---------------------------------------------------------------- 直接运行
if __name__ == "__main__":
    print(format_health(diagnose()))
