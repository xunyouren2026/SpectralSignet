"""
aiq-geometric-forensics.modules.forensics — 家族溯源判定
=====================================================================
基于几何指纹对模型做"家族身份鉴定"：
  * 指纹向量构建：Gamma 深度剖面（24 维）⊎ 曲率身份（K<0% / DEFF / Hmed / λ 等）
  * 家族判定：待检指纹 vs 基准库指纹的 MAD 最近邻
  * SFT 稳定性：base vs Instruct 指纹差极小 → 微调不改身份（溯源的根基）
  * 边距（margin）：可靠度度量，边距越大越可信

典型调用：
  import modules.forensics as F
  rep = F.trace(target="Qwen2.5-0.5B", reference="Qwen2.5-0.5B-Instruct")
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import schema
from .data import Metrics

# 判定阈值统一取自 schema 单一事实源
FTC_EPS = schema.FTC_EPS                    # 指纹均值差 < 0.01 → 同族（SFT 不改身份）
MARGIN_EPS = schema.MARGIN_EPS              # 边距判定阈值：可靠度门控
# 无真实层谱时的兜底均值（仅标注用；不从某个模型硬编码）
_FALLBACK_MEAN = 0.0
# 指纹剖面统一重采样点数：保证不同层数模型的指纹向量维度一致，从而可跨家族可比。
# 修复前 24 层(29 维) vs 22 层(27 维) 向量维度不同 → MAD 最近邻恒 unknown/inf。
_PROFILE_K = schema.PROFILE_K


def _interp_profile(profile: np.ndarray, k: int = _PROFILE_K) -> np.ndarray:
    """把逐层 Gamma 剖面按相对深度线性重采样到固定 k 点。

    使得不同层数（如 Llama 22 / Qwen 24 / 1.5B 28）的模型在
    指纹向量上拥有相同维度，MAD 最近邻才能跨家族计算。
    """
    n = profile.size
    if n <= 1:
        return np.full(k, float(profile[0]) if n else 0.0)
    xp = np.linspace(0.0, 1.0, n)
    xq = np.linspace(0.0, 1.0, k)
    return np.interp(xq, xp, profile)


# ---------------------------------------------------------------- 指纹向量
def fingerprint(m: Metrics, n_layers: int | None = None) -> dict[str, Any]:
    """构建单模型几何指纹（数值向量 + 语义字段）。

    指纹 = [Gamma 深度剖面(N)] ⊎ [曲率身份标量]。
    n_layers 从模型 arch 实时读取（缺省探测），而非硬编码 24；
    剖面缺失时用 Gamma 均值填充（如实标注 fill）。
    """
    if n_layers is None:
        n_layers = int(m.arch().get("n_layers", 24) or 24)
    layers = m.get_list("spectral.k_proj_gamma_layers")
    filled = not layers
    if filled:
        mean = m.get_float("spectral.k_proj_gamma_mean", _FALLBACK_MEAN)
        layers = [mean] * n_layers
    layers = (layers + [m.get_float("spectral.k_proj_gamma_mean", layers[0])]
              * (n_layers - len(layers)))[:n_layers]
    profile = np.asarray(layers, dtype=float)

    scalar_keys = ("curvature.K_neg_pct", "curvature.DEFF_plat",
                   "curvature.H_median", "curvature.phi_mean_deg",
                   "curvature.lambda_ratio")
    scalar = np.array([m.get_float(k, 0.0) for k in scalar_keys], dtype=float)

    # 指纹向量 = [Gamma 剖面(重采样到 _PROFILE_K 维)] ⊎ [曲率标量(5 维)]
    # 维度恒为 _PROFILE_K + 5，跨层数模型可比（修复 MAD 最近邻失效 BUG）
    descriptor = _interp_profile(profile, _PROFILE_K)
    vector = np.concatenate([descriptor, scalar]).astype(np.float32)
    return {"vector": vector, "profile": profile, "descriptor_k": _PROFILE_K,
            "scalar": {k.rsplit(".", 1)[-1]: m.get_float(k, 0.0)
                       for k in scalar_keys},
            "mean_gamma": float(profile.mean()), "filled_profile": filled}


def _mad_neighbor(query: np.ndarray, refs: dict[str, np.ndarray]) -> tuple[str, float]:
    """最小 MAD 最近邻：返回 (最近族, MAD 距离)。"""
    if not refs:
        return "unknown", float("inf")
    best, best_d = "unknown", float("inf")
    for fam, vec in refs.items():
        if vec.shape != query.shape:
            continue
        d = float(np.mean(np.abs(query - vec)))           # 平均绝对偏差 MAD
        if d < best_d:
            best, best_d = fam, d
    return best, best_d


def family_verdict(d_mean: float) -> tuple[str, bool]:
    """由指纹均值差给出三档家族判定（修复：让 MARGIN_EPS 真正参与判据）。

      |Δμ| < MARGIN_EPS       : 指纹几乎不变 → 同家族（SFT 微调不改身份）
      MARGIN_EPS ≤ |Δμ| < FTC : 指纹变异 → 微调/近族差异可被捕获
      |Δμ| ≥ FTC_FAR           : 明显异族 → 低可靠度同族
    """
    FTC_FAR = MARGIN_EPS
    if d_mean < FTC_EPS:
        return ("指纹几乎不变(|Δμ| 极小) → SFT 微调不改身份 → "
                "'几何指纹是身份度量，不是智能度量'", True)
    if d_mean >= FTC_FAR:
        return (f"指纹距参照显著({d_mean:.3f} ≥ {FTC_FAR:.2f}) → "
                f"明显异族/远亲，同族可靠度低", False)
    return ("指纹有显著变化 → 微调/家族差异可被指纹捕获", False)


# ---------------------------------------------------------------- 主入口
def trace(target: str = "Qwen2.5-0.5B",
          reference: str = "Qwen2.5-0.5B-Instruct",
          target_measurement: dict[str, Any] | None = None,
          reference_measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    """家族溯源判定 —— 待检模型 vs 参照模型（或基准库）。

    参数：
      target: 待检模型名
      reference: 参照/基准模型名（多家庭的基准库可后续扩展）
    返回：溯源报告（指纹均值差 + MAD 最近邻 + SFT 稳定性 + 判定）。
    """
    tgt = Metrics(target, target_measurement)
    ref = Metrics(reference, reference_measurement)

    f_tgt = fingerprint(tgt)
    f_ref = fingerprint(ref)
    d_mean = float(abs(f_tgt["mean_gamma"] - f_ref["mean_gamma"]))

    verdict, same = family_verdict(d_mean)
    # MAD 最近邻：当前单参照场景，多家庭库可扩展为 {fam: vector} 传入
    refs = {reference: f_ref["vector"]}
    nearest, mad = _mad_neighbor(f_tgt["vector"], refs)
    margin = float(abs(f_tgt["mean_gamma"] - f_ref["mean_gamma"]))

    return {
        "target": target, "reference": reference,
        "d_gamma_mean": d_mean,
        "gamma_target": f_tgt["mean_gamma"], "gamma_ref": f_ref["mean_gamma"],
        "fingerprint_target_filled": f_tgt["filled_profile"],
        "fingerprint_ref_filled": f_ref["filled_profile"],
        "nearest_family": nearest, "mad_distance": mad, "margin": margin,
        "same_family": same, "verdict": verdict,
        "data_source_target": tgt.report(),
        "data_source_ref": ref.report(),
    }


# ---------------------------------------------------------------- 格式化输出
def format_trace(rep: dict[str, Any]) -> str:
    """把溯源报告渲染成可读文本。"""
    lines = ["=" * 72, "家族溯源判定（几何指纹）", "=" * 72,
             f"  待检: {rep['target']}  ({rep['data_source_target']})",
             f"  参照: {rep['reference']}  ({rep['data_source_ref']})",
             f"\n  γ_target={rep['gamma_target']:.4f}  γ_ref={rep['gamma_ref']:.4f}",
             f"  指纹均值差 |Δμ| = {rep['d_gamma_mean']:.4f}",
             f"  MAD 最近邻 → {rep['nearest_family']}  (MAD={rep['mad_distance']:.4f})",
             f"  SFT 稳定性: {'指纹稳定(微调不改身份)' if rep['same_family'] else '指纹变异'}",
             f"  判定: {rep['verdict']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- 直接运行
if __name__ == "__main__":
    print(format_trace(trace()))
