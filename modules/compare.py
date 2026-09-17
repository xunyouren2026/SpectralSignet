"""
aiq-geometric-forensics.modules.compare — 与基准模型对比
=====================================================================
对待检模型 vs 已标定基准库做逐维几何 + 工程偏差对比：
  * 谱集中（k_proj Gamma）指纹差
  * 几何智能商（AIQ）差
  * 工程吞吐（tok/s）、KV 内存等
  * 家族归属判定（同架构同指纹 → 判定为基准同族）

参数：
  compare.compare(target, reference, target_measurement=..., reference_measurement=...)
"""
from __future__ import annotations

from typing import Any

from .data import Metrics, sanitize_path


# ---------------------------------------------------------------- 主入口
def compare(target: str = "Qwen2.5-0.5B",
            reference: str = "Qwen2.5-0.5B-Instruct",
            target_measurement: dict[str, Any] | None = None,
            reference_measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    """对待检模型输出 vs 基准库输出的偏差报告。"""
    tgt = Metrics(target, target_measurement)
    ref = Metrics(reference, reference_measurement)

    g_t = tgt.get_float("spectral.k_proj_gamma_mean", 0.0)
    g_r = ref.get_float("spectral.k_proj_gamma_mean", 0.0)
    aiq_t = tgt.get_float("aiq.AIQ", 0.0)
    aiq_r = ref.get_float("aiq.AIQ", 0.0)
    t_t = tgt.get_float("engine.tok_s", 0.0)
    t_r = ref.get_float("engine.tok_s", 0.0)
    kv_t = tgt.get_float("engine.kv_bytes", 0.0)
    kv_r = ref.get_float("engine.kv_bytes", 0.0)

    arch_t = tgt.arch()
    arch_r = ref.arch()
    same_dims = (arch_t.get("n_layers") == arch_r.get("n_layers")
                 and arch_t.get("hidden") == arch_r.get("hidden"))
    same_fam = same_dims and (abs(g_t - g_r) < 0.01)

    return {
        "target": target, "reference": reference,
        "metric": {"gamma_target": g_t, "gamma_ref": g_r, "gamma_diff": g_t - g_r,
                   "aiq_target": aiq_t, "aiq_ref": aiq_r, "aiq_diff": aiq_t - aiq_r,
                   "tok_s_target": t_t, "tok_s_ref": t_r, "tok_s_diff": t_t - t_r,
                   "kv_bytes_target": kv_t, "kv_bytes_ref": kv_r,
                   "kv_bytes_diff": kv_t - kv_r},
        "arch_target": sanitize_path(arch_t),
        "arch_ref": sanitize_path(arch_r),
        "same_architecture": same_dims,
        "same_family": same_fam,
        "verdict": (f"与基准 {reference} 同架构同指纹 → "
                    f"判定为该家族成员") if same_fam else \
                   "指纹/架构存在差异 → 需人工复核家族归属",
        "data_source_target": tgt.report(),
        "data_source_ref": ref.report(),
    }


# ---------------------------------------------------------------- 格式化输出
def format_compare(rep: dict[str, Any]) -> str:
    """把对比报告渲染成对齐表格。"""
    m = rep["metric"]
    lines = ["=" * 72, "与基准模型对比", "=" * 72,
             f"  待检: {rep['target']}  ({rep['data_source_target']})",
             f"  基准: {rep['reference']}  ({rep['data_source_ref']})",
             "",
             f"{'指标':<18}{'基准':>14}{'待检':>14}{'偏差':>12}",
             "-" * 58,
             f"{'k_proj Gamma':<18}{m['gamma_ref']:>14.4f}{m['gamma_target']:>14.4f}"
             f"{m['gamma_diff']:>+12.4f}",
             f"{'AIQ 几何智能商':<18}{m['aiq_ref']:>14.2f}{m['aiq_target']:>14.2f}"
             f"{m['aiq_diff']:>+12.2f}",
             f"{'tok/s':<18}{m['tok_s_ref']:>14.2f}{m['tok_s_target']:>14.2f}"
             f"{m['tok_s_diff']:>+12.2f}",
             f"{'KV(bytes)':<18}{m['kv_bytes_ref']:>14.0f}{m['kv_bytes_target']:>14.0f}"
             f"{m['kv_bytes_diff']:>+12.0f}",
             "",
             f"  同架构: {'是' if rep['same_architecture'] else '否'}   "
             f"同家族: {'是' if rep['same_family'] else '否'}",
             f"  判定: {rep['verdict']}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- 直接运行
if __name__ == "__main__":
    print(format_compare(compare()))
