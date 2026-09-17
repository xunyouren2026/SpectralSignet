# -*- coding: utf-8 -*-
"""
aiq-geometric-forensics 演示驱动 — 六大真实用途一键运行
=====================================================================
通过包公开 API（modules 相对导入）运行全部 6 大用例：

  案例一  家族溯源 / SFT 指纹对比（base vs Instruct）
  案例二  几何健康度诊断（AIQ + Gamma + 曲率身份）
  案例三  可压缩性评估（Gamma 谱 → KV 压缩建议）
  案例四  与基准模型对比（待检 base vs 基准 Instruct）
  案例五  溯源鲁棒性（扰动下指纹存活，SFT/截断/缩放）
  案例六  家族族谱（跨模型指纹亲缘聚类）

用法（在本包根目录下运行）：
  python aiq_forensics_demo.py
无需大模型/网络——使用 baselines/ 真实实测存档；缺失自动回退兜底值。
"""
import sys

from modules import (diagnose, format_health, trace, format_trace,
                     compare, format_compare, stability, format_stability,
                     family_tree, format_family)


def main() -> int:
    # ---- 案例一：家族溯源 ----
    print(format_trace(trace(target="Qwen2.5-0.5B",
                             reference="Qwen2.5-0.5B-Instruct")))
    print()
    # ---- 案例二：健康度诊断 ----
    print(format_health(diagnose("Qwen2.5-0.5B-Instruct")))
    print()
    # ---- 案例三：可压缩性（含在健康报告中，单独展示达标部分）----
    rep = diagnose("Qwen2.5-0.5B-Instruct")
    c = rep["compressibility"]
    def fmt(dct):
        return ", ".join(f"{k}({v:.3f})" for k, v in dct.items())
    print("=" * 72)
    print("案例三：可压缩性评估（Gamma 谱 → KV 压缩建议）")
    print("=" * 72)
    print(f"  ★ 高可压(γ≥{c['thresholds']['high']}): {fmt(c['high']) or '无'} "
          f"→ 激进剪枝 keep_ratio 0.25-0.4")
    print(f"  ○ 中可压: {fmt(c['mid']) or '无'} → 温和压缩 keep_ratio 0.5-0.6")
    print(f"  ✗ 难压缩(γ<{c['thresholds']['low']}): {fmt(c['low']) or '无'} "
          f"→ 保持全精度. 驱逐最后动")
    print(f"\n  KV 工程与压缩建议详见健康报告（对 o/k 剪枝可削减 KV 内存 40-50%）。")
    print()
    # ---- 案例四：基准对比 ----
    print(format_compare(compare(target="Qwen2.5-0.5B",
                                 reference="Qwen2.5-0.5B-Instruct")))
    print()
    # ---- 案例五：溯源鲁棒性（2.5.0 新增）----
    print(format_stability(stability(target="Qwen2.5-0.5B")))
    print()
    # ---- 案例六：家族族谱（2.5.0 新增）----
    print(format_family(family_tree()))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())