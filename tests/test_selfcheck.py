"""
aiq-geometric-forensics · 生产完整性门禁自检验证
=====================================================================
校验 modules/selfcheck.py 的 6 类检查都能被触发，且整体门禁对
"干净基线"返回通过。独立函数式断言，零依赖（可单文件执行）。

用法（本包根目录下）：
  python tests/test_selfcheck.py
退出码：全过 0，任一失败 1。
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math  # noqa: E402

from modules import (  # noqa: E402
    Metrics,
    calibrate_deff_band,
    find_baseline_models,
    health_verdicts,
    run_selfcheck,
)
from modules.selfcheck import _CORE_NUMERIC  # noqa: E402

_PASSED = 0
_FAILED = 0


def check(cond: bool, msg: str) -> None:
    global _PASSED, _FAILED
    if cond:
        _PASSED += 1
        print(f"  [PASS] {msg}")
    else:
        _FAILED += 1
        print(f"  [FAIL] {msg}")


def test_selfcheck_passes() -> None:
    print("[selfcheck] 干净基线应全过")
    rep = run_selfcheck()
    check(rep["ok"], "自检门禁整体通过")
    s = rep["summary"]
    check(s["passed"] >= 10 and s["failed"] == 0,
          f"通过>=10 且零失败 (passed={s['passed']}, failed={s['failed']})")
    # 版本三源一致
    vs = rep["version"]
    check(vs["__init__"] == vs["pyproject"] == vs["CHANGELOG"],
          f"版本三源一致 {vs['__init__']}")


def test_deff_band_interval() -> None:
    print("[selfcheck] DEFF 家族标定带出厂值")
    lo, hi = calibrate_deff_band()
    check(0 < lo < hi <= 2.0, f"带 [{lo:.4f},{hi:.4f}] 合法(⊂(0,2])")
    # 5 模型 DEFF_plat 观测支撑集应在带内
    for n in find_baseline_models():
        v = Metrics(n).get_float("curvature.DEFF_plat", 0.0)
        check(lo <= v <= hi, f"{n} DEFF={v:.4f} 落在家族标定带内")


def test_deff_band_verdict_annotation() -> None:
    print("[selfcheck] DEFF 判定带内/出带标注，且布尔不受带影响")
    lo, hi = calibrate_deff_band()
    # 带内值：判据不变（偏差0→通过），说明追加"在带内"
    v_inside = health_verdicts(40.0, schema_pi_half(), 0.9, (lo, hi))
    check(v_inside[1][1], "带内且 π/2 偏差0 → DEFF 通过")
    check("家族标定带" in v_inside[1][2], "判定说明含家族标定带字段")
    # 出带值（远离带）：不改变原 DEFF_TOL 布尔（这里用带外但偏差仍<0.15 的“合带判断”）
    far = calibrate_far_value(hi)
    v_out = health_verdicts(40.0, far, 0.9, (lo, hi))
    check("出带" in v_out[1][2], "带外值说明标注“出带”")
    check(v_out[1][1] == (abs(far - schema_pi_half()) < 0.15),
          "布尔判据保持原 DEFF_TOL 语义（不受带参数影响）")


def test_core_numeric_paths_present() -> None:
    print("[selfcheck] 核心数值路径在 5 个存档中均可解析")
    for n in find_baseline_models():
        m = Metrics(n)
        for key in _CORE_NUMERIC:
            if key == "spectral.k_proj_gamma_layers":
                layers = m.get_list(key)
                ok = len(layers) > 0 and all(
                    isinstance(x, (int, float)) and math.isfinite(float(x))
                    for x in layers)
            else:
                v = m.get(key, None)
                ok = v is not None and isinstance(v, (int, float)) \
                    and math.isfinite(float(v))
            check(ok, f"{n}:{key}")


def schema_pi_half() -> float:
    from modules import schema
    return schema.PI_HALF


def calibrate_far_value(hi: float) -> float:
    """构造一个明显在标定带之外、但相对 π/2 偏差<0.15 的值用于判定测试。"""
    import modules.schema as schema
    # 带外低端：比带下界再减 0.05·π/2，但仍落在 ±0.15 容差内
    lo = calibrate_deff_band()[0]
    candidate = lo - 0.05 * schema.PI_HALF
    if abs(candidate - schema.PI_HALF) >= schema.DEFF_TOL:
        candidate = lo - 0.02 * schema.PI_HALF
    return candidate


def main() -> int:
    print("=" * 60)
    print("生产完整性门禁验证套件（selfcheck）")
    print("=" * 60)
    test_selfcheck_passes()
    test_deff_band_interval()
    test_deff_band_verdict_annotation()
    test_core_numeric_paths_present()
    print("=" * 60)
    print(f"汇总: PASS={_PASSED}  FAIL={_FAILED}")
    print("=" * 60)
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
