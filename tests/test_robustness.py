"""
aiq-geometric-forensics 健壮性（异常路径）测试
=====================================================================
覆盖大厂生产要求的非 happy-path 场景，确保任何输入都不崩溃、不伪造：
  1. 未知模型 → 安全降级 fallback（占位值，非某模型实测）
  2. 损坏/不可读 JSON 存档 → 降级 fallback 而非抛异常
  3. 空 / 非数值 / 负值 H_median 的 AIQ 均「有界且非 nan」
  4. 缺 torch 时 measure 依赖守卫抛清晰 ImportError（绝不静默伪造）
  5. diagnose 对含非数值 proj_gamma 的存档仍正确（过滤后取独立 spl）

用法（本包根目录下）：
  python tests/test_robustness.py
退出码：全过 0，任一失败 1。
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from modules import schema  # noqa: E402
from modules.data import Metrics, find_baseline_models, sanitize_path  # noqa: E402
from modules.health import aiq_factors, diagnose  # noqa: E402

_PASS = 0
_FAIL = 0


def check(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if bool(cond):
        _PASS += 1
        print(f"  [PASS] {msg}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {msg}")


def _base_dir() -> str:
    return os.path.join(_ROOT, "baselines")


def _tmppath(name: str) -> str:
    return os.path.join(_base_dir(), name)


def test_unknown_model_fallback() -> None:
    print("\n[data] 未知模型 → 安全降级 fallback（不崩、不伪造）")
    pass_ = None
    try:
        m = Metrics("__definitely_not_a_real_model__")
        pass_ = True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] 未知模型抛出未捕获异常: {type(e).__name__}: {e}")
    check(bool(pass_), "构造未知模型 Metrics 不抛异常")
    if pass_:
        check(m.source == "fallback", f"source 如实标注 fallback（实际={m.source}）")
        check(m.get_float("aiq.AIQ", 9.9) == 0.0,
              "AIQ 无实测 → 返回兜底占位 0.0（非某模型实测值）")
        check(abs(m.get_float("curvature.DEFF_plat", 0.0) - schema.PI_HALF) < 1e-12,
              "DEFF_plat 理论锚点 π/2")


def test_corrupt_json_fallback() -> None:
    print("\n[data] 损坏 JSON 存档 → 降级 fallback 而非崩溃")
    probe = _tmppath("__corrupt_probe__.json")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("{ this is not valid json !!! ")
    try:
        m = Metrics("__corrupt_probe__")
        clean = True
    except Exception as e:  # noqa: BLE001
        clean = False
        print(f"  [FAIL] 损坏存档抛未捕获异常: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    check(clean, "损坏 JSON 不崩溃")
    if clean:
        check(m.source == "fallback", f"损坏存档降级为 fallback（实际={m.source}）")


def test_aiq_edge_inputs_bounded() -> None:
    print("\n[health] AIQ 边界输入均「有界且非 nan」")
    cases: list[tuple[str, dict[str, Any]]] = [
        ("空 gamma", {"gamma_layers": [], "deff_cv": 0.01, "h_median": 0.5,
                      "spl": 0.0}),
        ("负 H_median", {"gamma_layers": [0.5] * 8, "deff_cv": 0.0,
                         "h_median": -2.0, "spl": 0.4}),
        ("超大 H_median", {"gamma_layers": [0.5] * 8, "deff_cv": 0.5,
                           "h_median": 1e9, "spl": 0.4}),
        ("单层 gamma", {"gamma_layers": [0.8], "deff_cv": 0.1, "h_median": 1.2,
                        "spl": 0.5}),
    ]
    for label, kwargs in cases:
        r = aiq_factors(**kwargs)
        ok = (0.0 <= r["AIQ"] <= 100.0 and not np.isnan(r["AIQ"])
              and 0.0 <= r["f3"] <= 1.0)
        check(ok, f"{label}: AIQ={r['AIQ']:.3f} 有界, f3={r['f3']:.3f}∈[0,1]")


def test_missing_torch_guard() -> None:
    print("\n[harness] 缺 torch 时依赖守卫给出可解读错误（不静默兜底伪造）")
    import builtins  # 标准库导入前置（ruff I001 排序）

    from modules import harness  # noqa: E402  # noqa: PLC0415  函数内导入（模拟拦截）

    # 环境无关化：无论本机是否装有 torch，都模拟"缺失"场景。
    # 做法：临时将 harness 的全局缓存清空，并把 builtins.__import__ 替换为
    # 一个对 torch/transformers 抛 ImportError 的桩；结束后恢复，零副作用。
    _real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name in ("torch", "transformers") or name.startswith("torch.") \
                or name.startswith("transformers."):
            raise ImportError(
                f"No module named '{name}' (aiq measure 需 torch+transformers)")
        return _real_import(name, *args, **kwargs)

    harness._torch = None          # 清空模块级缓存，强制走 import 分支
    harness._tr = None
    builtins.__import__ = _blocked  # 拦截 torch/transformers 导入
    try:
        harness._load_ml()
        got = False
    except ImportError as e:
        got = "torch" in repr(e) or "transformers" in repr(e)
        print(f"  ↳ 守卫提示: {str(e)[:60]}…")
    finally:
        builtins.__import__ = _real_import  # 恢复真实 import，零副作用
        harness._torch = None       # 复位缓存（后续真实测量重新加载）
        harness._tr = None
    check(got, "缺 torch/transformers → ImportError 并指明所需依赖")


def test_diagnose_non_numeric_proj_robust() -> None:
    print("\n[health] 含非数值 proj_gamma 的存档仍可靠（过滤后取独立 spl）")
    # 找任意真实模型构建副本，篡改一个 proj_gamma 值为非数值字符串
    names = find_baseline_models()
    if not names:
        check(False, "无可用基准存档")
        return
    src = names[0]
    probe = _tmppath("__proj_probe__.json")
    with open(os.path.join(_base_dir(), f"{src}.json"), encoding="utf-8") as f:
        doc = json.load(f)
    doc["spectral"]["proj_gamma"]["k_proj"] = "not-a-number"   # 注入脏数据
    with open(probe, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    try:
        rep = diagnose("__proj_probe__")
        ok = 0 <= rep["aiq"]["AIQ"] <= 100 and not np.isnan(rep["aiq"]["AIQ"])
        ok = ok and abs(rep["aiq"]["f4"] - rep["aiq"]["f1"]) < 1.0  # 回落不越界
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] 脏 proj_gamma 崩溃: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    check(ok, "脏 proj_gamma → 过滤非数值后 AIQ 仍良定")


def test_sanitize_path_privacy() -> None:
    print("\n[data] sanitize_path 去除 model 字段隐私路径，只留 basename")
    out = sanitize_path({"model": "/home/bob/.cache/huggingface/Qwen2.5-0.5B",
                         "keep": "值"})
    check(out["model"] == "Qwen2.5-0.5B" and out["keep"] == "值",
          f"仅净化 model 字段: {out['model']}")
    # 跨平台：Windows 反斜杠路径也应净化为 basename
    win = sanitize_path({"model": "C:\\\\Users\\\\alice\\\\models\\\\bloom-560m"})
    check(win["model"] == "bloom-560m", f"Windows 路径净化: {win['model']}")


def main() -> int:
    print("=" * 62)
    print("aiq-geometric-forensics 健壮性（异常路径）套件")
    print("=" * 62)
    test_unknown_model_fallback()
    test_corrupt_json_fallback()
    test_aiq_edge_inputs_bounded()
    test_missing_torch_guard()
    test_diagnose_non_numeric_proj_robust()
    test_sanitize_path_privacy()
    print("=" * 62)
    print(f"健壮性汇总: PASS={_PASS}  FAIL={_FAIL}")
    print("=" * 62)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
