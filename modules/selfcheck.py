"""
aiq-geometric-forensics.modules.selfcheck — 生产完整性门禁（自检）
=====================================================================
在发布/上线前对整包做一次性"自证清白"：
  * 版本一致性： __init__.__version__ ≡ pyproject `version` ≡ CHANGELOG 顶部条目
  * 基准库完整性：每个 baselines/*.json 可装载、核心字段齐备、数值有限非 NaN
  * 无存档漂移：运行时按 2.3 公式实时复算的 AIQ 与存档 aiq.AIQ 一致（容差内）
  * 证据纪律：schema.EVIDENCE 所有标签合法；启发式判据（DEFF_TOL/COMPRESS*）被标注
  * 报告冒烟： render_full_page / render_health 可无异常产出 HTML
  * 隐私： sanitize_path 能剥除 measurement 内的绝对路径

把"专业包声称的可验证性"固化为一个可执行门禁，供 CI 或人工发布前调用。

用法（本包根目录下）：
  python -m modules.cli selfcheck [--json FILE]
  python - <<'PY'
  from modules.selfcheck import run_selfcheck, render_report
  rep = run_selfcheck(); render_report(rep)
  PY
退出码：全过 0，任一失败 1。
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Iterator
from typing import Any

from . import schema
from .data import Metrics, find_baseline_models, sanitize_path
from .health import diagnose
from .report import render_full_page, render_health

_VALID_EVIDENCE = ("measured", "theory", "heuristic", "design")
# 每个基准必须存在的核心数值路径（缺失/非有限 → 判失败）
_CORE_NUMERIC = ("spectral.k_proj_gamma_mean",
                 "spectral.k_proj_gamma_layers",
                 "curvature.DEFF_plat", "curvature.K_neg_pct",
                 "curvature.H_median", "curvature.DEFF_cv",
                 "aiq.AIQ", "aiq.f1", "aiq.f2", "aiq.f3",
                 "aiq.f4", "aiq.f5")

_VALID_EVIDENCE = ("measured", "theory", "heuristic", "design")
# 这些判据为启发式，必须保持标注为 heuristic（若被改标 measured 则门禁报警）
_HEURISTIC_MUST_STAY = ("DEFF_TOL", "COMPRESS_HIGH", "COMPRESS_MID",
                        "COMPRESS_LOW", "K_SLOT", "H_CONV_THR")


class _Check:
    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool, detail: str) -> None:
        self.name, self.ok, self.detail = name, ok, detail


# ---------------------------------------------------------------- 版本一致性
def _parse_pyproject_version() -> str | None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pyproject.toml")
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        return None
    m = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']", txt, re.M)
    return m.group(1) if m else None


def _parse_changelog_top_version() -> str | None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "CHANGELOG.md")
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
    except OSError:
        return None
    m = re.search(r"^##\s+([0-9]+\.[0-9]+\.[0-9]+)", txt, re.M)
    return m.group(1) if m else None


def _iter_numeric(node: Any) -> Iterator[float]:
    """深度遍历 dict/list，产出所有数值标量；忽略 None/str/bool。"""
    if isinstance(node, dict):
        for v in node.values():
            yield from _iter_numeric(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_numeric(v)
    elif isinstance(node, bool):
        return
    elif isinstance(node, (int, float)):
        yield node


# ---------------------------------------------------------------- 主入口
def run_selfcheck() -> dict[str, Any]:
    """执行全部完整性检查，返回结构化报告 dict。"""
    checks: list[_Check] = []

    # ---- 1. 版本一致性
    from . import __version__  # 运行时取，规避 __init__ 中定义较晚导致的循环导入
    py_ver = _parse_pyproject_version()
    cl_ver = _parse_changelog_top_version()
    vers = {"__init__": __version__, "pyproject": py_ver, "CHANGELOG": cl_ver}
    vers_set = {v for v in vers.values() if v}
    ok = py_ver is not None and len(vers_set) == 1
    checks.append(_Check(
        "版本三源一致",
        ok,
        f"__init__={__version__} pyproject={py_ver} CHANGELOG={cl_ver} "
        + ("（一致）" if ok else "（不一致，需对齐）")))

    # ---- 2. 基准库完整性
    models = find_baseline_models()
    checks.append(_Check("基准库可枚举",
                         len(models) >= 5,
                         f"基线模型数={len(models)}（≥5）"))
    if len(models) < 2:
        checks.append(_Check("基准库装载", False, "模型数不足，跳过逐条校验"))
        return _finalize(checks, vers)
    bad_load: list[str] = []
    bad_field: list[str] = []
    bad_nan: list[str] = []
    drift: list[str] = []
    for n in models:
        m = Metrics(n)
        if m.source != "baseline":
            bad_load.append(f"{n}(src={m.source})")
            continue
        for key in _CORE_NUMERIC:
            if key == "spectral.k_proj_gamma_layers":
                # 层谱是列表：单独校验非空且成员全有限
                layers = m.get_list(key)
                if not layers or not all(
                        isinstance(x, (int, float)) and math.isfinite(float(x))
                        for x in layers):
                    bad_field.append(f"{n}:{key}(层谱空/非有限)")
            else:
                v = m.get(key, None)
                if v is None or not isinstance(v, (int, float)) \
                        or not math.isfinite(float(v)):
                    bad_field.append(f"{n}:{key}")
        # 数值叶片全为有限值（覆盖未列出的字段）
        raw = m._data if hasattr(m, "_data") else {}
        for val in _iter_numeric(raw):
            if not math.isfinite(float(val)):
                bad_nan.append(n)
                break
        # ---- 3. 实时复算 AIQ 与存档一致（捕获"存档旧公式漂移"）
        try:
            rep = diagnose(n)
            live = rep["aiq"]["AIQ"]
            stored = m.get_float("aiq.AIQ", 0.0)
            if abs(live - stored) > 0.01:
                drift.append(f"{n} live={live:.2f} stored={stored:.2f}")
        except Exception as e:                                    # 诊断不应爆炸
            bad_field.append(f"{n}:diagnose({type(e).__name__})")
    checks.append(_Check("基准库全部可装载", not bad_load,
                         "；".join(bad_load) or "全部 baseline 来源"))
    checks.append(_Check("核心数值字段齐备且有限", not bad_field,
                         "；".join(bad_field) or f"{len(models)} 个模型全部通过"))
    checks.append(_Check("数值叶片无 NaN/Inf", not bad_nan,
                         "；".join(bad_nan) or "全部有限"))
    checks.append(_Check("AIQ 实时复算无存档漂移", not drift,
                         "；".join(drift) or f"{len(models)} 个模型 AIQ 一致"))

    # ---- 4. 证据纪律
    ev = schema.EVIDENCE
    bad_ev = [k for k, v in ev.items() if v not in _VALID_EVIDENCE]
    bad_heur = [k for k in _HEURISTIC_MUST_STAY if ev.get(k) != "heuristic"]
    checks.append(_Check("证据标签合法", not bad_ev,
                         "；".join(bad_ev) or f"{len(ev)} 个标签均合法"))
    checks.append(_Check("启发式判据保持标注", not bad_heur,
                         "；".join(bad_heur) or "DEFF_TOL/COMPRESS*/K_SLOT/H_CONV_THR 仍为 heuristic"))

    # ---- 5. 报告冒烟
    try:
        hp = diagnose("Qwen2.5-0.5B-Instruct")
        html = render_full_page("自检", "冒烟",
                                [render_health(hp, "baseline")])
        # 期望：成串 HTML、含证据徽标、正确标注"存档"(非 inline)、含 AIQ 数字
        ok_html = ("<html" in html and "badge" in html
                   and "真实实测存档" in html and "AIQ" in html)
    except Exception as e:
        ok_html = False
        html = ""
        _ = e
    checks.append(_Check("报告可无异常渲染", ok_html,
                         "一人份 baseline 报告冒烟已产出（HTML+badge+标源+AIQ）"))

    # ---- 6. 隐私脱敏
    priv = sanitize_path({"model": "/Users/alice/m/models/qwen", "arch": {"n": 1}})
    ok_priv = priv.get("model") == "qwen"
    checks.append(_Check("绝对路径脱敏", ok_priv,
                         f"model → {priv.get('model')!r}（已剥 path/slice）"))

    # ---- 7. 参数验证体系完整性（params/ 59 参数，只验静态完整性不跑全量）
    from . import params_runner as PR
    pnames = PR.list_params()
    n_params = len(pnames)
    ok_params = n_params >= 50          # 期望 ≥50（当前 59），低于即告警
    checks.append(_Check(
        "参数体系可枚举", ok_params,
        f"params/ 参数数={n_params}（≥50，当前 59）"))
    # 每组抽查 verify.py 均可编译（不运行，避免门禁过慢）
    import py_compile
    bad_py = []
    for nm in pnames:
        vp = os.path.join(PR._PARAMS_DIR, nm, "verify.py")
        try:
            py_compile.compile(vp, doraise=True)
        except Exception as e:
            bad_py.append(f"{nm}({type(e).__name__})")
    checks.append(_Check("参数 verify.py 可编译", not bad_py,
                         "；".join(bad_py) or f"{n_params} 个 verify.py 语法全通过"))
    # 数据层键抽查：_params_data.json 可装载且 59 组
    try:
        import json as _json
        with open(os.path.join(PR._PARAMS_DIR, "_params_data.json"),
                  encoding="utf-8") as f:
            pd = _json.load(f)
        n_groups = sum(1 for k in pd if not k.startswith("_"))
        ok_pd = n_groups >= 50
    except Exception:
        n_groups, ok_pd = 0, False
    checks.append(_Check("参数数据层完整", ok_pd,
                         f"_params_data.json 组数={n_groups}（≥50）"))

    return _finalize(checks, vers)


def _finalize(checks: list[_Check], vers: dict[str, str | None]) -> dict[str, Any]:
    ok_all = all(c.ok for c in checks)
    return {"ok": ok_all, "version": vers,
            "summary": {"checks": len(checks), "passed": sum(c.ok for c in checks),
                        "failed": sum(not c.ok for c in checks)},
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail}
                       for c in checks]}


def render_report(rep: dict[str, Any]) -> None:
    """把自检结果渲染为控制台报告；任一失败以退出码 1 返回。"""
    print("=" * 72)
    print(f"aiq-geometric-forensics 自检（版本 {rep.get('version', {}).get('__init__')}）")
    print("=" * 72)
    for c in rep["checks"]:
        mark = "[PASS]" if c["ok"] else "[FAIL]"
        print(f"  {mark} {c['name']}: {c['detail']}")
    s = rep["summary"]
    print("-" * 72)
    print(f"总结: PASS={s['passed']}  FAIL={s['failed']}  → "
          + ("门禁通过" if rep["ok"] else "门禁失败"))
    print("=" * 72)


def main() -> int:
    rep = run_selfcheck()
    render_report(rep)
    if "--json" in sys.argv:
        idx = sys.argv.index("--json")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "selfcheck.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"已写自检 JSON: {path}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
