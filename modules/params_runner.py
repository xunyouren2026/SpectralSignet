"""
aiq-geometric-forensics.modules.params_runner — 参数验证体系统一收口
=====================================================================
把插件内置的 `params/` 59 参数验证脚本（verify.py）统一收口为可编程
批量执行器，消除"59 个独立脚本散落运行、无统一入口"的工程短板。

能力：
  * list_params()          —— 枚举全部参数（按组 A-T 排序）
  * verify_one(name)       —— 运行单个参数验证（子进程隔离）
  * verify_all(limit=None) —— 批量运行全部参数（可选 limit 限前 N 个）

设计要点：
  * 每个 verify.py 以**子进程**方式运行（cwd=参数目录），保持其原有
    sys.path 注入与相对定位逻辑完全不变——零侵入、零风险；
  * 结果统一汇总（PASS/FAIL/时间），退出码=0 全过 / 1 有失败；
  * 供 cli `verify` / `verify-all` 子命令与 selfcheck 门禁调用。

用法：
  import modules.params_runner as PR
  PR.verify_all()                       # 批量跑全部 59 参数
  PR.verify_one("T04_AIQ因子权重")       # 跑单个参数
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

# params/ 目录 = 本模块上两级（modules/ -> skills 根），取 skills 根/params
_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARAMS_DIR = os.path.join(_SKILL_ROOT, "params")


@dataclass
class ParamResult:
    """单个参数验证结果。"""
    name: str          # 参数目录名（如 "A01_model"）
    ok: bool           # 是否通过
    exit_code: int     # 子进程退出码
    seconds: float     # 运行耗时（秒）
    group: str         # 组别（A/B/.../T）


def list_params() -> list[str]:
    """枚举 params/ 下全部含 verify.py 的参数目录（按组+编号排序）。

    返回：参数目录名列表，如 ["A01_model", "A02_prompt", ...]。
    """
    if not os.path.isdir(_PARAMS_DIR):
        return []
    names = [d for d in os.listdir(_PARAMS_DIR)
             if os.path.isdir(os.path.join(_PARAMS_DIR, d))
             and not d.startswith("__")
             and os.path.isfile(os.path.join(_PARAMS_DIR, d, "verify.py"))]
    return sorted(names)          # 按 A01 < A02 < ... < T07 字典序天然分组


def params_count() -> int:
    """参数总数（selfcheck 门禁用）。"""
    return len(list_params())


def verify_one(name: str, timeout: int = 120) -> ParamResult:
    """运行单个参数验证（子进程隔离）。

    参数：
      name: 参数目录名（如 "A01_model"）；不存在返回 ok=False。
      timeout: 子进程超时秒数（防单脚本卡死拖垮全量）。
    返回：ParamResult。
    """
    vpy = os.path.join(_PARAMS_DIR, name, "verify.py")
    if not os.path.isfile(vpy):
        return ParamResult(name=name, ok=False, exit_code=-1,
                           seconds=0.0, group=name[:1].upper())
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "verify.py"], cwd=os.path.dirname(vpy),
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
        ok = (r.returncode == 0)
        code = r.returncode
    except subprocess.TimeoutExpired:
        ok, code = False, -2          # -2 = 超时
    except Exception:
        ok, code = False, -3          # -3 = 运行异常
    return ParamResult(name=name, ok=ok, exit_code=code,
                       seconds=time.time() - t0, group=name[:1].upper())


def verify_all(limit: int | None = None, timeout: int = 120) -> dict[str, Any]:
    """批量运行全部参数验证。

    参数：
      limit: 可选，只跑前 N 个参数（快速冒烟用）。
      timeout: 每脚本超时秒数。
    返回：
      {"total", "passed", "failed", "seconds", "results": [ParamResult...],
       "ok": bool}
    """
    names = list_params()
    if limit is not None:
        names = names[:limit]
    results: list[ParamResult] = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        res = verify_one(name, timeout=timeout)
        results.append(res)
        tag = "PASS" if res.ok else "FAIL"
        print(f"  [{i:>2}/{len(names)}] {tag} {name:<26} "
              f"{res.seconds:.1f}s")
    passed = sum(1 for r in results if r.ok)
    return {
        "total": len(results), "passed": passed,
        "failed": len(results) - passed,
        "seconds": time.time() - t0,
        "results": [{"name": r.name, "ok": r.ok, "exit_code": r.exit_code,
                     "seconds": round(r.seconds, 2), "group": r.group}
                    for r in results],
        "ok": passed == len(results) and len(results) > 0,
    }
    # ---- SOTA 增强：按组聚合 + 性能统计 ----
    from collections import defaultdict
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rep["results"]:
        by_group[r["group"]].append(r)
    rep["groups"] = {g: {
        "total": len(rs),
        "passed": sum(1 for r in rs if r["ok"]),
        "failed": sum(1 for r in rs if not r["ok"]),
        "seconds": round(sum(r["seconds"] for r in rs), 2),
    } for g, rs in sorted(by_group.items())}
    secs = [r["seconds"] for r in rep["results"]]
    rep["performance"] = {
        "total_seconds": round(rep["seconds"], 2),
        "mean_seconds": round(sum(secs) / len(secs), 2) if secs else 0.0,
        "slowest": max(rep["results"], key=lambda r: r["seconds"])["name"] if rep["results"] else None,
        "slowest_seconds": round(max(secs), 2) if secs else 0.0,
        "fastest_seconds": round(min(secs), 2) if secs else 0.0,
    }
    rep["failures"] = [r["name"] for r in rep["results"] if not r["ok"]]
    return rep


def format_summary(rep: dict[str, Any]) -> str:
    """把批量结果渲染为汇总文本。"""
    lines = [
        "=" * 62,
        f"参数验证体系批量运行（{rep['total']} 个参数，共 {rep['seconds']:.1f}s）",
        "=" * 62,
    ]
    # 按组汇总
    from collections import defaultdict
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rep["results"]:
        by_group[r["group"]].append(r)
    for g in sorted(by_group):
        grp = by_group[g]
        g_ok = sum(1 for r in grp if r["ok"])
        lines.append(f"  [{g}] 组 {g_ok}/{len(grp)} 通过")
        for r in grp:
            mark = "✓" if r["ok"] else "✗"
            lines.append(f"      {mark} {r['name']} ({r['seconds']:.1f}s)")
    lines += ["-" * 62,
              f"总计: PASS={rep['passed']} FAIL={rep['failed']} "
              f"→ {'全部通过' if rep['ok'] else '存在失败'}"]
    # SOTA：性能统计
    if "performance" in rep:
        perf = rep["performance"]
        lines.append(f"性能: 总 {perf['total_seconds']}s / 均 {perf['mean_seconds']}s / "
                     f"最慢 {perf['slowest']}({perf['slowest_seconds']}s)")
    if rep.get("failures"):
        lines.append(f"失败明细: {', '.join(rep['failures'])}")
    lines.append("=" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------- 直接运行
if __name__ == "__main__":
    rep = verify_all()
    print(format_summary(rep))
    raise SystemExit(0 if rep["ok"] else 1)
