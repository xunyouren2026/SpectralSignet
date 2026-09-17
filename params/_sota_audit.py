# -*- coding: utf-8 -*-
"""AIQ 参数验证 — SOTA 审计（_sota_audit）
=====================================================================
对 params/ 下全部 59 个 verify.py 做四层工厂架构的一致性审计，三项指标：

  1. 类型安全（type_safety）  ：Config 类字段全部带类型注解；
  2. 错误处理（error_handling）：ValidatorEngine 子类的 validate_* 方法
     显式 raise 的异常均为 AIQValidationError 子类（含根异常本身），
     或明确使用 ValueError 防御；
  3. 防御性（defensiveness）  ：对外部数据（rng / 输入参数 / 真实数据）
     做校验（assert / 输入校验 raise / isfinite / None 回退守卫等）。

审计方式：动态 import 每个 verify.py（不执行 main()），解析其类结构与
validate_* 方法 AST，对 raise 目标做 issubclass 判定——比纯正则更可靠。

用法：
  python params\\_sota_audit.py            # 打印 59/59 统计报告 + 写 logs/sota_audit.md
  python params\\_sota_audit.py --no-md    # 仅打印，不写 md
=====================================================================
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import inspect
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---- 自举：确保 params/ 在 sys.path（使 _factory/_errors 可导入） ----
_HERE = os.path.dirname(os.path.abspath(__file__))          # params/
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _errors import AIQValidationError  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402


def _glob_verify_scripts(params_dir: Path) -> list[Path]:
    """扫描 params/ 下全部 <文件夹>/verify.py，按文件夹名排序返回（预期 59 个）。"""
    return sorted(p for p in params_dir.glob("*/verify.py") if p.is_file())


def _load_module(verify_path: Path, index: int) -> Any:
    """动态 import 单个 verify.py（唯一模块名，不执行 main()）。

    模块级只运行 setup_env（stdout 编码 / sys.path 注入），无重计算。
    返回模块对象；失败抛出异常由调用方记录为 ERROR。
    """
    mod_name = f"_sota_audit_v{index}_{verify_path.parent.name.replace(chr(37), '')}"
    spec = importlib.util.spec_from_file_location(mod_name, str(verify_path))
    assert spec is not None and spec.loader is not None, f"无法构造 spec: {verify_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- ① 类型安全
def _find_config_class(mod: Any) -> Any:
    """定位 Config 配置模型类：名字以 Config 结尾且为 pydantic 模型或 dataclass。"""
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and name.endswith("Config") and not name.endswith("Factory"):
            if hasattr(obj, "model_fields") or dataclasses.is_dataclass(obj):
                return obj
    return None


def _audit_type_safety(mod: Any) -> tuple[bool, dict]:
    """① 类型安全：Config 类字段全部有类型注解。"""
    cfg = _find_config_class(mod)
    if cfg is None:
        return False, {"error": "未找到 Config 配置模型类"}
    if hasattr(cfg, "model_fields"):  # pydantic v2
        fields = cfg.model_fields
        annotated = len(fields) > 0 and all(
            getattr(f, "annotation", None) is not None for f in fields.values()
        )
        kind = "pydantic"
    else:  # dataclass 回退
        fields = dataclasses.fields(cfg)
        annotated = len(fields) > 0 and all(
            f.type not in (None, inspect.Parameter.empty, "") for f in fields
        )
        kind = "dataclass"
    return bool(annotated), {"cfg_class": cfg.__name__, "kind": kind,
                             "n_fields": len(fields), "annotated": annotated}


# ---------------------------------------------------------------- ② 错误处理
def _find_validator_class(mod: Any) -> Any:
    """定位 ValidatorEngine 子类（本地定义、继承 _factory.ValidatorEngine）。"""
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, _EngineBase) and obj is not _EngineBase:
            return obj
    return None


def _raise_is_ok(node: ast.Raise, mod: Any) -> tuple[bool, str]:
    """判定单个 raise 语句是否合规：AIQValidationError 子类 / ValueError / 裸 re-raise。"""
    exc = node.exc
    if exc is None:  # 裸 raise（except 内 re-raise）
        return True, "bare re-raise"
    if isinstance(exc, ast.Name):  # raise <变量>（re-raise 模式）
        return True, "re-raise var"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        exc_name = exc.func.id
        obj = getattr(mod, exc_name, None)
        if isinstance(obj, type):
            if issubclass(obj, AIQValidationError):
                return True, f"{exc_name} (AIQValidationError 系)"
            if obj is ValueError:
                return True, "ValueError (防御)"
            return False, f"{exc_name} 非 AIQValidationError 子类亦非 ValueError"
        return False, f"{exc_name} 无法在模块命名空间解析"
    return False, f"不支持的 raise 形式: {type(exc).__name__}"


def _audit_error_handling(mod: Any) -> tuple[bool, dict]:
    """② 错误处理：validate_* 方法显式 raise 均使用 AIQValidationError 子类或 ValueError。"""
    engine = _find_validator_class(mod)
    if engine is None:
        return False, {"error": "未找到 ValidatorEngine 子类"}
    methods = [n for n in vars(engine) if n.startswith("validate_")]
    if not methods:
        return False, {"error": "Validator 子类无 validate_* 方法"}
    bad: list[dict] = []
    detail_methods: list[dict] = []
    for m in methods:
        fn = getattr(engine, m)
        try:
            # inspect.getsource 保留类内缩进，须先 dedent 才能 ast.parse
            src = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError):  # 内建/动态方法无法取源
            detail_methods.append({"method": m, "note": "无法取源码"})
            continue
        tree = ast.parse(src)
        raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
        m_ok = True
        for r in raises:
            ok, why = _raise_is_ok(r, mod)
            if not ok:
                m_ok = False
                bad.append({"method": m, "line": r.lineno, "reason": why})
        detail_methods.append({"method": m, "n_raises": len(raises), "ok": m_ok})
    return (not bad), {"engine_class": engine.__name__, "n_methods": len(methods),
                       "methods": detail_methods, "bad": bad}


# ---------------------------------------------------------------- ③ 防御性
def _audit_defensiveness(mod: Any, source: str) -> tuple[bool, dict]:
    """③ 防御性：对外部数据（rng/输入参数/真实数据）做校验。

    命中任一机制即视为防御性达标：
      - assert 语句（参数/不变量守卫）；
      - 输入校验 raise（ValueError / ConfigError / SynthesisError）；
      - isfinite 数值检查；
      - 真实数据 None 回退守卫（is None / has_real / real_or / phi_pairs）；
      - 显式 rng 处理（default_rng / np.random.seed / seed_rng / manual_seed）。
    """
    tree = ast.parse(source)
    n_assert = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
    flags = {
        "asserts": n_assert > 0,
        "input_raise": bool(re.search(r"raise (ValueError|ConfigError|SynthesisError)\b", source)),
        "isfinite": bool(re.search(r"isfinite", source)),
        "none_guard": bool(re.search(r"\bis None\b|has_real\(|real_or\(|phi_pairs\(|_get_real_data\(", source)),
        "rng": bool(re.search(r"default_rng|np\.random\.seed|seed_rng|manual_seed", source)),
    }
    ok = any(flags.values())
    return ok, {"n_assert": n_assert, "flags": flags}


# ---------------------------------------------------------------- 报告输出
def _render_table(rows: list[dict]) -> str:
    """渲染逐脚本明细表（脚本 / 类型安全 / 错误处理 / 防御性 / 结论）。"""
    lines = [
        f"{'脚本':<24} {'类型安全':>8} {'错误处理':>8} {'防御性':>8}  结论",
        "-" * 66,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:<24} {r['type_safety']:>8} {r['error_handling']:>8} "
            f"{r['defensiveness']:>8}  {'PASS' if r['pass'] else 'FAIL'}"
        )
    return "\n".join(lines)


def _render_md(rows: list[dict], stats: dict, fail_rows: list[dict]) -> str:
    """渲染 Markdown 审计报告（logs/sota_audit.md）。"""
    lines = [
        "# AIQ 参数验证 SOTA 审计报告",
        "",
        f"- 生成时间：{os.path.basename(_HERE)} 环境，共审计 **{stats['total']}** 个 verify.py",
        "",
        "## 三向达标统计",
        "",
        "| 指标 | 达标数 |",
        "| --- | --- |",
        f"| 类型安全（Config 字段类型注解） | {stats['type_safety']}/{stats['total']} |",
        f"| 错误处理（validate_* 用 AIQValidationError 子类/ValueError） | {stats['error_handling']}/{stats['total']} |",
        f"| 防御性（外部数据校验） | {stats['defensiveness']}/{stats['total']} |",
        "",
        "## 逐脚本明细",
        "",
        "| 脚本 | 类型安全 | 错误处理 | 防御性 | 结论 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {'PASS' if r['type_safety'] else 'FAIL'} | "
            f"{'PASS' if r['error_handling'] else 'FAIL'} | "
            f"{'PASS' if r['defensiveness'] else 'FAIL'} | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    if fail_rows:
        lines += ["", "## 未达标明细", ""]
        for r in fail_rows:
            lines.append(f"### {r['name']}")
            if r.get("import_error"):
                lines.append(f"- 动态 import 失败：{r['import_error']}")
            if r.get("detail_type_safety") and not r["type_safety"]:
                lines.append(f"- 类型安全：{r['detail_type_safety']}")
            if r.get("detail_error_handling") and not r["error_handling"]:
                d = r["detail_error_handling"]
                lines.append(f"- 错误处理：{d.get('error', '')}；违规项 {d.get('bad')}")
            if r.get("detail_defensiveness") and not r["defensiveness"]:
                lines.append(f"- 防御性：{r['detail_defensiveness']}")
    lines += ["", "---", "_由 params/_sota_audit.py 自动生成_"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """SOTA 审计入口：扫描 → 逐脚本三向审计 → 打印统计 + 可选写 md。"""
    parser = argparse.ArgumentParser(prog="sota_audit", description="AIQ 参数验证 SOTA 审计")
    parser.add_argument("--no-md", action="store_true", help="不写 logs/sota_audit.md")
    args = parser.parse_args(argv)

    params_dir = Path(_HERE)
    scripts = _glob_verify_scripts(params_dir)
    print(f"[sota_audit] 扫描到 {len(scripts)} 个 verify.py（params/）")

    rows: list[dict] = []
    for i, sp in enumerate(scripts, start=1):
        name = sp.parent.name
        row: dict = {"name": name, "type_safety": False, "error_handling": False,
                     "defensiveness": False, "pass": False, "import_error": None}
        try:
            mod = _load_module(sp, i)
            source = sp.read_text(encoding="utf-8", errors="replace")
            ts_ok, ts_d = _audit_type_safety(mod)
            eh_ok, eh_d = _audit_error_handling(mod)
            df_ok, df_d = _audit_defensiveness(mod, source)
            row["type_safety"] = ts_ok
            row["error_handling"] = eh_ok
            row["defensiveness"] = df_ok
            row["detail_type_safety"] = ts_d
            row["detail_error_handling"] = eh_d
            row["detail_defensiveness"] = df_d
        except Exception as exc:  # import/解析失败：整脚本记 FAIL
            row["import_error"] = f"{type(exc).__name__}: {exc}"
        row["pass"] = (row["type_safety"] and row["error_handling"]
                       and row["defensiveness"] and row["import_error"] is None)
        rows.append(row)
        mark = "PASS" if row["pass"] else "FAIL"
        print(f"[sota_audit] ({i}/{len(scripts)}) {name:<24} {mark}", flush=True)

    # ---- 统计汇总 ----
    total = len(rows)
    stats = {
        "total": total,
        "type_safety": sum(1 for r in rows if r["type_safety"]),
        "error_handling": sum(1 for r in rows if r["error_handling"]),
        "defensiveness": sum(1 for r in rows if r["defensiveness"]),
    }
    n_pass = sum(1 for r in rows if r["pass"])
    fail_rows = [r for r in rows if not r["pass"]]

    print("\n" + "=" * 74)
    print("SOTA 审计 — 逐脚本明细")
    print(_render_table(rows))
    print("\n" + "=" * 74)
    print("SOTA 审计 — 三向达标统计")
    print(f"  类型安全（Config 字段类型注解）            : {stats['type_safety']}/{total}")
    print(f"  错误处理（validate_* 用 AIQValidationError/ValueError）: {stats['error_handling']}/{total}")
    print(f"  防御性（外部数据校验）                    : {stats['defensiveness']}/{total}")
    print(f"  全项 PASS 脚本                            : {n_pass}/{total}")
    print("=" * 74)

    for r in fail_rows:
        print("\n" + "-" * 74)
        print(f"FAIL 明细: {r['name']}")
        if r.get("import_error"):
            print(f"  import 错误: {r['import_error']}")
        if r.get("detail_type_safety") and not r["type_safety"]:
            print(f"  类型安全: {r['detail_type_safety']}")
        if r.get("detail_error_handling") and not r["error_handling"]:
            print(f"  错误处理: {r['detail_error_handling']}")
        if r.get("detail_defensiveness") and not r["defensiveness"]:
            print(f"  防御性: {r['detail_defensiveness']}")

    # ---- 可选写 logs/sota_audit.md ----
    if not args.no_md:
        log_dir = os.path.join(os.path.dirname(_HERE), "logs")
        os.makedirs(log_dir, exist_ok=True)
        md_path = os.path.join(log_dir, "sota_audit.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_render_md(rows, stats, fail_rows))
        print(f"\n[sota_audit] Markdown 报告已写入: {md_path}")

    return 0 if n_pass == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
