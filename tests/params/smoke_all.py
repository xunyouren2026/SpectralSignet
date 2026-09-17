# -*- coding: utf-8 -*-
"""全量冒烟测试脚本 — 批量执行 params/ 下全部参数验证脚本
=====================================================================
用途：自动化执行 `params/` 下所有一级子文件夹中的 `verify.py`，对每个
脚本做进程隔离运行，并按"慢阈值 / 超时"判定 PASS / WARNING / FAIL，
输出逐项明细、汇总统计、失败详情与慢脚本清单，供 CI/CD 作为门禁。

用法（在 aiq-geometric-forensics 根目录下运行）：
    python tests/params/smoke_all.py                 # 全量顺序执行
    python tests/params/smoke_all.py --skip-run      # 仅静态扫描，不运行
    python tests/params/smoke_all.py --jobs 4        # 4 进程并行执行
    python tests/params/smoke_all.py --threshold 8.0 # 慢脚本阈值(秒)
    python tests/params/smoke_all.py --timeout 300   # 单脚本超时(秒)
    python tests/params/smoke_all.py --json smoke.json  # 导出 JSON 报告

参数：
    --params-root   params/ 根目录（默认：脚本上级两级 = 插件根/params）
    --python        Python 解释器（默认：当前 sys.executable）
    --timeout       单脚本超时秒数（默认 600）
    --threshold     慢脚本阈值秒数（默认 5.0，仅告警不判失败）
    --tail          失败诊断保留的 stdout/stderr 尾部字符数（默认 3000）
    --jobs          并发执行数（默认 1 = 顺序执行；>1 用线程池并行）
    --json          可选 JSON 报告输出路径
    --skip-run      仅列出全部 verify.py 路径，不实际执行

返回码：0 = 全部 PASS；1 = 存在至少一个 FAIL（WARNING 不影响）。

设计要点：
    - 路径处理用 pathlib，Windows/Linux 通用；
    - 每个 verify.py 以独立子进程运行，互不污染；
    - 超时（TimeoutExpired）与启动失败（FileNotFoundError/OSError）均计 FAIL，
      且不会中断后续脚本执行；
    - 结果按脚本名排序渲染，保证输出确定性；
    - 阈值与超时集中在 SmokeConfig 中，易于调整与扩展。
仅使用标准库（os/pathlib/subprocess/concurrent.futures/argparse/json），
兼容 Python 3.8+（from __future__ import annotations + typing 兼容写法）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 配置对象：所有可调参数集中于此，便于扩展
# ---------------------------------------------------------------------------


@dataclass
class SmokeConfig:
    """冒烟测试运行配置。

    Attributes:
        params_root: 参数脚本根目录（含各参数子文件夹）。
        python_exe: 用于执行 verify.py 的解释器路径。
        timeout_sec: 单脚本超时（秒），超时判 FAIL。
        slow_threshold_sec: 慢脚本阈值（秒），超过仅 WARNING 不影响返回码。
        tail_chars: 失败诊断保留的 stdout/stderr 尾部字符数。
        jobs: 并发执行数；1 为顺序执行，>1 使用线程池并行。
    """

    params_root: Path
    python_exe: str = sys.executable
    timeout_sec: float = 600.0
    slow_threshold_sec: float = 5.0
    tail_chars: int = 3000
    jobs: int = 1


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """单个 verify.py 的执行结果。

    Attributes:
        name: 参数文件夹名（如 A01_model）。
        script: verify.py 的绝对路径。
        exit_code: 进程退出码（超时/启动失败时为 None）。
        elapsed_sec: 墙钟耗时（秒）。
        timed_out: 是否超时。
        startup_failed: 是否启动失败（解释器缺失/路径异常等）。
        stdout_tail: stdout 尾部（供失败诊断）。
        stderr_tail: stderr 尾部（供失败诊断）。
        error: 启动异常信息（无则 None）。
    """

    name: str
    script: Path
    exit_code: Optional[int] = None
    elapsed_sec: float = 0.0
    timed_out: bool = False
    startup_failed: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: Optional[str] = None

    @property
    def status(self) -> str:
        """状态判定：FAIL（非0/超时/启动失败）> WARNING（慢）> PASS。"""
        if self.exit_code != 0 or self.timed_out or self.startup_failed:
            return "FAIL"
        if self.elapsed_sec > cfg_slow_threshold_global():
            return "WARNING"
        return "PASS"


# 全局慢阈值（供 status 属性使用；以配置构造时经 _set_slow_threshold 注入）
_slow_threshold_sec: float = 5.0


def cfg_slow_threshold_global() -> float:
    """返回全局慢阈值（status 属性判定用）。"""
    return _slow_threshold_sec


def _set_slow_threshold(value: float) -> None:
    """设置全局慢阈值（main 启动时由配置同步）。"""
    global _slow_threshold_sec
    _slow_threshold_sec = value


# ---------------------------------------------------------------------------
# 自动发现
# ---------------------------------------------------------------------------


def discover_scripts(params_root: Path) -> List[Path]:
    """递归扫描 params_root 下所有一级子文件夹中的 verify.py。

    Args:
        params_root: 参数脚本根目录。

    Returns:
        按文件夹名排序的 verify.py 绝对路径列表。

    Raises:
        FileNotFoundError: params_root 不存在或目录下无任何 verify.py。
    """
    if not params_root.is_dir():
        raise FileNotFoundError(f"参数根目录不存在: {params_root}")
    scripts = [
        d / "verify.py"
        for d in sorted(params_root.iterdir())
        if d.is_dir() and (d / "verify.py").is_file()
    ]
    if not scripts:
        raise FileNotFoundError(f"未在 {params_root} 下发现任何 verify.py")
    return scripts


# ---------------------------------------------------------------------------
# 单脚本执行（进程隔离）
# ---------------------------------------------------------------------------


def run_one(config: SmokeConfig, script: Path) -> CaseResult:
    """在新进程中执行单个 verify.py，返回结果记录。

    异常兜底：FileNotFoundError（解释器缺失）、OSError（启动失败）、
    TimeoutExpired（超时）均被捕获并转为 FAIL 结果，不影响其他脚本。

    Args:
        config: 冒烟测试配置。
        script: verify.py 绝对路径。

    Returns:
        CaseResult 记录（状态在 status 属性中判定）。
    """
    name = script.parent.name
    result = CaseResult(name=name, script=script)
    cmd = [config.python_exe, str(script)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_sec,
            errors="replace",  # 中文日志在 Windows 下避免解码崩溃
        )
        result.exit_code = proc.returncode
        result.stdout_tail = proc.stdout[-config.tail_chars :]
        result.stderr_tail = proc.stderr[-config.tail_chars :]
    except subprocess.TimeoutExpired as exc:
        result.timed_out = True
        result.exit_code = None
        result.stdout_tail = (exc.stdout or "")[-config.tail_chars :]
        result.stderr_tail = (exc.stderr or "")[-config.tail_chars :]
        result.error = f"超时（>{config.timeout_sec:g}s）"
    except FileNotFoundError:
        result.startup_failed = True
        result.error = f"解释器不存在: {config.python_exe}"
    except OSError as exc:
        result.startup_failed = True
        result.error = f"启动失败: {exc}"
    except Exception as exc:  # 兜底：任何意外异常不中断整体
        result.startup_failed = True
        result.error = f"异常: {exc!r}"
    finally:
        result.elapsed_sec = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# 渲染（明细 / 汇总 / 失败详情 / 慢脚本 / JSON）
# ---------------------------------------------------------------------------


def render_table(results: Sequence[CaseResult]) -> str:
    """渲染逐项明细表格。"""
    header = f"{'脚本':<34}{'退出码':>8}{'耗时(s)':>10}  状态"
    line = "-" * len(header)
    rows = [header, line]
    for r in results:
        code = "-" if r.exit_code is None else str(r.exit_code)
        rows.append(f"{r.name:<34}{code:>8}{r.elapsed_sec:>10.2f}  {r.status}")
    return "\n".join(rows)


def summarize(results: Sequence[CaseResult]) -> Dict[str, float]:
    """汇总统计：PASS/FAIL/WARNING 数量与总耗时。"""
    counts = {"PASS": 0, "FAIL": 0, "WARNING": 0}
    total = 0.0
    for r in results:
        counts[r.status] += 1
        total += r.elapsed_sec
    return {"counts": counts, "total_sec": total}


def render_failures(results: Sequence[CaseResult], tail: int = 3000) -> str:
    """渲染全部 FAIL 脚本的 stdout/stderr 尾部，供 CI 诊断。"""
    fails = [r for r in results if r.status == "FAIL"]
    if not fails:
        return ""
    blocks = ["=" * 74, f"失败详情（{len(fails)} 个 FAIL）", "=" * 74]
    for r in fails:
        blocks.append(f"\n--- [{r.name}] 退出码={r.exit_code} 耗时={r.elapsed_sec:.2f}s "
                      f"超时={r.timed_out} 启动失败={r.startup_failed} 错误={r.error or '-'}")
        blocks.append(f"    stdout 尾部:\n{r.stdout_tail}")
        blocks.append(f"    stderr 尾部:\n{r.stderr_tail}")
    return "\n".join(blocks)


def render_warnings(results: Sequence[CaseResult]) -> str:
    """渲染慢脚本清单（按耗时降序）。"""
    warns = sorted(
        (r for r in results if r.status == "WARNING"),
        key=lambda r: r.elapsed_sec,
        reverse=True,
    )
    if not warns:
        return ""
    blocks = ["慢脚本清单（WARNING，按耗时降序）:"]
    for r in warns:
        blocks.append(f"  {r.name:<34} {r.elapsed_sec:>8.2f}s")
    return "\n".join(blocks)


def render_json_report(results: Sequence[CaseResult]) -> dict:
    """构建 JSON 报告结构（供 --json 输出与分析）。"""
    summary = summarize(results)
    return {
        "summary": {
            "total": len(results),
            "pass": summary["counts"]["PASS"],
            "fail": summary["counts"]["FAIL"],
            "warning": summary["counts"]["WARNING"],
            "total_sec": round(summary["total_sec"], 3),
        },
        "cases": [
            {
                "name": r.name,
                "script": str(r.script),
                "exit_code": r.exit_code,
                "elapsed_sec": round(r.elapsed_sec, 3),
                "status": r.status,
                "timed_out": r.timed_out,
                "startup_failed": r.startup_failed,
                "error": r.error,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------


def run_smoke(config: SmokeConfig, progress: bool = True) -> Tuple[int, List[CaseResult]]:
    """执行全部冒烟测试（支持顺序/并发），返回 (返回码, 结果列表)。

    Args:
        config: 冒烟测试配置。
        progress: 是否打印实时进度（(i/N) 运行 xxx ...）。

    Returns:
        (0 全部 PASS / 1 存在 FAIL, 按脚本名排序的结果列表)。
    """
    scripts = discover_scripts(config.params_root)
    _set_slow_threshold(config.slow_threshold_sec)

    results: List[CaseResult] = []
    total = len(scripts)
    if config.jobs > 1:
        # 并发执行：线程池并行启动子进程；完成即记录，最后统一排序渲染
        with ThreadPoolExecutor(max_workers=config.jobs) as pool:
            futures = {pool.submit(run_one, config, s): s for s in scripts}
            for i, fut in enumerate(as_completed(futures), 1):
                s = futures[fut]
                res = fut.result()
                results.append(res)
                if progress:
                    print(f"[smoke_all] ({i}/{total}) {res.name} -> {res.status} "
                          f"exit={res.exit_code} {res.elapsed_sec:.2f}s")
    else:
        # 顺序执行：稳定、输出有序
        for i, script in enumerate(scripts, 1):
            if progress:
                print(f"[smoke_all] ({i}/{total}) 运行 {script.parent.name} ...")
            res = run_one(config, script)
            results.append(res)
            if progress:
                print(f"[smoke_all]   -> {res.status}: exit={res.exit_code}, "
                      f"elapsed={res.elapsed_sec:.2f}s")

    results.sort(key=lambda r: r.name)  # 确定性排序
    return (0 if not any(r.status == "FAIL" for r in results) else 1), results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="AIQ 参数验证全量冒烟测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    script_dir = Path(__file__).resolve().parent
    default_root = (script_dir.parent.parent / "params").resolve()
    parser.add_argument("--params-root", type=Path, default=default_root,
                        help="params/ 根目录")
    parser.add_argument("--python", default=sys.executable,
                        help="Python 解释器路径")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="单脚本超时（秒）")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="慢脚本阈值（秒），仅告警不判失败")
    parser.add_argument("--tail", type=int, default=3000,
                        help="失败诊断保留的 stdout/stderr 尾部字符数")
    parser.add_argument("--jobs", type=int, default=1,
                        help="并发执行数（1=顺序）")
    parser.add_argument("--json", type=Path, default=None,
                        help="可选 JSON 报告输出路径")
    parser.add_argument("--skip-run", action="store_true",
                        help="仅静态扫描列出 verify.py，不执行")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：解析参数 → 扫描/执行 → 渲染报告 → 返回 CI 退出码。"""
    args = parse_args(argv)

    root: Path = args.params_root.resolve()
    scripts = discover_scripts(root)

    if args.skip_run:
        print(f"[smoke_all] 静态扫描（{len(scripts)} 个 verify.py，未执行）:")
        for s in scripts:
            print(f"  {s.relative_to(root.parent)}")
        return 0

    config = SmokeConfig(
        params_root=root,
        python_exe=args.python,
        timeout_sec=args.timeout,
        slow_threshold_sec=args.threshold,
        tail_chars=args.tail,
        jobs=args.jobs,
    )
    print(f"[smoke_all] 扫描到 {len(scripts)} 个 verify.py（params/ 根: {root}）")
    print(f"[smoke_all] 配置: timeout={config.timeout_sec:g}s "
          f"threshold={config.slow_threshold_sec:g}s jobs={config.jobs}")

    code, results = run_smoke(config)

    print("\n" + "=" * 74)
    print("AIQ 参数验证全量冒烟 — 逐项明细")
    print(render_table(results))
    print("=" * 74)
    summary = summarize(results)
    c = summary["counts"]
    print(f"汇总: {c['PASS']} PASS / {c['FAIL']} FAIL / {c['WARNING']} WARNING"
          f"（共 {len(results)} 个脚本）")
    print(f"总耗时: {summary['total_sec']:.2f}s")
    print("=" * 74)

    if c["WARNING"]:
        print(render_warnings(results))
    if c["FAIL"]:
        print(render_failures(results, args.tail))

    if args.json is not None:
        args.json.write_text(
            json.dumps(render_json_report(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[smoke_all] JSON 报告已写入: {args.json}")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
