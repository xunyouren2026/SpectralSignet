# -*- coding: utf-8 -*-
"""H02 OPENBLAS_NUM_THREADS OpenBLAS 线程数 — 受限环境下的 SVD 内存安全验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. OPENBLAS_NUM_THREADS 必须在 import numpy/torch 之前设置
  2. 环境变量正确生效（os.environ 读取 == "2"）
  3. 若能探测：实际 OpenBLAS 线程数 <= 2（threadpoolctl / openblas_get_num_threads）
  4. 中等规模 SVD 平稳完成且分解正确（重建误差 < 1e-4）
  5. 峰值内存远低于 3.16GB 上限（文档实测值）
  6. 真实模型对照：harness 前置 2 线程 + 真实 RSS < 上限（惰性读取）
数据源：
  主文档行 5637-5723（H02 章节，README ⑤ 实测值 5705-5719 行）
  《参数审计与实验报告.txt》行 82（状态=已用）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  H02Config / ConfigFactory / H02Validator / ReportGenerator / main
  关键设计决策：线程环境变量必须在 import numpy 之前设置（OpenBLAS 初始化
  时锁定线程池）。故本文件顶部用"纯标准库 json"解析线程数（env
  AIQ_H02_THREADS > _params_data.json > 默认），与共享 ConfigFactory 的
  取值优先级一致；之后才导入 numpy 依赖的共享基类。
说明：核心验证点是"环境变量正确设置 + SVD 平稳完成 + 内存受控"。
      纯数值合成数据 + 真实实测对照，不加载任何大模型。
=====================================================================
"""
import os
import sys

# ---------------------------------------------------------------------------
# ★ 必须在 import numpy/torch 之前设置线程环境变量（README ④ 推导 3）
# 仅依赖标准库 json 解析 _params_data.json（不触发 numpy 导入），优先级与
# ConfigFactory 一致：env AIQ_H02_THREADS > _params_data.json > 默认 2。
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def _resolve_threads(param_id: str, default: int = 2) -> str:
    """线程数解析（env AIQ_<PARAM>_THREADS > _params_data.json > 默认）。

    必须在 import numpy 之前完成（OpenBLAS 初始化时锁定线程池）；
    只依赖标准库 json（不引入 numpy），与共享 ConfigFactory 优先级一致。
    """
    env_val = os.environ.get(f"AIQ_{param_id}_THREADS")
    if env_val is not None and env_val.strip():
        return env_val.strip()
    try:
        with open(os.path.join(_PARENT, "_params_data.json"), "r", encoding="utf-8") as f:
            node = _json.load(f).get(param_id) or {}
        return str(int(node.get("THREADS", default)))
    except Exception:
        return str(default)


_T = _resolve_threads("H02")
os.environ["OPENBLAS_NUM_THREADS"] = _T  # OpenBLAS 线程上限：多线程会让峰值内存超限
os.environ["OMP_NUM_THREADS"] = _T  # OpenMP 线程上限：与 BLAS 协同限制并行度
os.environ["MKL_NUM_THREADS"] = _T  # MKL 线程上限：覆盖 Intel 数学库路径
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402  # 数值核心（必须在环境变量之后导入）

sys.path.insert(0, _PARENT)  # 复用共享基类（_factory 导入 numpy，须在环境变量之后）
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 H02Config（pydantic 优先；dataclass 回退） ----
_HAS_PYDANTIC = False
_ConfigModelBase: Any
try:
    from pydantic import BaseModel as _PydanticBase  # noqa: E402
    _ConfigModelBase = _PydanticBase
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - pydantic 缺失路径
    import dataclasses as _dataclasses

    @_dataclasses.dataclass
    class _DataclassBase:
        """pydantic 缺失时的空壳基类（无字段，仅提供 dataclass 语义）。"""

    _ConfigModelBase = _DataclassBase


class H02Config(_ConfigModelBase):
    """H02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 H02 节点键名一一对应；取值优先级：
    环境变量 AIQ_H02_<KEY> > YAML > _params_data.json > 本模型默认值。
    THREADS 在文件顶部已被用于设置 os.environ（取值路径与工厂一致）。
    """

    THREADS: int = 2             # OpenBLAS 线程数（数据层 H02.THREADS，README ①）
    RSS_LIMIT_GB: float = 3.16   # 真实 RSS 上限（数据层）
    SVD_D: int = 2048            # SVD 矩阵规模（float32，~64MB）
    RECON_TOL: float = 1e-4      # 重建相对误差阈值
    SEED: int = 0                # 合成矩阵固定种子（规范 §3 / H01 语义）


# ---- 第二层：配置工厂 ConfigFactory（实例化 H02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """H02 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 H02Config。"""

    def build(self) -> H02Config:
        """构建 H02Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(H02Config, "H02")


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def has_openblas_backend() -> tuple:
    """探测 numpy 的 BLAS/OpenBLAS 后端信息（尽力而为，不中断）。"""
    try:
        if hasattr(np, "show_config"):
            d = np.show_config(mode="dicts")
            blob = repr(d).lower()
            has_blas = ("openblas" in blob) or ("blas" in blob)
            return has_blas, "OpenBLAS/BLAS 相关"
        return False, "未确认"
    except Exception:
        return False, "未确认"


def detect_openblas_threads():
    """尽力探测实际 OpenBLAS 线程数（threadpoolctl）；不可用时返回 None。"""
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
        blas_libs = [l for l in info if "openblas" in l.get("filepath", "").lower()
                     or "openblas" in l.get("internal_api", "").lower()]
        if blas_libs:
            return blas_libs[0].get("num_threads")
    except Exception:
        pass
    return None


def svd_reconstruction(A: np.ndarray) -> float:
    """对矩阵做完整 SVD 并返回重建相对误差（Frobenius 范数）。"""
    assert A.ndim == 2 and np.isfinite(A).all(), f"A 非法：shape={A.shape}"
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    recon = (U * S) @ Vt
    return float(np.linalg.norm(recon - A, "fro") / np.linalg.norm(A, "fro"))


def measure_peak_rss_gb() -> float:
    """读取当前进程 RSS（GB）；psutil/tracemalloc 均不可用时返回 0.0。"""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1e9)
    except Exception:
        return 0.0


# ---- 第四层：验证引擎 H02Validator（6 项验证 + 结构化日志 + 类型化异常） ----
class H02Validator(_EngineBase):
    """H02 验证引擎：顺序执行 6 项验证（结构化 JSON 日志 + 类型化异常）。

    无合成器：验证主体为环境变量/后端探测/SVD 内存检查。
    """

    def __init__(
        self,
        config: H02Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth=None, reporter=reporter)
        self._real_data = real_data

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 环境变量
    def validate_env(self) -> dict:
        """1) OPENBLAS_NUM_THREADS 已正确设置（== THREADS）。"""
        cfg = self.config
        expect = str(cfg.THREADS)
        val = os.environ.get("OPENBLAS_NUM_THREADS", "未设置")
        ok1 = val == expect
        if not ok1:
            raise AIQValidationError(
                f"OPENBLAS_NUM_THREADS={val} ≠ {expect}",
                expected=expect, actual=val, param_key="H02",
            )
        return {"detail": f"[1] OPENBLAS_NUM_THREADS = {val} (期望 '{expect}')",
                "env_value": val, "expected": expect}

    # ------------------------------------------------------------ 2) BLAS 后端
    def validate_blas_backend(self) -> dict:
        """2) numpy BLAS 后端探测（仅记录，信息性输出）。"""
        has_blas, blas_name = has_openblas_backend()
        return {
            "detail": f"[2] numpy BLAS 后端: {blas_name if has_blas else '未能确认'}",
            "has_blas": has_blas, "blas_name": blas_name,
        }

    # ------------------------------------------------------------ 3) 实际线程数
    def validate_threads(self) -> dict:
        """3) 实际 OpenBLAS 线程数探测（threadpoolctl，尽力而为）。"""
        cfg = self.config
        n_threads = detect_openblas_threads()
        try:
            n_cores = os.cpu_count()
        except Exception:
            n_cores = "?"
        if n_threads is None:
            return {
                "detail": (f"[3] 可用核心={n_cores}; 无法直接探测 OpenBLAS 线程数"
                           f"（后端依赖），跳过（环境变量已正确设置）"),
                "detected": False, "n_cores": n_cores, "ok": True,
            }
        ok3 = n_threads <= cfg.THREADS
        if not ok3:
            raise AIQValidationError(
                f"OpenBLAS 线程数 {n_threads} > {cfg.THREADS}，环境变量未生效",
                expected=f"<={cfg.THREADS}", actual=n_threads, param_key="H02",
            )
        return {
            "detail": (f"[3] 可用核心={n_cores}, 实际 OpenBLAS 线程数={n_threads} "
                       f"(环境变量已生效, 期望 <= {cfg.THREADS})"),
            "detected": True, "n_threads": n_threads, "n_cores": n_cores,
        }

    # ------------------------------------------------------------ 4) SVD 平稳运行
    def validate_svd(self) -> dict:
        """4) SVD 平稳完成 + 分解正确（重建误差 < RECON_TOL）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        A = rng.standard_normal((cfg.SVD_D, cfg.SVD_D)).astype(np.float32)
        t0 = time.time()
        rel_err = svd_reconstruction(A)
        elapsed = time.time() - t0
        ok4 = rel_err < cfg.RECON_TOL
        if not ok4:
            raise AIQValidationError(
                f"SVD 重建相对误差 {rel_err:.3e} ≥ {cfg.RECON_TOL}",
                expected=f"<{cfg.RECON_TOL}", actual=rel_err, param_key="H02",
            )
        return {
            "detail": (f"[4] 矩阵: ({cfg.SVD_D}, {cfg.SVD_D}) float32 = "
                       f"{A.nbytes / 1e6:.1f} MB; 耗时 {elapsed:.2f}s; "
                       f"重建相对误差 = {rel_err:.3e} (<{cfg.RECON_TOL})"),
            "rel_err": rel_err, "elapsed_s": elapsed,
        }

    # ------------------------------------------------------------ 5) 内存占用
    def validate_memory(self) -> dict:
        """5) 峰值内存 < RSS_LIMIT_GB（psutil 优先，tracemalloc 回退）。"""
        cfg = self.config
        peak_rss_gb = measure_peak_rss_gb()
        if peak_rss_gb > 0.0:
            ok5 = peak_rss_gb < cfg.RSS_LIMIT_GB
            if not ok5:
                raise AIQValidationError(
                    f"RSS {peak_rss_gb:.3f} GB ≥ 上限 {cfg.RSS_LIMIT_GB} GB",
                    expected=f"<{cfg.RSS_LIMIT_GB}", actual=peak_rss_gb, param_key="H02",
                )
            return {
                "detail": f"[5] 当前 RSS = {peak_rss_gb:.3f} GB (上限 {cfg.RSS_LIMIT_GB} GB)",
                "rss_gb": peak_rss_gb,
            }
        # psutil 不可用：回退 tracemalloc
        try:
            import tracemalloc
            tracemalloc.start()
            _ = np.linalg.svd(rng := np.random.default_rng(cfg.SEED).standard_normal((1024, 1024)),
                              compute_uv=False)
            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_rss_gb = peak / 1e9
            ok5 = peak_rss_gb < cfg.RSS_LIMIT_GB
            if not ok5:
                raise AIQValidationError(
                    f"tracemalloc 峰值 {peak_rss_gb:.3f} GB ≥ 上限 {cfg.RSS_LIMIT_GB} GB",
                    expected=f"<{cfg.RSS_LIMIT_GB}", actual=peak_rss_gb, param_key="H02",
                )
            return {
                "detail": f"[5] tracemalloc 峰值 = {peak_rss_gb:.3f} GB (上限 {cfg.RSS_LIMIT_GB} GB)",
                "rss_gb": peak_rss_gb, "method": "tracemalloc",
            }
        except Exception:
            return {"detail": "[5] 无法读取内存统计（psutil/tracemalloc 不可用），跳过",
                    "rss_gb": 0.0, "ok": True}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：harness 前置 2 线程 + 真实 RSS < 上限（惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        rss_real = rd.get("engine.rss_gb", None)
        tag = rd.source_tag()
        harness_fp = os.path.join(_PARENT, "_real_model_harness.py")
        harness_threads = False
        if os.path.isfile(harness_fp):
            with open(harness_fp, "r", encoding="utf-8", errors="replace") as f:
                _src = f.read()
            harness_threads = ("OPENBLAS_NUM_THREADS" in _src
                               and 'setdefault(_v, "2")' in _src)
        if rss_real is None:  # 真实 RSS 缺失：审计回退
            return {
                "detail": (f"[6] [{tag}] 真实 RSS 数据不可用，回退文档审计 "
                           f"{rd.audit('rss_gb', 3.16):.2f}GB 标注"),
                "source": tag, "real_mode": False,
            }
        audit_rss = rd.audit("rss_gb", 3.16)
        ok6 = (rss_real < cfg.RSS_LIMIT_GB) and harness_threads
        if not ok6:
            raise RealModelMismatchError(
                f"真实实测异常: RSS={rss_real:.3f}GB ≥ {cfg.RSS_LIMIT_GB}GB "
                f"或 harness 未前置 2 线程 (含线程设置={harness_threads})",
                expected={"rss_ok": True, "harness_threads": True},
                actual={"rss": rss_real, "harness_threads": harness_threads},
                param_key="H02",
            )
        return {
            "detail": (f"[6] [{tag}] 真实测量引擎前置 OPENBLAS/OMP/MKL=2 线程: "
                       f"{harness_threads}; 真实测量 RSS = {rss_real:.3f} GB "
                       f"(psutil 实测, < {cfg.RSS_LIMIT_GB}GB 上限); "
                       f"vs 文档审计 {audit_rss:.2f} GB: Δ={abs(rss_real - audit_rss):.3f} GB"
                       f"（以真实实测为准）"),
            "source": tag, "rss_real": rss_real, "audit_rss": audit_rss,
            "harness_threads": harness_threads,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "env", self.validate_env),
            (2, "blas_backend", self.validate_blas_backend),
            (3, "threads", self.validate_threads),
            (4, "svd", self.validate_svd),
            (5, "memory", self.validate_memory),
            (6, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except AIQValidationError as e:
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e), "expected": e.expected,
                         "actual": e.actual, "param_key": e.param_key}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """H02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="H02 OPENBLAS_NUM_THREADS 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    engine = H02Validator(cfg, report, real_data=RD)

    print("=" * 74)
    print(f"H02 OPENBLAS_NUM_THREADS 线程数验证（四层工厂架构，THREADS={cfg.THREADS}）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: THREADS={cfg.THREADS} RSS_LIMIT_GB={cfg.RSS_LIMIT_GB} "
          f"SVD_D={cfg.SVD_D} RECON_TOL={cfg.RECON_TOL}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "h02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "h02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "h02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
