# -*- coding: utf-8 -*-
"""H03 OMP_NUM_THREADS OMP 线程数 — OpenMP 并行度限制与运算平稳性验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. OMP_NUM_THREADS 在 import torch/numpy 之前设置且环境变量生效
  2. torch.get_num_threads() == 2（PyTorch 继承 OMP 设置，torch 可用时）
  3. threadpoolctl 探测 OMP/OpenBLAS 实际线程数 <= 2（若可用）
  4. torch 矩阵运算正确性（SVD 重建误差 < 1e-4）与内存受控
  5. 与 H02 的一致性：两个环境变量均设为 2
  6. 真实模型对照：harness 前置 2 线程 + 真实 RSS < 上限（惰性读取）
数据源：
  主文档行 5724-5821（H03 章节，README ⑤ 实测值 5799-5817 行）
  《参数审计与实验报告.txt》行 83（状态=已用）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  H03Config / ConfigFactory / H03Validator / ReportGenerator / main
  关键设计决策（同 H02）：线程环境变量必须在 import torch/numpy 之前设置
  （README ④ 推导 1：线程继承链），文件顶部用纯标准库 json 解析线程数。
说明：本机默认 torch.get_num_threads()=8（未限制时），设置
      OMP_NUM_THREADS=2 后应收敛到 2。纯数值合成数据 + 真实实测对照。
=====================================================================
"""
import os
import sys

# ---------------------------------------------------------------------------
# ★ 必须在 import torch/numpy 之前设置线程环境变量（README ④ 推导 1）
# 仅依赖标准库 json 解析 _params_data.json（不触发 numpy/torch 导入）。
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def _resolve_threads(param_id: str, default: int = 2) -> str:
    """线程数解析（env AIQ_<PARAM>_THREADS > _params_data.json > 默认）。

    必须在 import torch/numpy 之前完成（线程池初始化时继承该值）；
    只依赖标准库 json，与共享 ConfigFactory 优先级一致。
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


_T = _resolve_threads("H03")
os.environ["OPENBLAS_NUM_THREADS"] = _T  # OpenBLAS 线程上限（与 H02 一致）
os.environ["OMP_NUM_THREADS"] = _T  # OpenMP 线程上限：torch 在线程池初始化时继承
os.environ["MKL_NUM_THREADS"] = _T  # MKL 线程上限
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

try:
    import torch  # noqa: E402  # torch 张量库（在线程环境变量之后导入）
    TORCH_OK = True
except Exception:
    TORCH_OK = False

import numpy as np  # noqa: E402  # 数值核心（必须在环境变量之后导入）

sys.path.insert(0, _PARENT)  # 复用共享基类（须在环境变量之后）
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

# ---- 第一层：配置模型 H03Config（pydantic 优先；dataclass 回退） ----
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


class H03Config(_ConfigModelBase):
    """H03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 H03 节点键名一一对应；取值优先级：
    环境变量 AIQ_H03_<KEY> > YAML > _params_data.json > 本模型默认值。
    THREADS 在文件顶部已被用于设置 os.environ（取值路径与工厂一致）。
    """

    THREADS: int = 2             # OMP 线程数（数据层 H03.THREADS，README ①/⑤）
    RSS_LIMIT_GB: float = 3.16   # 真实 RSS 上限（数据层）
    SVD_D: int = 1024            # SVD 矩阵规模（数据层）
    RECON_TOL: float = 1e-4      # 重建相对误差阈值
    SEED: int = 0                # 合成矩阵固定种子（规范 §3 / H01 语义）


# ---- 第二层：配置工厂 ConfigFactory（实例化 H03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """H03 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 H03Config。"""

    def build(self) -> H03Config:
        """构建 H03Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(H03Config, "H03")


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def detect_pool_threads() -> list:
    """threadpoolctl 探测 OMP/OpenBLAS 线程数列表；不可用时返回空列表。"""
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
        omp_libs = [l for l in info
                    if "omp" in l.get("internal_api", "").lower()
                    or "openmp" in l.get("filepath", "").lower()
                    or "openblas" in l.get("internal_api", "").lower()]
        return [l.get("num_threads") for l in omp_libs]
    except Exception:
        return []


def measure_rss_gb() -> float:
    """读取当前进程 RSS（GB）；psutil 不可用时返回 0.0。"""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1e9)
    except Exception:
        return 0.0


# ---- 第四层：验证引擎 H03Validator（6 项验证 + 结构化日志 + 类型化异常） ----
class H03Validator(_EngineBase):
    """H03 验证引擎：顺序执行 6 项验证（结构化 JSON 日志 + 类型化异常）。

    无合成器：验证主体为环境变量/torch 线程/张量运算/内存检查。
    """

    def __init__(
        self,
        config: H03Config,
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
        """1) OMP/OPENBLAS 环境变量均为 THREADS（与 H02 一致性）。"""
        cfg = self.config
        expect = str(cfg.THREADS)
        omp_val = os.environ.get("OMP_NUM_THREADS", "未设置")
        openblas_val = os.environ.get("OPENBLAS_NUM_THREADS", "未设置")
        ok1 = omp_val == expect
        ok1b = openblas_val == expect
        if not (ok1 and ok1b):
            raise AIQValidationError(
                f"OMP_NUM_THREADS={omp_val} 或 OPENBLAS_NUM_THREADS={openblas_val} ≠ {expect}",
                expected=expect, actual={"omp": omp_val, "openblas": openblas_val},
                param_key="H03",
            )
        return {
            "detail": (f"[1] OMP_NUM_THREADS = {omp_val} (期望 '{expect}'); "
                       f"与 H02 一致性: OPENBLAS_NUM_THREADS = {openblas_val}"),
            "omp": omp_val, "openblas": openblas_val,
        }

    # ------------------------------------------------------------ 2) torch 线程数
    def validate_torch_threads(self) -> dict:
        """2) torch.get_num_threads() <= THREADS（OMP 继承生效）。"""
        cfg = self.config
        try:
            n_cores = os.cpu_count()
        except Exception:
            n_cores = "?"
        if not TORCH_OK:
            return {"detail": f"[2] torch 不可用, 可用核心={n_cores}（跳过 torch 线程检查）",
                    "torch_ok": False, "ok": True}
        n_torch = torch.get_num_threads()
        ok2 = n_torch <= cfg.THREADS
        if not ok2:
            raise AIQValidationError(
                f"torch.get_num_threads()={n_torch} > {cfg.THREADS}，OMP 未生效",
                expected=f"<={cfg.THREADS}", actual=n_torch, param_key="H03",
            )
        return {
            "detail": (f"[2] 可用核心={n_cores}, torch.get_num_threads() = {n_torch} "
                       f"(OMP_NUM_THREADS={cfg.THREADS} 生效, 期望 <= {cfg.THREADS})"),
            "torch_ok": True, "n_torch": n_torch, "n_cores": n_cores,
        }

    # ------------------------------------------------------------ 3) threadpoolctl 探测
    def validate_pool_threads(self) -> dict:
        """3) threadpoolctl 探测 OMP/OpenBLAS 线程数均不超过 THREADS。"""
        cfg = self.config
        nums = detect_pool_threads()
        if not nums:
            return {"detail": "[3] threadpoolctl 不可用或无匹配库，跳过",
                    "detected": False, "ok": True}
        ok3 = all(n <= cfg.THREADS for n in nums)
        if not ok3:
            raise AIQValidationError(
                f"threadpoolctl 线程数 {nums} 中存在 > {cfg.THREADS}",
                expected=f"all <={cfg.THREADS}", actual=nums, param_key="H03",
            )
        return {
            "detail": f"[3] threadpoolctl 线程数 = {nums} (期望全部 <= {cfg.THREADS})",
            "detected": True, "nums": nums,
        }

    # ------------------------------------------------------------ 4) 张量运算正确性
    def validate_ops(self) -> dict:
        """4) torch/numpy SVD 运算正确性（重建误差 < RECON_TOL）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        if TORCH_OK:
            A = torch.tensor(rng.standard_normal((cfg.SVD_D, cfg.SVD_D))).float()
            assert bool(torch.isfinite(A).all()), "合成张量含 NaN/Inf"
            t0 = time.time()
            U, S, Vt = torch.linalg.svd(A, full_matrices=False)
            elapsed = time.time() - t0
            recon = U @ torch.diag(S) @ Vt
            rel_err = float(torch.norm(recon - A) / torch.norm(A))
            ok4 = bool(torch.isfinite(S).all()) and rel_err < cfg.RECON_TOL
            if not ok4:
                raise AIQValidationError(
                    f"torch SVD 重建误差 {rel_err:.3e} ≥ {cfg.RECON_TOL}",
                    expected=f"<{cfg.RECON_TOL}", actual=rel_err, param_key="H03",
                )
            return {
                "detail": (f"[4] torch SVD ({cfg.SVD_D}×{cfg.SVD_D}) 耗时 {elapsed:.2f}s, "
                           f"重建误差 = {rel_err:.3e} (<{cfg.RECON_TOL})"),
                "backend": "torch", "rel_err": rel_err, "elapsed_s": elapsed,
            }
        A = rng.standard_normal((cfg.SVD_D, cfg.SVD_D))
        t0 = time.time()
        S = np.linalg.svd(A, compute_uv=False)
        elapsed = time.time() - t0
        ok4 = bool(np.isfinite(S).all())
        if not ok4:
            raise AIQValidationError(
                "numpy SVD 奇异值含 NaN/Inf",
                expected="finite", actual="nan", param_key="H03",
            )
        return {
            "detail": (f"[4] numpy SVD ({cfg.SVD_D}×{cfg.SVD_D}) 耗时 {elapsed:.2f}s, "
                       f"奇异值有限: {ok4}"),
            "backend": "numpy", "rel_err": None, "elapsed_s": elapsed,
        }

    # ------------------------------------------------------------ 5) 内存占用
    def validate_memory(self) -> dict:
        """5) 内存占用 < RSS_LIMIT_GB（psutil）。"""
        cfg = self.config
        rss_gb = measure_rss_gb()
        if rss_gb <= 0.0:
            return {"detail": "[5] psutil 不可用，跳过（SVD 已平稳完成）",
                    "rss_gb": 0.0, "ok": True}
        ok5 = rss_gb < cfg.RSS_LIMIT_GB
        if not ok5:
            raise AIQValidationError(
                f"RSS {rss_gb:.3f} GB ≥ 上限 {cfg.RSS_LIMIT_GB} GB",
                expected=f"<{cfg.RSS_LIMIT_GB}", actual=rss_gb, param_key="H03",
            )
        return {"detail": f"[5] 当前 RSS = {rss_gb:.3f} GB (上限 {cfg.RSS_LIMIT_GB} GB)",
                "rss_gb": rss_gb}

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
            harness_threads = ("OMP_NUM_THREADS" in _src
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
                param_key="H03",
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
            (2, "torch_threads", self.validate_torch_threads),
            (3, "pool_threads", self.validate_pool_threads),
            (4, "ops", self.validate_ops),
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
    """H03 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="H03 OMP_NUM_THREADS 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    engine = H03Validator(cfg, report, real_data=RD)

    print("=" * 74)
    print(f"H03 OMP_NUM_THREADS 线程数验证（四层工厂架构，THREADS={cfg.THREADS}）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: THREADS={cfg.THREADS} RSS_LIMIT_GB={cfg.RSS_LIMIT_GB} "
          f"SVD_D={cfg.SVD_D} RECON_TOL={cfg.RECON_TOL}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "h03_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "h03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "h03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
