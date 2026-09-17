# -*- coding: utf-8 -*-
"""H05 dtype 模型精度 — float32 数值稳定性、溢出与内存行为验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. float32 vs float64：同一矩阵的 SVD/Gamma/均值计算，相对误差 ~1e-6（ε32 量级）
  2. float16 溢出演示：大数值平方（协方差元素）在 float16 上溢为 inf，float32 有限
  3. 内存计算：0.494B 参数 × 4B/参数 ≈ 1.98GB（与文档 ~2.0GB 一致）；float16 减半
  4. float32 下 Gamma 计算稳定（有限、范围合理）
  5. 真实模型对照：真实权重 float32 内存实测（W=1976131072 B = 1.976 GB）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  H05Config / ConfigFactory / H05DataSynthesizer / H05Validator /
  ReportGenerator / main —— 同 A01（env AIQ_H05_<KEY> 覆盖由共享工厂处理）。
数据源：
  主文档行 5934-6037（H05 dtype 章节，README ⑤ 实测值 6020-6037 行）
  《参数审计与实验报告.txt》行 85（状态=已用）
真实模型对照：
  _real_metrics.json 真实 W（float32 加载实测）与文档估算对照。
说明：H05 固定 float32。float32 单位舍入 ε32=2^-24≈5.96e-8；
      float16 上限 65504，超出即 inf。纯数值合成数据 + 真实实测对照，
      不加载任何大模型。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
import warnings
from typing import Any

import numpy as np

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, RealModelMismatchError, SynthesisError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 H05Config（pydantic 优先；dataclass 回退） ----
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


class H05Config(_ConfigModelBase):
    """H05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 H05 节点键名一一对应（未在 JSON 的字段
    以模型默认值兜底）；取值优先级：环境变量 AIQ_H05_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # 合成数据固定种子（H01 语义）
    DTYPE: str = "float32"           # 模型加载精度（from_pretrained dtype 参数）
    N_PARAMS: float = 494000000.0    # Qwen 0.5B 约 0.494B 参数（README ②）
    MEM_F32_REF_GB: float = 1.976    # float32 内存参考值 ~2.0GB（集中配置用真实实测 1.976GB）
    B: int = 512                     # 合成激活样本数
    D: int = 128                     # 合成激活隐藏维
    BIG_VALUE: float = 1.0e5         # 归一化前激活量级（README ④ 推导 2）
    GAMMA_REL_TOL: float = 1e-5      # Gamma 相对误差阈值（ε32 量级）
    MEAN_REL_TOL: float = 1e-6       # 均值相对误差阈值


# ---- 第二层：配置工厂 ConfigFactory（实例化 H05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """H05 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 H05Config。"""

    def build(self) -> H05Config:
        """构建 H05Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(H05Config, "H05")


# ---------------- 算法逻辑常量 ----------------
EPS = 1e-12    # 除零保护小量


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def gamma_float32(H: np.ndarray) -> float:
    """float32 下 SPL Gamma：中心化后前3奇异值之和/总和。"""
    Hf = H.astype(np.float32)                          # 降精度到 float32
    Hc = Hf - Hf.mean(axis=0, keepdims=True)           # 中心化（float32 内完成）
    S = np.linalg.svd(Hc, compute_uv=False).astype(np.float64)  # 奇异值转回 float64 汇总
    return float(S[:3].sum() / (S.sum() + EPS))        # 前 3 主成分占比，EPS 防除零


def gamma_float64(H: np.ndarray) -> float:
    """float64 参考 Gamma。"""
    Hc = H - H.mean(axis=0, keepdims=True)             # 中心化
    S = np.linalg.svd(Hc, compute_uv=False)            # 奇异值
    return float(S[:3].sum() / (S.sum() + EPS))        # 前 3 主成分占比（参考真值口径）


# ---- 第三层：合成器 H05DataSynthesizer（低秩+噪声激活矩阵） ----
class H05DataSynthesizer(_SynthBase):
    """H05 合成激活生成器：低秩 + 10% 噪声矩阵（float64，固定种子可复现）。"""

    def __init__(self, cfg: H05Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def make_activation(self) -> np.ndarray:
        """生成低秩+噪声合成激活 H（float64，形状 (B,D)）。"""
        cfg = self._cfg
        if cfg.B < 2 or cfg.D < 3:
            raise SynthesisError(
                f"非法的激活规模: B={cfg.B}, D={cfg.D}",
                actual=(cfg.B, cfg.D),
            )
        rng = np.random.default_rng(self._seed)
        U = rng.standard_normal((cfg.B, 3))            # 低秩因子 U
        Vt = rng.standard_normal((cfg.D, 3))           # 低秩因子 V
        H = (U @ Vt.T + 0.1 * rng.standard_normal((cfg.B, cfg.D))).astype(np.float64)
        if not np.isfinite(H).all():
            raise SynthesisError("合成数据含 NaN/Inf", actual=H.shape)
        return H


# ---- 第四层：验证引擎 H05Validator（5 项验证 + 结构化日志 + 类型化异常） ----
class H05ValidationError(AIQValidationError):
    """H05 dtype 精度行为验证失败。"""


class H05Validator(_EngineBase):
    """H05 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL。
    """

    def __init__(
        self,
        config: H05Config,
        synth: H05DataSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data      # 惰性注入（None 时 validate_real_model 方法内 import）
        self.H: np.ndarray | None = None  # 合成激活（步骤 1 生成，供后续复用）
        self.g32: float | None = None      # float32 Gamma（供步骤 4 复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 数值稳定性
    def validate_f32_vs_f64(self) -> dict:
        """1) float32 vs float64 数值稳定性：Gamma/均值相对误差须在 ε32 量级容差内。"""
        cfg = self.config
        assert self.synth is not None
        H = self.synth.make_activation()
        self.H = H
        g32 = gamma_float32(H)
        g64 = gamma_float64(H)
        self.g32 = g32
        rel_gamma = abs(g32 - g64) / abs(g64)          # Gamma 相对误差
        m32 = float(H.astype(np.float32).mean())       # float32 均值
        m64 = float(H.mean())                          # float64 均值
        rel_mean = abs(m32 - m64) / abs(m64)           # 均值相对误差
        ok = bool(rel_gamma < cfg.GAMMA_REL_TOL and rel_mean < cfg.MEAN_REL_TOL)
        if not ok:
            raise H05ValidationError(
                f"[1] 相对误差超阈：rel_gamma={rel_gamma:.2e}, rel_mean={rel_mean:.2e}",
                expected={"gamma": cfg.GAMMA_REL_TOL, "mean": cfg.MEAN_REL_TOL},
                actual={"gamma": rel_gamma, "mean": rel_mean},
                param_key="H05",
            )
        return {
            "detail": (f"[1] float32 vs float64: Gamma f32={g32:.8f} f64={g64:.8f} "
                       f"相对误差={rel_gamma:.2e}; 均值 f32={m32:.8f} f64={m64:.8f} "
                       f"相对误差={rel_mean:.2e} (ε32≈5.96e-8, 期望 ~1e-6)"),
            "rel_gamma": rel_gamma, "rel_mean": rel_mean,
            "g32": g32, "g64": g64,
        }

    # ------------------------------------------------------------ 2) 溢出行为
    def validate_overflow(self) -> dict:
        """2) float16 溢出 vs float32 稳定：协方差元素（大值平方）f16 -> inf。"""
        cfg = self.config
        big = np.float64(cfg.BIG_VALUE)                # 大激活量级（1e5）
        cov_elem_f64 = np.float64(big) * np.float64(big)   # float64 平方：1e10 量级仍有限
        cov_elem_f32 = np.float32(big) * np.float32(big)   # float32 平方：同样有限
        with warnings.catch_warnings():                # 临时捕获 numpy 溢出警告
            warnings.simplefilter("ignore")            # float16 溢出为预期行为
            cov_elem_f16 = np.float16(big) * np.float16(big)  # float16 平方 -> inf
        ok = bool(np.isfinite(cov_elem_f32)) and bool(not np.isfinite(cov_elem_f16))
        if not ok:
            raise H05ValidationError(
                f"[2] 溢出行为不符：f32有限={np.isfinite(cov_elem_f32)}, "
                f"f16有限={np.isfinite(cov_elem_f16)}",
                expected={"f32_finite": True, "f16_finite": False},
                actual={"f32_finite": bool(np.isfinite(cov_elem_f32)),
                        "f16_finite": bool(np.isfinite(cov_elem_f16))},
                param_key="H05",
            )
        return {
            "detail": (f"[2] 大值 {big:.0e} 的平方: f64={cov_elem_f64:.3e}(有限), "
                       f"f32={float(cov_elem_f32):.3e}(有限), "
                       f"f16={float(cov_elem_f16)}(inf/溢出) -> float16 溢出为 inf, "
                       f"float32 保持有限"),
            "f16_overflows": True,
        }

    # ------------------------------------------------------------ 3) 内存占用
    def validate_memory(self) -> dict:
        """3) 内存占用：float32 全参内存 ≈ 文档参考值（5% 内）。"""
        cfg = self.config
        mem_f32 = cfg.N_PARAMS * 4 / 1e9               # float32 全参内存（GB）
        ok = bool(np.isclose(mem_f32, cfg.MEM_F32_REF_GB, rtol=0.05))
        if not ok:
            raise H05ValidationError(
                f"[3] float32 内存 {mem_f32:.2f} GB 与文档 {cfg.MEM_F32_REF_GB}GB 偏差超 5%",
                expected=cfg.MEM_F32_REF_GB, actual=mem_f32, param_key="H05",
            )
        demos = "; ".join(
            f"{name} {bpp} B/参数 -> {cfg.N_PARAMS * bpp / 1e9:.2f} GB"
            for name, bpp in [("float32", 4), ("float16", 2), ("int8", 1)]
        )
        return {"detail": (f"[3] 内存占用（Qwen 0.5B ≈ {cfg.N_PARAMS:.3e} 参数）: "
                           f"{demos}; float32 ≈ {mem_f32:.2f} GB 与文档一致"),
                "mem_f32_gb": mem_f32}

    # ------------------------------------------------------------ 4) Gamma 稳定性
    def validate_gamma_stability(self) -> dict:
        """4) float32 下 Gamma 计算稳定性：有限且 ∈(0,1)。"""
        cfg = self.config
        g = self.g32 if self.g32 is not None else gamma_float32(self.H)
        ok = bool(np.isfinite(g)) and 0.0 < g < 1.0
        if not ok:
            raise H05ValidationError(
                f"[4] Gamma(float32)={g:.6f} 非有限或越界",
                expected="finite and in (0,1)", actual=g, param_key="H05",
            )
        return {"detail": (f"[4] float32 下 Gamma 计算稳定性: Gamma(float32) = {g:.6f}, "
                           f"有限且 ∈(0,1)"), "gamma": g}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实权重 float32 内存实测 vs 文档估算。"""
        cfg = self.config
        rd = self._get_real_data()
        tag = rd.source_tag()
        est_gb = cfg.N_PARAMS * 4 / 1e9                # 文档估算值
        W_real = rd.get("arch.W_bytes", None)          # 真实权重字节数
        if W_real is None:                             # 真实权重缺失：回退文档估算标注
            return {"detail": (f"[5] [{tag}] 真实 W 数据不可用，回退文档估算 "
                               f"{est_gb:.2f}GB 标注"), "fallback": True}
        w_f32_gb = W_real / 1e9                        # 真实权重内存（GB）
        w_f16_gb = w_f32_gb / 2.0                      # float16 减半演示
        rel_dev = abs(w_f32_gb - est_gb) / est_gb      # 实测与估算相对偏差
        ok = bool(np.isclose(w_f32_gb, cfg.MEM_F32_REF_GB, rtol=0.05)) and rel_dev < 0.05
        if not ok:
            raise RealModelMismatchError(
                f"[5] 真实 W 内存异常: 实测 {w_f32_gb:.3f}GB vs 估算 {est_gb:.3f}GB "
                f"(偏差 {rel_dev * 100:.2f}%)",
                expected=cfg.MEM_F32_REF_GB, actual=w_f32_gb, param_key="H05",
            )
        return {
            "detail": (f"[5] [{tag}] 真实模型 fp32 实测 W = {W_real} B = {w_f32_gb:.3f} GB; "
                       f"vs 文档估算 {est_gb:.3f} GB，相对偏差 {rel_dev * 100:.3f}%; "
                       f"float16 减半 {w_f16_gb:.3f} GB（对比演示）"),
            "source": tag, "w_real_bytes": int(W_real), "w_f32_gb": w_f32_gb,
            "rel_dev": rel_dev,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "f32_vs_f64", self.validate_f32_vs_f64),
            (2, "overflow", self.validate_overflow),
            (3, "memory", self.validate_memory),
            (4, "gamma_stability", self.validate_gamma_stability),
            (5, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except AIQValidationError as e:  # 类型化异常：携带 expected/actual
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
    """H05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="H05 dtype 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = H05DataSynthesizer(cfg)                   # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = H05Validator(cfg, synth, report, real_data=RD)  # ③ 验证层

    print("=" * 74)
    print("H05 dtype 精度行为验证（四层工厂架构，固定 float32，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: DTYPE={cfg.DTYPE} N_PARAMS={cfg.N_PARAMS:.3e} "
          f"MEM_F32_REF_GB={cfg.MEM_F32_REF_GB} B={cfg.B} D={cfg.D}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "h05_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "h05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "h05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
