# -*- coding: utf-8 -*-
"""G04 target_metric 参考 Gamma — 论文值 0.9966 的复现与架构判定验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 复现 Gamma 计算机制（前3主成分奇异值占比，SPL 口径 + 能量口径）
  2. Post-Norm 型（近 3 维低秩 + 极弱噪声）数据 -> Gamma 接近论文值 0.9966
  3. Pre-Norm 型（各向同性弥散）数据 -> Gamma 远低于 target（约 3/d）
  4. 架构类型判定器（>0.90 / >0.50 / >0.20）对论文实测值表逐一判定正确
  5. 合成 Post-Norm Gamma 与 target 0.9966 的相对差距标注 + 真实模型对照
数据源：
  主文档行 5311-5420（G04 target_metric 章节，README ⑤ 对照表 5385-5394 行）
  源码 spectral_analysis.py（SPL Gamma 实现）
  《参数审计与实验报告.txt》行 79（状态=对照）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  G04Config / ConfigFactory / G04TargetSynthesizer / G04Validator /
  ReportGenerator / main —— 同 G01（env AIQ_G04_<KEY> 覆盖由共享工厂处理）。
说明：纯数值合成数据（算法逻辑验证）+ 真实实测值对照，不加载任何大模型。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
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

RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 G04Config（pydantic 优先；dataclass 回退） ----
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


class G04Config(_ConfigModelBase):
    """G04 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 G04 节点键名一一对应；取值优先级：
    环境变量 AIQ_G04_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                  # 合成数据固定种子（规范 §3 / H01 语义）
    TARGET: float = 0.9966         # 论文参考 Gamma（README ①）
    B: int = 1000                  # 合成规模（token 数）
    D: int = 128                   # 合成规模（隐藏维）
    NOISE_POST: float = 0.02       # Post-Norm 极弱噪声强度
    SCALE_PRE: float = 0.1         # Pre-Norm 弥散缩放
    REL_TOL: float = 0.05          # 相对差距阈值 <5%
    ARCH_POST: float = 0.90        # Post-Norm 凝聚型下限（TH_CONC）
    ARCH_MIX: float = 0.50         # 混合/早期架构下限（TH_MIX）
    ARCH_PRE: float = 0.20         # Pre-Norm 弥散型下限（TH_SPREAD）
    EPS: float = 1e-12             # 除零保护小量


# ---- 第二层：配置工厂 ConfigFactory（实例化 G04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """G04 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 G04Config。"""

    def build(self) -> G04Config:
        """构建 G04Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(G04Config, "G04")


# ---- 第三层：合成器 G04TargetSynthesizer（Post/Pre-Norm 数据生成） ----
class G04TargetSynthesizer(_SynthBase):
    """G04 数据合成器：Post-Norm（3 维低秩+弱噪声）与 Pre-Norm（各向同性）两种激活。"""

    def __init__(self, cfg: G04Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def make_post(self) -> np.ndarray:
        """Post-Norm 型：3 维低秩信号 + 极弱噪声（近 3 维低秩，Gamma 接近 1）。"""
        cfg = self._cfg
        rng = np.random.default_rng(self._seed)
        U = rng.standard_normal((cfg.B, 3))
        Vt = rng.standard_normal((cfg.D, 3))
        signal = U @ Vt.T
        noise_pn = cfg.NOISE_POST * rng.standard_normal((cfg.B, cfg.D))
        H = signal + noise_pn
        if H.shape != (cfg.B, cfg.D) or not np.isfinite(H).all():
            raise SynthesisError("Post-Norm 合成数据异常", actual=H.shape)
        return H

    def make_pre(self) -> np.ndarray:
        """Pre-Norm 型：各向同性高斯弥散数据。"""
        cfg = self._cfg
        rng = np.random.default_rng(self._seed)
        return rng.standard_normal((cfg.B, cfg.D)) * cfg.SCALE_PRE


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def gamma_svd_squared(H: np.ndarray, eps: float) -> float:
    """能量口径 Gamma = (s1^2+s2^2+s3^2)/sum(s^2)（公式规格书 101 项）。"""
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2 and H.shape[0] >= 2, f"H 需二维且 B>=2，实际 {H.shape}"
    assert np.isfinite(H).all(), "H 含 NaN/Inf"
    Hc = H - H.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Hc, compute_uv=False)
    S2 = S ** 2
    return float(S2[:3].sum() / (S2.sum() + eps))


def gamma_svd(H: np.ndarray, eps: float) -> float:
    """SPL 源码口径 Gamma = (s1+s2+s3)/sum(s)（spectral_analysis.py）。"""
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2 and H.shape[0] >= 2, f"H 需二维且 B>=2，实际 {H.shape}"
    Hc = H - H.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Hc, compute_uv=False)
    return float(S[:3].sum() / (S.sum() + eps))


def classify_arch(gamma: float, arch_post: float, arch_mix: float, arch_pre: float) -> str:
    """架构类型判定（主文档第 5373-5383 行 / README ② 公式）。"""
    assert np.isfinite(gamma), f"Gamma={gamma} 非有限"
    assert 0.0 <= gamma <= 1.0, f"Gamma={gamma} 越界，要求∈[0,1]"
    if gamma > arch_post:
        return "Post-Norm凝聚型"
    if gamma > arch_mix:
        return "混合/早期架构"
    if gamma > arch_pre:
        return "Pre-Norm弥散型"
    return "高度弥散型"


# ---- 第四层：验证引擎 G04Validator（5 项验证 + 结构化日志 + 类型化异常） ----
class G04Validator(_EngineBase):
    """G04 验证引擎：顺序执行 5 项验证（结构化 JSON 日志 + 类型化异常）。"""

    def __init__(
        self,
        config: G04Config,
        synth: G04TargetSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.H_post: np.ndarray | None = None  # Post-Norm 数据（共享中间结果）
        self.H_pre: np.ndarray | None = None   # Pre-Norm 数据

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) Post-Norm 复现
    def validate_post_norm(self) -> dict:
        """1) Post-Norm 型数据 -> Gamma 接近论文值 TARGET（相对差距 < REL_TOL）。"""
        cfg = self.config
        H_post = self.synth.make_post()
        self.H_post = H_post
        g_post_e = gamma_svd_squared(H_post, cfg.EPS)
        g_post_s = gamma_svd(H_post, cfg.EPS)
        rel_err = abs(g_post_e - cfg.TARGET) / cfg.TARGET
        ok1 = rel_err < cfg.REL_TOL
        if not ok1:
            raise AIQValidationError(
                f"Post-Norm Gamma={g_post_e:.6f} 与 target={cfg.TARGET} 相对差距 "
                f"{rel_err * 100:.2f}% ≥5%",
                expected=f"<{cfg.REL_TOL * 100:.0f}%", actual=rel_err, param_key="G04",
            )
        return {
            "detail": (f"[1] Post-Norm 型合成数据 (3维低秩+{cfg.NOISE_POST * 100:.0f}%噪声): "
                       f"Gamma(能量口径) = {g_post_e:.6f} vs 论文 target={cfg.TARGET}; "
                       f"Gamma(SPL 口径) = {g_post_s:.6f}; 相对差距 = {rel_err * 100:.2f}% (<5%)"),
            "gamma_e": g_post_e, "gamma_s": g_post_s, "rel_err": rel_err,
        }

    # ------------------------------------------------------------ 2) Pre-Norm 弥散
    def validate_pre_norm(self) -> dict:
        """2) Pre-Norm 型数据 -> Gamma 远低于 target（落入弥散区间 < ARCH_PRE）。"""
        cfg = self.config
        H_pre = self.synth.make_pre()
        self.H_pre = H_pre
        g_pre = gamma_svd_squared(H_pre, cfg.EPS)
        theo_pre = 3.0 / cfg.D
        ok2 = g_pre < cfg.ARCH_PRE
        if not ok2:
            raise AIQValidationError(
                f"Pre-Norm Gamma={g_pre:.6f} 未落入弥散区间 (<{cfg.ARCH_PRE})",
                expected=f"<{cfg.ARCH_PRE}", actual=g_pre, param_key="G04",
            )
        return {
            "detail": (f"[2] Pre-Norm 型合成数据 (各向同性): Gamma(能量口径) = {g_pre:.6f} "
                       f"(理论值约 3/d={theo_pre:.6f}); << target {cfg.TARGET}, "
                       f"差值 = {cfg.TARGET - g_pre:.4f}"),
            "gamma_pre": g_pre, "theo_pre": theo_pre,
        }

    # ------------------------------------------------------------ 3) 论文实测对照表
    def validate_paper_table(self) -> dict:
        """3) 论文实测对照表（主文档第 5385-5394 行）架构判定逐项正确。"""
        cfg = self.config
        ref = [
            ("ResNet", 0.9992, "Post-Norm凝聚型"),
            ("target_metric", cfg.TARGET, "Post-Norm凝聚型"),
            ("GPT-2", 0.457, "Pre-Norm弥散型"),
            ("Qwen 0.5B", 0.1625, "高度弥散型"),
            ("RWKV-5", 0.1237, "高度弥散型"),
        ]
        parts = []
        ok3 = True
        for name, g, expect in ref:
            got = classify_arch(g, cfg.ARCH_POST, cfg.ARCH_MIX, cfg.ARCH_PRE)
            ok = (got == expect)
            if name == "GPT-2":
                ok = (g > cfg.ARCH_PRE) and (g <= cfg.ARCH_MIX) and got == "Pre-Norm弥散型"
            if name in ("Qwen 0.5B", "RWKV-5"):
                ok = (g <= cfg.ARCH_PRE) and got == "高度弥散型"
            ok3 = ok3 and ok
            parts.append(f"{name}: Gamma={g:.4f} -> {got}")
            if not ok:
                raise AIQValidationError(
                    f"模型 {name} Gamma={g:.4f} 判定 {got} ≠ 期望 {expect}",
                    expected=expect, actual=got, param_key="G04",
                )
        return {"detail": "[3] 论文实测对照表判定: " + "; ".join(parts), "n_models": len(ref), "ok": ok3}

    # ------------------------------------------------------------ 4) 门控方向性
    def validate_gating_direction(self) -> dict:
        """4) 门控向 target 逼近的方向性（G01 联动）：完全投影 -> Gamma 凝聚化（>0.9）。"""
        cfg = self.config
        assert self.H_post is not None and self.H_pre is not None
        g0 = gamma_svd_squared(self.H_post, cfg.EPS)
        Hc = self.H_pre - self.H_pre.mean(axis=0, keepdims=True)
        _, _, Vt2 = np.linalg.svd(Hc, full_matrices=False)
        V = Vt2[:2].T
        Pi = self.H_pre.mean(axis=0, keepdims=True) + (Hc @ V) @ V.T
        g_gated = gamma_svd_squared(Pi, cfg.EPS)
        ok4 = g_gated > cfg.ARCH_POST
        if not ok4:
            raise AIQValidationError(
                f"完全投影后 Gamma={g_gated:.4f} 未凝聚化 (>0.9)",
                expected=f">{cfg.ARCH_POST}", actual=g_gated, param_key="G04",
            )
        return {
            "detail": (f"[4] Post-Norm 基线 Gamma = {g0:.6f}（已接近 target {cfg.TARGET}）; "
                       f"Pre-Norm 数据完全投影到 C-子空间后 Gamma = {g_gated:.4f} (>0.9, 凝聚化)"),
            "g0": g0, "g_gated": g_gated,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实 k_proj Gamma vs 论文 target（惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        real_gamma = rd.get("spectral.k_proj_gamma_mean", None)
        tag = rd.source_tag()
        if real_gamma is None:  # 真实值缺失：回退审计值标注
            audit_gamma = rd.audit("k_proj_gamma_mean", 0.625)
            return {
                "detail": (f"[5] [{tag}] 真实 Gamma 不可用，使用审计值 {audit_gamma:.4f} 回退标注"),
                "source": tag, "real_gamma": None, "audit_gamma": audit_gamma,
            }
        got_real = classify_arch(float(real_gamma), cfg.ARCH_POST, cfg.ARCH_MIX, cfg.ARCH_PRE)
        audit_gamma = rd.audit("k_proj_gamma_mean", 0.625)
        ok5 = 0.0 < real_gamma < 1.0 and got_real == "Pre-Norm弥散型"
        if not ok5:
            raise RealModelMismatchError(
                f"真实 Gamma={real_gamma:.4f} 判定 {got_real} 异常，"
                f"期望落入 Pre-Norm 弥散区间 (0.2,0.5]",
                expected="Pre-Norm弥散型", actual=got_real, param_key="G04",
            )
        return {
            "detail": (f"[5] [{tag}] 真实 k_proj Gamma 均值 = {real_gamma:.4f}; "
                       f"vs 论文 target {cfg.TARGET}: 差距 Δ={cfg.TARGET - real_gamma:.4f} "
                       f"(= {abs(cfg.TARGET - real_gamma) / cfg.TARGET * 100:.1f}% 相对); "
                       f"架构判定: {got_real} (0.2<γ≤0.5); "
                       f"vs 文档审计 {audit_gamma:.4f}: Δ={abs(real_gamma - audit_gamma):.4f} "
                       f"（以真实实测为准）; 差异说明：预训练大模型（Pre-Norm、非残差 toy 结构）"
                       f"并非 Post-Norm 强低秩，论文值 0.9966 来自理想化架构设定"),
            "source": tag, "real_gamma": real_gamma, "audit_gamma": audit_gamma,
            "got_real": got_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "post_norm", self.validate_post_norm),
            (2, "pre_norm", self.validate_pre_norm),
            (3, "paper_table", self.validate_paper_table),
            (4, "gating_direction", self.validate_gating_direction),
            (5, "real_model", self.validate_real_model),
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
    """G04 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="G04 target_metric 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    synth = G04TargetSynthesizer(cfg)
    report = ReportGenerator()
    engine = G04Validator(cfg, synth, report, real_data=RD)

    print("=" * 74)
    print(f"G04 target_metric 参考 Gamma 验证（四层工厂架构，论文值 {cfg.TARGET}）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: B={cfg.B} D={cfg.D} TARGET={cfg.TARGET} NOISE_POST={cfg.NOISE_POST} "
          f"REL_TOL={cfg.REL_TOL}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "g04_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "g04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "g04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
