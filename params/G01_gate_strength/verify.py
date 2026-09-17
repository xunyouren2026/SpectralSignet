# -*- coding: utf-8 -*-
"""G01 gate_strength 谱门控强度 — 谱门控公式 H_gated = H + s·(Π_C H − H) 验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. s=0   : H_gated == H（恒等映射，无操作）
  2. s=0.3 : 插值性质 ||H_gated − Π_C H||_F = (1−s)·||H − Π_C H||_F，Gamma 提升
  3. s=1   : H_gated == Π_C H（完全投影），且残差 H − Π_C H 与主方向 v1,v2 正交
  4. Gamma（前 3 主成分能量占比）随 s 单调不减
  5. 真实模型对照（_real_metrics.json k_proj Gamma 均值 0.4695，惰性读取）
数据源：
  主文档行 4851-4956（G01 gate_strength 章节，README ⑤ 实测表）
  源码 spectral_gating.py（project_to_c_subspace 第 44-79 行、gate 第 81-86 行）
  《参数审计与实验报告.txt》行 76（状态=理论）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  G01Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 G01Config（env AIQ_G01_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  GateSynthesizer        —— 合成激活矩阵生成（synth_activation，固定种子）
  G01Validator           —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用基类）
  main()                 —— 仅编排 cfg→synth→engine→report

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

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 G01Config（pydantic 优先；dataclass 回退） ----
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


class G01Config(_ConfigModelBase):
    """G01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 G01 节点键名一一对应（SEED/N_COMPONENTS/
    EPS/RTOL/ATOL/B/D 为算法逻辑常量兜底）；取值优先级：
    环境变量 AIQ_G01_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0              # 合成数据固定种子（规范 §3 / H01 语义）
    N_COMPONENTS: int = 3      # C-子空间维数 M=3：均值 + 前 2 个主方向（README ② / B01）
    EPS: float = 1e-12         # 除零保护小量：防 Gamma 分母为零
    RTOL: float = 1e-6         # 相对容差（np.isclose 默认）
    ATOL: float = 1e-9         # 绝对容差：极小数值比较
    B: int = 200               # 合成激活 token 数
    D: int = 64                # 合成激活隐藏维
    GATE_STRENGTH: float = 0.3  # 门控强度默认档位（README ②）
    S_VALUES: list = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]  # 门控强度扫描档位（README ③）


# ---- 第二层：配置工厂 ConfigFactory（实例化 G01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """G01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 G01Config。"""

    def build(self) -> G01Config:
        """构建 G01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(G01Config, "G01")


# ---- 第三层：合成器 GateSynthesizer（合成激活矩阵生成） ----
class GateSynthesizer(_SynthBase):
    """G01 激活矩阵合成器：生成低秩信号 + 噪声的激活矩阵（固定种子可复现）。"""

    def __init__(self, cfg: G01Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_activation(self) -> np.ndarray:
        """合成激活：强低秩（C-子空间承载大量能量）+ 各向同性噪声。"""
        cfg = self._cfg
        rng = np.random.default_rng(self._seed)
        # 低秩信号：U(B×3)·V(3×D) 构成 C-子空间的主体能量；5% 量级噪声模拟弥散分量
        U = rng.standard_normal((cfg.B, 3))
        Vtrue = rng.standard_normal((cfg.D, 3))
        signal = U @ Vtrue.T
        noise = 0.05 * rng.standard_normal((cfg.B, cfg.D))
        H = signal + noise
        if H.shape != (cfg.B, cfg.D) or not np.isfinite(H).all():
            raise SynthesisError("合成数据异常（NaN/Inf 或形状不符）", actual=H.shape)
        return H


# ---------------- 纯函数工具（与验证逻辑解耦，与原脚本逐项一致） ----------------
def project_to_c_subspace(H: np.ndarray, n_components: int, eps: float) -> np.ndarray:
    """numpy 版 C-子空间投影：均值 + 前 n_components-1 个主方向。

    与 spectral_gating.py project_to_c_subspace 一致（README ② 公式）。
    防御：二维检查、NaN/Inf 检查、空/退化输入（B<2 或 k<=0）回退。
    """
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2, f"H 必须为二维 (B,d)，实际 ndim={H.ndim}"
    assert np.isfinite(H).all(), "H 含 NaN/Inf，投影结果不可信"
    B = H.shape[0]
    d = H.shape[1]
    if B < 2:
        return H
    mu = H.mean(axis=0, keepdims=True)
    Hc = H - mu
    k = min(n_components - 1, B, d)
    if k <= 0:
        return np.broadcast_to(mu, H.shape).copy()
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    V = Vt[:k].T
    proj = (Hc @ V) @ V.T
    return mu + proj


def gate(H: np.ndarray, s: float, n_components: int) -> np.ndarray:
    """谱门控：H_gated = H + s·(Π_C H − H)（README ② 核心公式）。

    s<=0 时返回原 H（恒等映射）；s>=1 时退化为完全投影（浮点近似）。
    """
    H = np.asarray(H, dtype=np.float64)
    assert 0.0 <= s <= 1.0, f"门控强度 s={s} 越界，要求 s∈[0,1]"
    if s <= 0.0:
        return H
    C_sub = project_to_c_subspace(H, n_components=n_components, eps=1e-12)
    return H + s * (C_sub - H)


def gamma_svd(H: np.ndarray, eps: float) -> float:
    """SPL Gamma 口径（spectral_analysis.py）：中心化后奇异值前3之和/总和。"""
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2 and H.shape[0] >= 2, f"H 需二维且 B>=2，实际 {H.shape}"
    Hc = H - H.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Hc, compute_uv=False)
    return float(S[:3].sum() / (S.sum() + eps))


def gamma_energy(H: np.ndarray, eps: float) -> float:
    """能量口径 Gamma = (s1^2+s2^2+s3^2)/sum(s^2)（公式规格书 101 项）。"""
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2 and H.shape[0] >= 2, f"H 需二维且 B>=2，实际 {H.shape}"
    Hc = H - H.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Hc, compute_uv=False)
    S2 = S ** 2
    return float(S2[:3].sum() / (S2.sum() + eps))


# ---- 第四层：验证引擎 G01Validator（5 项验证 + 结构化日志 + 类型化异常） ----
class G01Validator(_EngineBase):
    """G01 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性注入（None 时 validate_real_model 方法内 import）。
    """

    def __init__(
        self,
        config: G01Config,
        synth: GateSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.H: np.ndarray | None = None    # 合成激活矩阵（共享中间结果）
        self.C_sub: np.ndarray | None = None  # Π_C H 投影（供后续步骤复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) s=0 恒等
    def validate_s0_identity(self) -> dict:
        """1) s=0：H_gated == H（恒等映射，无操作）。"""
        cfg = self.config
        H = self.synth.synth_activation()  # 合成激活（首次在此生成并落盘共享）
        C_sub = project_to_c_subspace(H, cfg.N_COMPONENTS, cfg.EPS)
        self.H, self.C_sub = H, C_sub
        H0 = gate(H, 0.0, cfg.N_COMPONENTS)
        err0 = float(np.abs(H0 - H).max())
        ok = bool(np.isclose(err0, 0.0, rtol=cfg.RTOL, atol=cfg.ATOL))
        if not ok:
            raise AIQValidationError(
                f"s=0 非恒等映射 max|Δ|={err0:.3e}",
                expected=0.0, actual=err0, param_key="G01",
            )
        return {"detail": f"[1] s=0 -> max|H_gated-H| = {err0:.3e}  (应为 0)", "err0": err0}

    # ------------------------------------------------------------ 2) 插值性质
    def validate_interp(self) -> dict:
        """2) s=0.3：插值性质 + Gamma 提升。"""
        cfg = self.config
        assert self.H is not None and self.C_sub is not None
        s = cfg.GATE_STRENGTH
        Hg = gate(self.H, s, cfg.N_COMPONENTS)
        dist_orig = float(np.linalg.norm(self.H - self.C_sub))
        dist_gated = float(np.linalg.norm(Hg - self.C_sub))
        assert dist_orig > 0.0, f"dist_orig={dist_orig} 除零风险"
        ratio = dist_gated / dist_orig
        ok2a = bool(np.isclose(ratio, 1.0 - s, rtol=cfg.RTOL, atol=cfg.ATOL))
        g0, g3 = gamma_svd(self.H, cfg.EPS), gamma_svd(Hg, cfg.EPS)
        self._g0, self._g3 = g0, g3  # 落盘共享：供 [5] 真实模型对照的门控提升标注
        ok2b = bool(g3 > g0)
        if not (ok2a and ok2b):
            raise AIQValidationError(
                f"插值比例 {ratio:.6f} ≠ {1-s} 或 Gamma 未提升 {g0:.6f}->{g3:.6f}",
                expected={"ratio": 1.0 - s, "gamma_up": True},
                actual={"ratio": ratio, "gamma0": g0, "gamma_s": g3},
                param_key="G01",
            )
        return {
            "detail": (f"[2] s={s} -> ||Hg-PiC H||/||H-PiC H|| = {ratio:.6f} "
                       f"(期望 {1-s}); Gamma 提升 {g0:.6f} -> {g3:.6f} (Δ={g3-g0:+.6f})"),
            "ratio": ratio, "gamma0": g0, "gamma_s": g3,
        }

    # ------------------------------------------------------------ 3) s=1 完全投影
    def validate_full_proj(self) -> dict:
        """3) s=1：完全投影 + 残差正交性。"""
        cfg = self.config
        assert self.H is not None and self.C_sub is not None
        H1 = gate(self.H, 1.0, cfg.N_COMPONENTS)
        err1 = float(np.abs(H1 - self.C_sub).max())
        ok3a = bool(np.isclose(err1, 0.0, rtol=cfg.RTOL, atol=cfg.ATOL))
        Hc = self.H - self.H.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        V = Vt[:2].T
        R = self.H - self.C_sub
        orth = float(np.abs(R @ V).max())
        ok3b = bool(np.isclose(orth, 0.0, rtol=cfg.RTOL, atol=cfg.ATOL))
        if not (ok3a and ok3b):
            raise AIQValidationError(
                f"s=1 非完全投影 max|Δ|={err1:.3e} 或残差非正交 max|R·V|={orth:.3e}",
                expected={"proj_err": 0.0, "orth": 0.0},
                actual={"proj_err": err1, "orth": orth},
                param_key="G01",
            )
        return {
            "detail": (f"[3] s=1 -> max|H_gated-PiC H| = {err1:.3e} (完全投影); "
                       f"残差正交性 max|R·V| = {orth:.3e} (≈0)"),
            "proj_err": err1, "orth": orth,
        }

    # ------------------------------------------------------------ 4) Gamma 单调
    def validate_gamma_monotonic(self) -> dict:
        """4) Gamma（SVD 口径）随 s 单调不减。"""
        cfg = self.config
        assert self.H is not None
        gammas = [gamma_svd(gate(self.H, ss, cfg.N_COMPONENTS), cfg.EPS) for ss in cfg.S_VALUES]
        diffs = np.diff(gammas)
        ok4 = bool(np.all(diffs >= -cfg.ATOL))
        if not ok4:
            raise AIQValidationError(
                f"Gamma 非单调不减: {[f'{g:.4f}' for g in gammas]}",
                expected="monotone non-decreasing", actual=gammas, param_key="G01",
            )
        return {
            "detail": f"[4] Gamma(s) 单调不减: {[f'{g:.4f}' for g in gammas]}",
            "gammas": gammas, "s_values": cfg.S_VALUES,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：k_proj Gamma 作为门控增强基准（_real_metrics.json 惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        real_gamma = rd.get("spectral.k_proj_gamma_mean", None)
        tag = rd.source_tag()
        if real_gamma is None:  # 真实值缺失：回退审计值标注，且该项不判失败
            audit_gamma = rd.audit("k_proj_gamma_mean", 0.625)
            return {
                "detail": (f"[5] [{tag}] k_proj Gamma 均值 = {audit_gamma:.4f} "
                           f"（审计回退值，未找到 _real_metrics.json）"),
                "source": tag, "real_gamma": None, "audit_gamma": audit_gamma,
            }
        real_layers = rd.get("spectral.k_proj_gamma_layers", [])
        audit_gamma = rd.audit("k_proj_gamma_mean", 0.625)
        g_min = float(min(real_layers)) if real_layers else float("nan")
        g_max = float(max(real_layers)) if real_layers else float("nan")
        ok5 = bool(np.isfinite(real_gamma)) and 0.0 < real_gamma < 1.0
        if not ok5:
            raise RealModelMismatchError(
                f"真实 Gamma={real_gamma:.4f} 非有限或越界",
                expected="(0,1)", actual=real_gamma, param_key="G01",
            )
        # 引用 [2] 的门控提升量作为"增强空间"标注（合成层，仅展示）
        lift = ""
        if getattr(self, "_g0", None) is not None and getattr(self, "_g3", None) is not None:
            lift = f"; 合成 s=0.3 门控 {self._g0:.4f}->{self._g3:.4f}（Δ={self._g3 - self._g0:+.4f}）"
        return {
            "detail": (f"[5] [{tag}] k_proj Gamma 均值 = {real_gamma:.4f} "
                       f"(24 层, range=[{g_min:.4f},{g_max:.4f}]); "
                       f"vs 文档审计值 {audit_gamma:.4f}，Δ={abs(real_gamma - audit_gamma):.4f}"
                       f"（以真实实测为准）{lift}"),
            "source": tag, "real_gamma": real_gamma, "audit_gamma": audit_gamma,
            "g_min": g_min, "g_max": g_max,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "s0_identity", self.validate_s0_identity),
            (2, "interp", self.validate_interp),
            (3, "full_proj", self.validate_full_proj),
            (4, "gamma_monotonic", self.validate_gamma_monotonic),
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
    """G01 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="G01 gate_strength 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()               # ① 配置层（env > YAML > JSON > 默认）
    synth = GateSynthesizer(cfg)                # ② 合成层
    report = ReportGenerator()                  # 报告器（复用 _factory 基类）
    engine = G01Validator(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("G01 gate_strength 谱门控公式验证（四层工厂架构，合成数据）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: B={cfg.B} D={cfg.D} N_COMPONENTS={cfg.N_COMPONENTS} "
          f"GATE_STRENGTH={cfg.GATE_STRENGTH} S_VALUES={cfg.S_VALUES}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "g01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "g01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "g01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
