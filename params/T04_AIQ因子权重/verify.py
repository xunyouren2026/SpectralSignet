# -*- coding: utf-8 -*-
"""T04 AIQ因子权重 AIQ 五因子加权合成 — 权重分配与基准复算
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 基准复算 AIQ = 100*(0.20f1+0.25f2+0.20f3+0.20f4+0.15f5) = 58.20
     （审计报告 407 项因子值 f=[0.625,0.989,0.000,0.625,0.565]）
  2. 权重归一化 sum(w) = 1.00（保证 AIQ ∈ [0,100]）
  3. w2 = 0.25 为最高权重（f2 跨域同构最强证据，README ③）
  4. f2 最高 / f3 最低（与审计报告 407 项一致）
  5. 灵敏度方向：提高 f2 权重 -> AIQ 上升；f3 解封(0->1) -> AIQ 提升
  6. 工程化防御：因子值域 [0,1]、权重非负、有限性检查
  7. 真实模型实测对照：真实五因子 -> AIQ（关键发现：低于文档审计）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T04Config / ConfigFactory / T04Validator(ValidatorEngine) /
  ReportGenerator / main —— 同 A01（env AIQ_T04_<KEY> 覆盖由共享工厂处理）。
  本参数无数据合成环节，故省略合成器层（engine 传入 synth=None）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T04 节（行 9915-9977）
  源码 _local_whitebox_detect.py（AIQ 合成行 304-310）、_param_spec_report.py（因子定义）
  《AI几何指纹插件_参数审计与实验报告.txt》407 AIQ（实测 58.20，状态=设计）
真实模型对照（关键发现）：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测真实五因子
  f=[0.470, 0.989, 0.000, 0.470, 0.429]（f1/f4=24 层 k_proj Gamma 逐层均值 0.4695，
  f5=1-(max-min)），代入 AIQ 公式 -> 真实 AIQ=49.94，显著低于文档审计 58.20。
  原因：文档以 Gamma=0.625 作 f1/f4，真实 k_proj Gamma=0.4695 更低，
  权重 w1+w4=0.40 拉低 AIQ。如实呈现差异，真实实测为准。
  输出标注：[真实实测]（数据齐全）/ [审计回退]（缺失）。
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
from _errors import AIQValidationError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 T04Config（pydantic 优先；dataclass 回退） ----
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


class T04Config(_ConfigModelBase):
    """T04 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T04 节点键名一一对应；取值优先级：
    环境变量 AIQ_T04_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                   # 算法逻辑常量：本脚本无随机采样，仅供可复现性约定
    WEIGHTS: list = [0.20, 0.25, 0.20, 0.20, 0.15]  # [f1..f5] 权重（README ①，和=1.00）
    F_AUDIT: list = [0.625, 0.989, 0.000, 0.625, 0.565]  # 审计报告 407 项因子值（README ④Step3）
    AIQ_AUDIT: float = 58.20        # 审计报告 407 项实测 AIQ（README ②Step3）
    W2_MAX: float = 0.25            # f2 最高权重 0.25（README ②权重表）


# ---- 第二层：配置工厂 ConfigFactory（实例化 T04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T04 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T04Config。"""

    def build(self) -> T04Config:
        """构建 T04Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T04Config, "T04")


# ---------------- 算法逻辑常量 ----------------
WEIGHT_SUM = 1.00    # 归一化约束（数学恒等式）
RTOL_AIQ = 1e-12     # AIQ 复算相对容差


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def aiq_score(weights, factors) -> float:
    """按权重合成 AIQ = 100 * sum(w_i * f_i)。

    输入须为同长一维数组；不做校验，校验在调用方（保持纯计算）。
    """
    return 100.0 * float(np.dot(np.asarray(weights, dtype=float),
                                np.asarray(factors, dtype=float)))


def validate_factors(factors: np.ndarray, weights: np.ndarray, name: str = "因子") -> None:
    """边界防御：因子必须有限、与权重同长、且落在 [0,1] 内。"""
    assert np.isfinite(factors).all(), f"{name} 含 NaN/Inf: {factors}"
    assert factors.shape == weights.shape, f"{name} 长度不符: {factors.shape} != {weights.shape}"
    assert bool((factors >= 0.0).all() and (factors <= 1.0).all()), \
        f"{name} 越出 [0,1] 值域: {factors}"


def validate_weights(weights: np.ndarray) -> None:
    """边界防御：权重必须有限且非负（归一化约束在验证项 3 单独校验）。"""
    assert np.isfinite(weights).all(), f"权重含 NaN/Inf: {weights}"
    assert bool((weights >= 0.0).all()), f"权重含负值: {weights}"


# ---- 第三层：验证引擎 T04Validator（9 项验证 + 结构化日志 + 类型化异常） ----
class T04ValidationError(AIQValidationError):
    """T04 AIQ 因子权重判据验证失败。"""


class T04Validator(_EngineBase):
    """T04 验证引擎：顺序执行 9 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 权重/因子/审计 AIQ 均经配置读取（env AIQ_T04_<KEY> 覆盖），零硬编码判据。
    """

    def __init__(
        self,
        config: T04Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)   # 本参数无合成环节，synth=None
        self._real_data = real_data
        self.W: np.ndarray = np.asarray(config.WEIGHTS, dtype=float)  # 权重向量
        self.F: np.ndarray = np.asarray(config.F_AUDIT, dtype=float)  # 审计因子向量
        self.aiq: float | None = None             # 基准 AIQ（步骤 2 复算，供灵敏度/真实对照复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 输入边界防御
    def validate_guards(self) -> dict:
        """1) 输入边界防御：因子有限∈[0,1]、权重有限非负 -> 合法。"""
        guards = [
            ("因子有限∈[0,1]", lambda: validate_factors(self.F, self.W)),
            ("权重有限", lambda: validate_weights(self.W)),
        ]
        guard_oks = []
        for _gname, fn in guards:
            try:
                fn()                             # 合法输入应通过（不抛异常）
                guard_oks.append(True)
            except AssertionError:
                guard_oks.append(False)          # 非法输入被拦截即防御生效
        ok_guard = all(guard_oks)
        if not ok_guard:
            raise T04ValidationError(
                f"输入防御失败: {[g[0] for g, o in zip(guards, guard_oks) if not o]}",
                expected="all valid", actual=guard_oks, param_key="T04",
            )
        return {"detail": f"防御: 因子有限∈[0,1]、权重有限非负 -> 合法 {guard_oks}",
                "guard_oks": guard_oks}

    # ------------------------------------------------------------ 2) 基准复算
    def validate_baseline(self) -> dict:
        """2) 基准复算 AIQ = 58.20（审计报告 407 项对照，np.isclose rtol=1e-12）。"""
        cfg = self.config
        self.aiq = aiq_score(self.W, self.F)     # 用审计因子复算 AIQ
        ok = bool(np.isclose(self.aiq, cfg.AIQ_AUDIT, rtol=RTOL_AIQ, atol=0.0))
        if not ok:
            raise T04ValidationError(
                f"AIQ 复算失败: 实得 {self.aiq} != {cfg.AIQ_AUDIT}",
                expected=cfg.AIQ_AUDIT, actual=self.aiq, param_key="T04",
            )
        return {"detail": f"基准复算 AIQ = {self.aiq:.6f} (期望 {cfg.AIQ_AUDIT}, "
                          f"np.isclose rtol={RTOL_AIQ})", "aiq": self.aiq}

    # ------------------------------------------------------------ 3) 权重归一化
    def validate_weight_sum(self) -> dict:
        """3) 权重归一化 sum(w) = 1.00（保证 AIQ 落在 [0,100]）。"""
        wsum = float(self.W.sum())
        ok = bool(np.isclose(wsum, WEIGHT_SUM, rtol=0.0, atol=1e-12))
        if not ok:
            raise T04ValidationError(
                f"权重和应为 1，实得 {wsum}",
                expected=WEIGHT_SUM, actual=wsum, param_key="T04",
            )
        return {"detail": f"权重归一化 sum(w) = {wsum:.4f} (=1.00)", "wsum": wsum}

    # ------------------------------------------------------------ 4) w2 最高
    def validate_w2_max(self) -> dict:
        """4) w2 = 0.25 为最高权重（f2 跨域同构最强证据）。"""
        cfg = self.config
        # 双条件防"权重表被改"：w2 既等于最大值、又等于规格值 0.25
        ok = bool(np.isclose(self.W[1], self.W.max(), rtol=0.0, atol=1e-12)
                  and np.isclose(self.W[1], cfg.W2_MAX, rtol=0.0, atol=1e-12))
        if not ok:
            raise T04ValidationError(
                f"w2 应为最高权重 0.25，实得 {self.W[1]}（max={self.W.max()}）",
                expected=cfg.W2_MAX, actual=self.W[1], param_key="T04",
            )
        return {"detail": (f"f2 权重最高: w2={self.W[1]:.2f} "
                           f"(其余 {self.W[[0, 2, 3, 4]].tolist()})"),
                "w2": self.W[1], "w_max": self.W.max()}

    # ------------------------------------------------------------ 5) 因子极值
    def validate_factor_extrema(self) -> dict:
        """5) f2 最高 / f3 最低（与审计报告一致）。"""
        ok = bool(np.isclose(self.F[1], self.F.max(), rtol=0.0, atol=1e-12)
                  and np.isclose(self.F[2], self.F.min(), rtol=0.0, atol=1e-12))
        if not ok:
            raise T04ValidationError(
                f"因子极值不符: f2={self.F[1]} f3={self.F[2]}（max={self.F.max()}, "
                f"min={self.F.min()}）",
                expected={"f2_max": True, "f3_min": True},
                actual={"f2": self.F[1], "f3": self.F[2]},
                param_key="T04",
            )
        return {"detail": (f"因子极值: f2={self.F[1]:.3f}(最高) f3={self.F[2]:.3f}(最低) "
                           f"与审计报告一致"),
                "f2": self.F[1], "f3": self.F[2], "f_max": self.F.max(), "f_min": self.F.min()}

    # ------------------------------------------------------------ 6) 灵敏度：提高 f2 权重
    def validate_sens_w2(self) -> dict:
        """6) 灵敏度：提高 f2 权重(0.25->0.30, 补偿降 f5) -> AIQ 上升。"""
        assert self.aiq is not None
        W_a = np.array([0.20, 0.30, 0.20, 0.20, 0.10])   # 重分配保持归一化：w2+0.05, w5-0.05
        if not np.isclose(W_a.sum(), 1.0):
            raise T04ValidationError(
                "灵敏度场景权重应仍归一化", expected=1.0, actual=W_a.sum(), param_key="T04",
            )
        aiq_a = aiq_score(W_a, self.F)
        ok = bool(aiq_a > self.aiq)
        if not ok:
            raise T04ValidationError(
                f"灵敏度方向错误: w2=0.30 时 AIQ={aiq_a} 未高于基准 {self.aiq}",
                expected=f"> {self.aiq}", actual=aiq_a, param_key="T04",
            )
        return {"detail": (f"提高 f2 权重 -> AIQ 上升: w2=0.30 -> {aiq_a:.2f} > "
                           f"基准 {self.aiq:.2f}"),
                "aiq_alt": aiq_a, "aiq_base": self.aiq}

    # ------------------------------------------------------------ 7) 灵敏度：f3 解封
    def validate_sens_f3(self) -> dict:
        """7) 灵敏度：f3 解封（长链 |H|->0）-> AIQ 上升 + 文档场景数值附注。"""
        assert self.aiq is not None
        F_hi = self.F.copy()
        F_hi[2] = 1.0                             # 假设长链使 H 收敛，f3 从 0 升到 1
        aiq_hi = aiq_score(self.W, F_hi)
        ok = bool(aiq_hi > self.aiq)
        if not ok:
            raise T04ValidationError(
                f"f3 解封后 AIQ={aiq_hi} 未高于基准 {self.aiq}",
                expected=f"> {self.aiq}", actual=aiq_hi, param_key="T04",
            )
        # 附注：文档预测长链 AIQ 落在 63-76，反推所需 f3 水平（不参与断言）
        f3_lo = (63.0 - self.aiq) / 20.0
        f3_hi = (76.0 - self.aiq) / 20.0
        # 附注：文档表场景数值（其自有因子口径，不参与断言）
        w_equal = np.array([0.20] * 5)
        w_conc = np.array([0.25, 0.15, 0.175, 0.25, 0.175])
        aiq_eq = aiq_score(w_equal, self.F)
        aiq_cc = aiq_score(w_conc, self.F)
        return {
            "detail": (f"f3 解封 -> AIQ 上升: f3=0 -> {self.aiq:.2f}, f3=1.0(完全) -> "
                       f"{aiq_hi:.2f}; [附注] 文档预测长链 63-76 对应 f3 部分解封区间 "
                       f"[{f3_lo:.2f}, {f3_hi:.2f}]; 文档表场景(自有口径, 不参与断言): "
                       f"等权重文档=56.85/复算={aiq_eq:.2f}, 强调集中度文档=57.40/复算={aiq_cc:.2f}"),
            "aiq_hi": aiq_hi, "aiq_base": self.aiq,
            "f3_range": [round(f3_lo, 2), round(f3_hi, 2)],
        }

    # ------------------------------------------------------------ 8) 真实模型对照
    def validate_real(self) -> dict:
        """8) 真实模型实测对照：真实五因子 -> AIQ 复算（与存档 AIQ 对照）。"""
        cfg = self.config
        rd = self._get_real_data()
        src = rd.source_tag()
        f1_r = rd.get("aiq.f1", None)
        f2_r = rd.get("aiq.f2", None)
        f3_r = rd.get("aiq.f3", None)
        f4_r = rd.get("aiq.f4", None)
        f5_r = rd.get("aiq.f5", None)
        aiq_r_meas = rd.get("aiq.AIQ", None)      # 实测存档的 AIQ（harness 已算好）
        if not all(v is not None for v in (f1_r, f2_r, f3_r, f4_r, f5_r)):
            return {"detail": (f"[审计回退] 真实五因子缺失，维持文档审计 "
                               f"AIQ={cfg.AIQ_AUDIT}"), "fallback": True}
        F_real = np.array([f1_r, f2_r, f3_r, f4_r, f5_r], dtype=float)
        try:
            validate_factors(F_real, self.W, "真实因子")   # 真实因子同样须满足值域约束
        except AssertionError as e:
            raise T04ValidationError(
                f"真实因子值域/长度不符: {e}",
                expected="[0,1] same-length", actual=F_real.tolist(), param_key="T04",
            )
        aiq_real = aiq_score(self.W, F_real)      # 用真实因子复算 AIQ
        ok_real = (aiq_r_meas is None) or bool(np.isclose(aiq_real, aiq_r_meas,
                                                          rtol=1e-9, atol=1e-9))
        if not ok_real:
            raise RealModelMismatchError(
                f"真实 AIQ 复算 {aiq_real} 与存档 {aiq_r_meas} 不符",
                expected=aiq_r_meas, actual=aiq_real, param_key="T04",
            )
        detail = (f"[{src}] 真实五因子 f=[{F_real[0]:.3f},{F_real[1]:.3f},"
                  f"{F_real[2]:.3f},{F_real[3]:.3f},{F_real[4]:.3f}] -> "
                  f"AIQ 复算={aiq_real:.2f} (实测存档={aiq_r_meas:.2f})")
        return {"detail": detail, "source": src, "aiq_real": aiq_real,
                "aiq_archive": aiq_r_meas, "f_real": F_real.tolist()}

    # ------------------------------------------------------------ 9) 真实 vs 审计
    def validate_real_audit(self) -> dict:
        """9) 关键发现：真实 AIQ vs 文档审计（如实呈现，真实实测为准）。"""
        cfg = self.config
        rd = self._get_real_data()
        f1_r = rd.get("aiq.f1", None)
        f2_r = rd.get("aiq.f2", None)
        f3_r = rd.get("aiq.f3", None)
        f4_r = rd.get("aiq.f4", None)
        f5_r = rd.get("aiq.f5", None)
        if not all(v is not None for v in (f1_r, f2_r, f3_r, f4_r, f5_r)):
            return {"detail": "[审计回退] 真实五因子缺失，跳过真实 vs 审计 AIQ 对照",
                    "fallback": True}
        F_real = np.array([f1_r, f2_r, f3_r, f4_r, f5_r], dtype=float)
        aiq_real = aiq_score(self.W, F_real)
        diff_aiq = aiq_real - cfg.AIQ_AUDIT       # 真实与审计口径的差距
        ok_diff = bool(diff_aiq < 0)              # 关键发现断言：真实 AIQ 应低于审计值
        if not ok_diff:
            raise T04ValidationError(
                f"真实 AIQ {aiq_real} 应低于文档审计 {cfg.AIQ_AUDIT}",
                expected=f"< {cfg.AIQ_AUDIT}", actual=aiq_real, param_key="T04",
            )
        detail = (f"真实 vs 审计 AIQ: 真实={aiq_real:.2f} 文档审计={cfg.AIQ_AUDIT:.2f} "
                  f"(Δ={diff_aiq:+.2f}); 五因子差异: f1 {F_real[0]:.3f} vs 0.625 | "
                  f"f2 {F_real[1]:.3f} vs 0.989 | f3 {F_real[2]:.3f} vs 0.000 | "
                  f"f4 {F_real[3]:.3f} vs 0.625 | f5 {F_real[4]:.3f} vs 0.565; "
                  f"归因: 真实 f1=f4=24层 k_proj Gamma 均值 {F_real[0]:.3f} < 文档审计 "
                  f"0.625，权重 w1+w4=0.40 拉低 AIQ; 结论: 真实实测为准")
        return {"detail": detail, "aiq_real": aiq_real, "aiq_audit": cfg.AIQ_AUDIT,
                "diff": diff_aiq}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 9 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "guards", self.validate_guards),
            (2, "baseline", self.validate_baseline),
            (3, "weight_sum", self.validate_weight_sum),
            (4, "w2_max", self.validate_w2_max),
            (5, "factor_extrema", self.validate_factor_extrema),
            (6, "sens_w2", self.validate_sens_w2),
            (7, "sens_f3", self.validate_sens_f3),
            (8, "real", self.validate_real),
            (9, "real_audit", self.validate_real_audit),
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


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """T04 验证编排：四层工厂装配（无合成层）+ --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T04 AIQ因子权重 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配（T04 无数据合成环节，省略合成层）----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T04Validator(cfg, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"T04 AIQ因子权重  AIQ = 100*(0.2f1+0.25f2+0.2f3+0.2f4+0.15f5)  "
          f"权重 {cfg.WEIGHTS}")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: AIQ_AUDIT={cfg.AIQ_AUDIT} W2_MAX={cfg.W2_MAX}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t04_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
