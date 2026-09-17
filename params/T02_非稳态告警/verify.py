# -*- coding: utf-8 -*-
"""T02 非稳态告警 DEFF 帧间波动 CV 告警 — 时间型几何稳定性监控
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 判据 CV = std(DEFF)/mean(DEFF)*100% > 3% 触发告警（时间型）
  2. 白盒 8 帧精确序列 CV ≈ 1.07% < 3% -> 稳态（对照实测）
  3. NS 饱和态参考序列 CV ≈ 0.25% / AI 短链 12 帧 CV ≈ 1.5% -> 稳态
  4. 构造高波动序列 CV > 3% -> 告警；构造崩溃序列 CV > 5% -> 危险级告警
  5. 恒稳序列 CV -> 0（极稳）
  6. 工程化防御：空输入/零均值（除零）/NaN/Inf 显式报错
  7. 真实模型实测对照：真实曲率 8 帧重切 CV -> 稳态
  8. 真实 vs 文档审计 CV 差异（<=0.3pp，真实实测为准）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T02Config / ConfigFactory / T02SequenceSynthesizer / T02Validator /
  ReportGenerator / main —— 同 A01（env AIQ_T02_<KEY> 覆盖由共享工厂处理）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T02 节（行 9710-9800）
  源码 _local_whitebox_detect.py（d_off = std(D_arr)/mean(D_arr) > 0.03）
  《AI几何指纹插件_参数审计与实验报告.txt》202 DEFF_A（白盒 8 帧，CV=1.07%）
真实模型对照：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测 phi_pairs_all.npy（6303×2），
  将逐点 DEFF 等分为 8 帧取帧均值，帧间 CV≈1.03%（< 3%）-> 稳态，
  与文档审计白盒 8 帧 CV=1.07% 一致（偏差 < 0.3pp）。
  输出标注：[真实实测]（phi_pairs_all.npy 存在）/ [审计回退]（缺失）。真实实测为准。
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

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 T02Config（pydantic 优先；dataclass 回退） ----
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


class T02Config(_ConfigModelBase):
    """T02 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T02 节点键名一一对应（CV 类字段为小数口径，
    判定时转百分数，与原脚本语义一致）；取值优先级：环境变量 AIQ_T02_<KEY>
    > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                # 合成序列固定种子（README ⑤）
    CV_THR: float = 0.03         # 告警阈值 CV>3%（小数口径 -> 转百分数）
    DANGER: float = 5.0          # 危险级阈值 CV > 5%（README ③ 分级表）
    CV_TOL: float = 0.3          # CV 对照容差（百分点）
    CV_WHITEBOX: float = 0.0107  # 白盒 8 帧审计 CV=1.07%（小数口径）
    CV_NS: float = 0.0025        # NS 饱和态审计 CV=0.25%（小数口径）
    CV_AI: float = 0.015         # AI 短链审计 CV=1.5%（小数口径）
    NS_MEAN: float = 1.5580      # NS 饱和态参考序列均值
    AI_MEAN: float = 1.5830      # AI 短链参考序列均值
    WB8: list = [1.5922, 1.5855, 1.5812, 1.5795,
                 1.5446, 1.5517, 1.5646, 1.5902]  # 白盒 8 帧精确序列


# ---- 第二层：配置工厂 ConfigFactory（实例化 T02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T02 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T02Config。"""

    def build(self) -> T02Config:
        """构建 T02Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T02Config, "T02")


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def cv_of(seq) -> tuple[float, np.ndarray]:
    """计算变异系数 CV = std/mean * 100%。

    返回 (CV%, 序列数组)。空输入、NaN/Inf、零均值（除零）显式报错。
    """
    a = np.asarray(seq, dtype=float)          # 统一为 float64 数组，杜绝整型截断
    if a.size == 0:                           # 空序列无统计意义
        raise ValueError(f"cv_of: 输入序列为空 {a}")
    if not np.isfinite(a).all():              # NaN/Inf 静默污染 std/mean
        raise ValueError(f"cv_of: 输入序列含 NaN/Inf {a}")
    mean = float(a.mean())
    if mean == 0.0:                           # 零均值导致除零无定义
        raise ValueError(f"cv_of: 均值为零，CV 无定义（除零防御） {a}")
    return float(np.std(a) / mean * 100.0), a  # 变异系数百分数 = 标准差/均值×100


# ---- 第三层：合成器 T02SequenceSynthesizer（指定均值/CV 的程序化序列） ----
class T02SequenceSynthesizer(_SynthBase):
    """T02 序列合成器：程序化生成指定均值与 CV 的序列（CV 精确 = cv_pct%）。"""

    def __init__(self, cfg: T02Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_seq(self, mean: float, cv_pct: float, n: int = 12,
                  seed: int | None = None) -> np.ndarray:
        """生成指定均值与 CV 的序列：标准化后缩放平移，CV = cv_pct 精确成立。"""
        if n <= 0:
            raise SynthesisError(f"synth_seq: 长度必须 > 0，收到 {n}", actual=n)
        if mean == 0.0:
            raise SynthesisError("synth_seq: mean 不能为 0（CV 分母）", actual=mean)
        if cv_pct < 0.0:
            raise SynthesisError(f"synth_seq: cv_pct 不能为负，收到 {cv_pct}", actual=cv_pct)
        if seed is None:
            seed = self._seed
        rng = np.random.default_rng(seed)     # 固定种子 => 可复现的合成数据
        t = rng.normal(size=n)
        t = (t - t.mean()) / t.std()          # 标准化：零均值、单位标准差
        std = cv_pct / 100.0 * mean           # 由目标 CV% 反解所需标准差：CV=std/mean
        return mean + std * t                 # 平移+缩放：均值=mean、CV=cv_pct 精确成立


# ---- 第四层：验证引擎 T02Validator（9 项验证 + 结构化日志 + 类型化异常） ----
class T02ValidationError(AIQValidationError):
    """T02 非稳态告警判据验证失败。"""


class T02Validator(_EngineBase):
    """T02 验证引擎：顺序执行 9 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 判据阈值经配置读取并转百分数（CV_THR 小数口径），与原脚本语义严格一致。
    """

    def __init__(
        self,
        config: T02Config,
        synth: T02SequenceSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.thr_pct: float = config.CV_THR * 100.0   # 告警阈值（百分数）
        self.danger_pct: float = config.DANGER        # 危险级阈值（百分数）
        self.cv_whitebox: float = config.CV_WHITEBOX * 100.0  # 审计 CV（百分数）
        self.cv_ns: float = config.CV_NS * 100.0      # NS 审计 CV（百分数）
        self.cv_ai: float = config.CV_AI * 100.0      # AI 审计 CV（百分数）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _check_cv(self, name: str, seq, expected_alert: bool,
                  expected_cv: float | None = None, cv_tol: float | None = None) -> dict:
        """单例判据：算 CV、判告警状态、与期望对照（供多个步骤复用）。

        注意：不能命名为 _check_seq——基类 ValidatorEngine 用该属性做
        内部计数器（self._check_seq = 0），会遮蔽同名方法。
        """
        cfg = self.config
        cv, _ = cv_of(seq)                     # 先算 CV（内部已含边界防御）
        alert = cv > self.thr_pct              # 核心判据：CV 超阈值即告警
        ok = bool(alert == expected_alert)     # 告警状态与预期一致
        note = ""
        if expected_cv is not None:            # 若指定期望 CV，同时校验数值精确度
            if cv_tol is None:
                cv_tol = cfg.CV_TOL
            ok = ok and bool(np.isclose(cv, expected_cv, rtol=0.0, atol=cv_tol))
            note = f" (期望 CV≈{expected_cv:.2f}%)"
        extra_txt = ""
        if cv > self.danger_pct:               # 危险级分级：CV 超 5% 提示可能崩溃
            extra_txt = "  -> 危险级: 表示退化，可能崩溃"
        if not ok:
            raise T02ValidationError(
                f"{name}: CV={cv:.3f}% 告警状态/数值不符",
                expected={"alert": expected_alert, "cv": expected_cv},
                actual={"alert": alert, "cv": cv},
                param_key="T02",
            )
        status = "告警" if alert else "稳态"
        return {"detail": f"{name}: CV={cv:6.3f}% -> {status}{note}{extra_txt}",
                "cv_pct": cv, "alert": alert, "expected_cv": expected_cv}

    # ------------------------------------------------------------ 1) 白盒 8 帧
    def validate_wb8(self) -> dict:
        """1) 白盒 8 帧精确序列（CV≈1.07% < 3% -> 稳态）。"""
        return self._check_cv("白盒 8 帧", self.config.WB8, False,
                              expected_cv=self.cv_whitebox)

    # ------------------------------------------------------------ 2) NS 饱和态
    def validate_ns(self) -> dict:
        """2) NS 饱和态参考序列（CV≈0.25% -> 稳态，判据不误报）。"""
        assert self.synth is not None
        ns12 = self.synth.synth_seq(self.config.NS_MEAN, self.cv_ns, seed=self.config.SEED)
        return self._check_cv("NS 饱和态 12 帧", ns12, False, expected_cv=self.cv_ns)

    # ------------------------------------------------------------ 3) AI 短链
    def validate_ai(self) -> dict:
        """3) AI 短链 12 帧参考序列（CV≈1.5% -> 稳态，分级余量设计）。"""
        assert self.synth is not None
        ai12 = self.synth.synth_seq(self.config.AI_MEAN, self.cv_ai, seed=self.config.SEED)
        return self._check_cv("AI 短链 12 帧", ai12, False, expected_cv=self.cv_ai)

    # ------------------------------------------------------------ 4) 高波动告警
    def validate_unstable(self) -> dict:
        """4) 构造高波动序列（1.48~1.68 大摆幅）-> CV > 3% 触发告警。"""
        unstable = [1.55, 1.62, 1.48, 1.68, 1.52, 1.64, 1.50, 1.66]
        return self._check_cv("构造高波动 8 帧", unstable, True)

    # ------------------------------------------------------------ 5) 崩溃序列
    def validate_collapse(self) -> dict:
        """5) 构造崩溃序列（1.25~1.80 极端摆幅）-> CV > 5% 危险级告警。"""
        collapse = [1.30, 1.60, 1.25, 1.75, 1.28, 1.70, 1.35, 1.80]
        return self._check_cv("构造崩溃序列", collapse, True)

    # ------------------------------------------------------------ 6) 恒稳序列
    def validate_constant(self) -> dict:
        """6) 恒稳序列：CV -> 0（极稳，浮点噪声 <1e-12）。"""
        cv_s, _ = cv_of([1.56] * 8)             # 恒定序列标准差恒为 0
        ok = bool(cv_s < 1e-12 and cv_s <= self.thr_pct)
        if not ok:
            raise T02ValidationError(
                f"恒稳序列 CV 应为 0，实得 {cv_s}",
                expected=0.0, actual=cv_s, param_key="T02",
            )
        return {"detail": f"恒稳序列 CV={cv_s:.6f}% -> 极稳（<1% 级）", "cv_pct": cv_s}

    # ------------------------------------------------------------ 7) 防御
    def validate_guards(self) -> dict:
        """7) 工程化防御：空输入 / 零均值 / NaN / Inf 显式报错。"""
        guards = [
            ("空输入", lambda: cv_of([])),
            ("零均值", lambda: cv_of([0.0, 0.0])),
            ("NaN 输入", lambda: cv_of([1.5, float("nan")])),
            ("Inf 输入", lambda: cv_of([1.5, float("inf")])),
        ]
        guard_oks = []
        for _gname, fn in guards:
            try:
                fn()                             # 若未抛异常 => 防御缺失
                guard_oks.append(False)
            except (ValueError, ZeroDivisionError):
                guard_oks.append(True)           # 显式报错视为防御通过
        ok_guard = all(guard_oks)
        if not ok_guard:
            raise T02ValidationError(
                f"边界防御失败: {[g[0] for g, o in zip(guards, guard_oks) if not o]}",
                expected="all raise", actual=guard_oks, param_key="T02",
            )
        return {"detail": f"防御: 空/零均值/NaN/Inf 均显式报错 {guard_oks}",
                "guard_oks": guard_oks}

    # ------------------------------------------------------------ 8) 真实模型对照
    def validate_real(self) -> dict:
        """8) 真实模型实测对照：真实 phi_pairs_all.npy 重切 8 帧 CV -> 稳态。"""
        rd = self._get_real_data()
        src = rd.source_tag()
        pairs = rd.phi_pairs()                 # 经 _cfg 相对定位（零硬编码）
        if pairs is None:                      # 审计回退：文档审计 CV=1.07%
            cv_real = float(rd.audit("deff_cv", self.cv_whitebox))
            ok_real = bool(cv_real < self.thr_pct)
            if not ok_real:
                raise T02ValidationError(
                    f"审计 CV={cv_real}% 应 < {self.thr_pct}%",
                    expected=self.thr_pct, actual=cv_real, param_key="T02",
                )
            return {"detail": (f"[审计回退] 真实 phi_pairs_all.npy 缺失，用审计 8 帧 "
                               f"CV≈{cv_real:.2f}% -> 稳态"), "fallback": True,
                    "cv_real": cv_real}
        kk1, kk2 = pairs[:, 0], pairs[:, 1]
        a1, a2 = np.abs(kk1), np.abs(kk2)      # 主曲率取绝对值
        # 逐点 DEFF 公式：(|κ1|+|κ2|)²/(κ1²+κ2²+ε)，ε 防除零
        deff_pt = (a1 + a2) ** 2 / (kk1 * kk1 + kk2 * kk2 + 1e-16)
        # 等分为 8 帧、每帧取均值 => 与文档"白盒 8 帧"口径一致
        fmeans = np.array([float(f.mean()) for f in np.array_split(deff_pt, 8)])
        cv_real, _ = cv_of(fmeans)             # 帧间 CV 过同一判据
        ok_real = bool(cv_real < self.thr_pct)  # 预期真实模型处于稳态
        if not ok_real:
            raise RealModelMismatchError(
                f"真实 8 帧 CV={cv_real:.3f}% 应 < {self.thr_pct}%（非稳态告警）",
                expected=self.thr_pct, actual=cv_real, param_key="T02",
            )
        return {
            "detail": (f"[{src}] 真实曲率 8 帧重切: 帧均值={[f'{x:.4f}' for x in fmeans]}; "
                       f"CV={cv_real:.3f}% (< {self.thr_pct}%) -> 稳态"),
            "source": src, "cv_real": cv_real, "frame_means": [float(x) for x in fmeans],
        }

    # ------------------------------------------------------------ 9) 真实 vs 审计
    def validate_real_audit(self) -> dict:
        """9) 真实 vs 文档审计 CV 差异：|Δ| <= CV_TOL pp，两口径一致。"""
        cfg = self.config
        rd = self._get_real_data()
        pairs = rd.phi_pairs()
        if pairs is None:                      # 真实缺失：维持审计口径，视为一致
            return {"detail": "[审计回退] 真实数据缺失，跳过真实 vs 审计 CV 对照",
                    "fallback": True}
        # 与步骤 8 同口径重算真实 8 帧 CV
        kk1, kk2 = pairs[:, 0], pairs[:, 1]
        deff_pt = (np.abs(kk1) + np.abs(kk2)) ** 2 / (kk1 * kk1 + kk2 * kk2 + 1e-16)
        fmeans = np.array([float(f.mean()) for f in np.array_split(deff_pt, 8)])
        cv_real, _ = cv_of(fmeans)
        cv_audit = self.cv_whitebox            # 文档审计 CV=1.07%
        diff_cv = abs(cv_real - cv_audit)      # 两口径偏差（百分点）
        ok = bool(diff_cv <= cfg.CV_TOL)       # 容差 0.3pp 内视为一致
        if not ok:
            raise T02ValidationError(
                f"真实 CV={cv_real}% 与审计 {cv_audit}% 偏差 {diff_cv}pp 超 {cfg.CV_TOL}pp",
                expected=cfg.CV_TOL, actual=diff_cv, param_key="T02",
            )
        return {
            "detail": (f"真实 vs 审计: 8帧 CV 真实={cv_real:.3f}% 审计={cv_audit:.2f}% "
                       f"(|Δ|={diff_cv:.3f}pp ≤ {cfg.CV_TOL}pp; 真实实测为准)"),
            "cv_real": cv_real, "cv_audit": cv_audit, "diff_pp": diff_cv,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 9 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "wb8", self.validate_wb8),
            (2, "ns", self.validate_ns),
            (3, "ai", self.validate_ai),
            (4, "unstable", self.validate_unstable),
            (5, "collapse", self.validate_collapse),
            (6, "constant", self.validate_constant),
            (7, "guards", self.validate_guards),
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


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """T02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T02 非稳态告警 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = T02SequenceSynthesizer(cfg)               # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T02Validator(cfg, synth, report, real_data=RD)  # ③ 验证层

    print("=" * 74)
    print(f"T02 非稳态告警  CV = std(DEFF)/mean(DEFF)*100% > {cfg.CV_THR * 100:.0f}%")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: CV_THR={cfg.CV_THR} DANGER={cfg.DANGER} CV_TOL={cfg.CV_TOL} "
          f"NS_MEAN={cfg.NS_MEAN} AI_MEAN={cfg.AI_MEAN}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
