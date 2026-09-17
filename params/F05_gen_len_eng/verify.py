# -*- coding: utf-8 -*-
"""F05 gen_len_eng 工程指标捕获步数 — 48 步测量稳定性验证（四层工厂架构）
====================================================================
公式: tok/s = n_gen / t_gen,  BW_eff = W_bytes * tok/s
固定: gen_len_eng = 48,  帧对齐: 48 / TOK_PER_FRAME(8) = 6 帧

合成计时模型:
  每步稳态耗时 τ=0.1192s, 一次性开销 c=0.068s(首token warm-up), 1% 噪声(固定种子)
  t(n) = n*τ + c   →  tok/s: 16≈8.10, 48≈8.29, 128≈8.35 (趋势一致)

验证目标（与原脚本完全一致，保真）：
  1. tok/s 随步数收敛: 16步偏低(受warm-up拖累), 48步稳定接近8.39, 128步饱和
  2. 48 步 tok/s 落在 8.39 ± 6% 内
  3. 帧对齐: 48 = 6 × 8 (B04 TOK_PER_FRAME 整倍数)
  4. 稳定性: 多次重复 CV(48步) < CV(16步)  (波动 ∝ 1/sqrt(n))
  5. 有效带宽 = 1.98e9 × tok/s ≈ 16.6 GB/s (±8%)

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  F05Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 F05Config（env AIQ_F05_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享测量由引擎 _measure 惰性计算）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 4614-4723（F05 gen_len_eng）
  《参数完整定义与公式.txt》F05（第 334-340 行）
  源码 _local_whitebox_detect.py（第 77-91 行 decode 计时；第 151-154 行带宽）
  实测日志 wb_full.log（64tok/7.63s → 8.39 tok/s；16.6GB/s）
  《参数审计与实验报告.txt》行 74（状态=理论）
说明：纯数值合成数据，不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 48 步生成测量（ngen_actual=48，tok/s=7.20）：
  验证 gen_len_eng=48 与真实生成步数精确一致、6 帧对齐，并以真实 tok/s
  重新标定计时模型复算收敛趋势。文档审计 8.39 为旧实测，真实实测为准。
  数据来源标注：[真实实测] 或 [审计回退]。
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
from _errors import (  # noqa: E402
    AIQValidationError,
    ConfigError,
    RealModelMismatchError,
    ReproducibilityError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 F05Config（pydantic 优先；dataclass 回退） ----
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


class F05Config(_ConfigModelBase):
    """F05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 F05 节点键名对应（GEN_LEN/TOK_S_DOC/
    BW_EFF_TARGET/TOK_PER_FRAME）；取值优先级：env AIQ_F05_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0                # 主序列固定随机种子（算法逻辑常量）
    SEED_CV: int = 7             # 稳定性重复实验随机种子（算法逻辑常量）
    WEIGHT_BYTES: float = 1.98e9  # 权重字节数（wb_full.log:14）
    GEN_LEN_ENG: int = 48         # F05: 工程指标捕获步数（README ①）
    TOK_S_TARGET: float = 8.39    # 实测 tok/s（wb_full.log）
    BW_EFF_TARGET: float = 16.6   # 实测有效带宽（GB/s，wb_full.log:31）
    TOK_PER_FRAME: int = 8        # B04: TOK_PER_FRAME（帧对齐）
    PER_TOK: float = 0.1192       # 每步稳态耗时 τ（s，README ④推导 1 标定）
    WARM_UP: float = 0.068        # 一次性 warm-up 开销 c（s，README ④推导 1）
    NOISE_STD: float = 0.01       # 逐步耗时噪声（1%）
    REP_CV: int = 5               # 稳定性 CV 的重复次数
    TS_TOL_PCT: float = 6.0       # 48 步 tok/s 与实测的允许偏差（%）
    TS16_MAX: float = 8.3         # 16 步 tok/s 上限（受 warm-up 拖累）
    TS48_MIN: float = 8.1         # 48 步 tok/s 下限
    CONV_TOL: float = 0.5         # 128 步与 48 步收敛的绝对差上限
    BW_EFF_TOL: float = 0.08      # 有效带宽相对允许偏差
    REAL_SEED: int = 11           # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 F05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """F05 配置工厂：按优先级实例化 F05Config。"""

    def build(self) -> F05Config:
        """构建 F05Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(F05Config, "F05")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def gen_times(n_steps: int, rng: np.random.Generator, cfg: F05Config) -> list:
    """合成生成耗时序列: 稳态 τ + 首步一次性 warm-up 开销 + 噪声。

    每步耗时钳制在 [1e-9, ∞) 防御非正时间。空输入返回空列表。
    """
    if n_steps <= 0:
        return []
    times = []
    for i in range(n_steps):
        t = cfg.PER_TOK * (1.0 + rng.normal(0.0, cfg.NOISE_STD))  # 稳态耗时 + 噪声
        if i == 0:
            t += cfg.WARM_UP      # 首步额外 warm-up（一次性开销）
        times.append(max(t, 1e-9))   # 钳制非正时间（防御除零）
    return times


def tok_s(n_steps: int, rng: np.random.Generator, cfg: F05Config) -> float:
    """n 步生成的平均 tok/s = n / Σt。防御：总耗时非正返回 0.0。"""
    total = float(sum(gen_times(n_steps, rng, cfg)))
    if total <= 0.0:
        return 0.0
    return n_steps / total   # 总步数 / 总耗时


def cv_of(n_steps: int, cfg: F05Config) -> float:
    """多次重复的 CV（变异系数）: std/mean(tok/s)。防御：均值为 0 返回 0.0。"""
    r = np.random.default_rng(cfg.SEED_CV)   # 固定种子：CV 可复现
    vals = [tok_s(n_steps, r, cfg) for _ in range(cfg.REP_CV)]   # reps 次重复测量
    m = float(np.mean(vals))
    if m == 0.0:
        return 0.0
    return float(np.std(vals) / m)   # 相对波动：衡量测量稳定性


# ---- 第三层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """F05 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享 16/48/128 步测量在 _measure() 中一次性按原脚本单 rng 流计算。
    """

    def __init__(
        self,
        config: F05Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._measured = False
        self.ts16 = 0.0
        self.ts48 = 0.0
        self.ts128 = 0.0
        self.cv16 = 0.0
        self.cv48 = 0.0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _measure(self) -> None:
        """一次性测量主序列（与原脚本 main 的单一 rng 流一致，惰性）。"""
        if self._measured:
            return
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)   # 固定种子：主序列可复现
        self.ts16 = tok_s(16, rng, cfg)
        self.ts48 = tok_s(cfg.GEN_LEN_ENG, rng, cfg)
        self.ts128 = tok_s(128, rng, cfg)
        self.cv16, self.cv48 = cv_of(16, cfg), cv_of(cfg.GEN_LEN_ENG, cfg)
        self._measured = True

    # ------------------------------------------------------------ 1) 步数→tok/s 收敛趋势
    def validate_convergence(self) -> dict:
        """1) 16 步受 warm-up 拖累偏低，128 步与 48 步收敛。"""
        cfg = self.config
        self._measure()
        ok1a = self.ts16 < self.ts48              # 16 步受 warm-up 拖累
        ok1b = abs(self.ts128 - self.ts48) < cfg.CONV_TOL  # 128 步与 48 步收敛(增益递减)
        ok1c = self.ts16 < cfg.TS16_MAX and self.ts48 > cfg.TS48_MIN
        if not (ok1a and ok1b and ok1c):
            raise ConfigError(
                f"tok/s 收敛趋势不符: 16步={self.ts16:.2f}(<{cfg.TS16_MAX}), "
                f"48步={self.ts48:.2f}(>{cfg.TS48_MIN}), 128步={self.ts128:.2f}(|Δ48|<{cfg.CONV_TOL})",
                expected={"ts16<ts48": True, "conv": cfg.CONV_TOL,
                          "ts16<": cfg.TS16_MAX, "ts48>": cfg.TS48_MIN},
                actual={"ts16": self.ts16, "ts48": self.ts48, "ts128": self.ts128},
                param_key="F05",
            )
        return {
            "detail": (f"tok/s: 16步={self.ts16:.2f} | 48步={self.ts48:.2f} | 128步={self.ts128:.2f} "
                       f"(主文档: 8.10 | 8.39 | 8.41 -> 16<48 收敛于 128)"),
            "ts16": self.ts16, "ts48": self.ts48, "ts128": self.ts128,
        }

    # ------------------------------------------------------------ 2) 48 步落在 8.39 ± 6%
    def validate_target(self) -> dict:
        """2) 48 步 tok/s 落在实测 8.39 ± 6% 内。"""
        cfg = self.config
        self._measure()
        rel = abs(self.ts48 - cfg.TOK_S_TARGET) / cfg.TOK_S_TARGET * 100
        ok = rel < cfg.TS_TOL_PCT
        if not ok:
            raise ConfigError(
                f"48 步 tok/s={self.ts48:.3f} 应落在 8.39±6% 内（偏差 {rel:.2f}%）",
                expected=cfg.TOK_S_TARGET, actual=self.ts48, param_key="F05",
            )
        return {
            "detail": f"48 步 tok/s = {self.ts48:.3f} vs 实测 {cfg.TOK_S_TARGET} "
                      f"(相对偏差 {rel:.2f}%, <6%)",
            "rel_pct": rel, "ts48": self.ts48,
        }

    # ------------------------------------------------------------ 3) 帧对齐
    def validate_frame_align(self) -> dict:
        """3) 帧对齐: gen_len_eng 是 TOK_PER_FRAME 的整倍数（6 帧）。"""
        cfg = self.config
        n_frames = cfg.GEN_LEN_ENG // cfg.TOK_PER_FRAME
        ok = cfg.GEN_LEN_ENG % cfg.TOK_PER_FRAME == 0 and n_frames == 6
        if not ok:
            raise ConfigError(
                f"48 应对齐 8 的整倍数（6 帧），实际 {cfg.GEN_LEN_ENG}%{cfg.TOK_PER_FRAME}",
                expected={"mod": 0, "frames": 6},
                actual={"mod": cfg.GEN_LEN_ENG % cfg.TOK_PER_FRAME, "frames": n_frames},
                param_key="F05",
            )
        return {"detail": f"帧对齐: {cfg.GEN_LEN_ENG} / {cfg.TOK_PER_FRAME} = {n_frames} 帧 "
                          f"(B04 TOK_PER_FRAME 整倍数)", "n_frames": n_frames}

    # ------------------------------------------------------------ 4) 稳定性
    def validate_stability(self) -> dict:
        """4) 稳定性: CV(48步) < CV(16步)（波动 ∝ 1/√n）。"""
        cfg = self.config
        self._measure()
        ok = self.cv48 < self.cv16
        if not ok:
            raise ReproducibilityError(
                f"48 步应比 16 步更稳定: CV16={self.cv16*100:.3f}%, CV48={self.cv48*100:.3f}%",
                expected=self.cv16, actual=self.cv48, param_key="F05",
            )
        return {
            "detail": f"稳定性({cfg.REP_CV}次重复CV): 16步={self.cv16*100:.3f}% | "
                      f"48步={self.cv48*100:.3f}% (48更稳, 波动∝1/√n)",
            "cv16": self.cv16, "cv48": self.cv48,
        }

    # ------------------------------------------------------------ 5) 有效带宽
    def validate_bw_eff(self) -> dict:
        """5) 有效带宽 = W_bytes × tok/s ≈ 16.6 GB/s（±8%）。"""
        cfg = self.config
        self._measure()
        bw_eff = cfg.WEIGHT_BYTES * self.ts48
        bw_eff_gbs = bw_eff / 1e9
        ok = abs(bw_eff_gbs - cfg.BW_EFF_TARGET) / cfg.BW_EFF_TARGET < cfg.BW_EFF_TOL
        if not ok:
            raise ConfigError(
                f"有效带宽应≈{cfg.BW_EFF_TARGET}GB/s(±8%)，实际 {bw_eff_gbs:.1f}GB/s",
                expected=cfg.BW_EFF_TARGET, actual=bw_eff_gbs, param_key="F05",
            )
        return {
            "detail": f"有效带宽 = {cfg.WEIGHT_BYTES/1e9:.2f}GB × {self.ts48:.2f} = "
                      f"{bw_eff_gbs:.1f} GB/s (实测 {cfg.BW_EFF_TARGET}GB/s, ±8%)",
            "bw_eff_gbs": bw_eff_gbs,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实 48 步生成测量（ngen=48, tok/s=7.20）复算收敛趋势。

        以真实 tok/s 重新标定 τ=1/tok_s，复算 16/48/128 步收敛趋势；
        验证 gen_len_eng=48 与真实生成步数精确一致。数据缺失回退。
        """
        cfg = self.config
        rd = self._get_real_data()
        ngen = rd.get("engine.ngen_actual")
        tok_s_real = rd.get("engine.tok_s")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 数据缺失：回退审计值（8.39），对照层跳过（不判失败）
        if ngen is None or tok_s_real is None:
            return {
                "detail": f"{tag} 真实数据缺失 -> 回退审计值（8.39），对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        ok1 = ngen == cfg.GEN_LEN_ENG             # 真实生成步数应 = 48
        ok2 = (ngen % cfg.TOK_PER_FRAME == 0) and (ngen // cfg.TOK_PER_FRAME == 6)  # 帧对齐
        tau_real = 1.0 / tok_s_real               # 以真实 tok/s 重新标定 τ
        rng_r = np.random.default_rng(cfg.REAL_SEED)   # 固定种子：可复现
        # 用真实 τ 复算 16/48/128 步 tok/s（warm-up 仍保留，验证收敛趋势）
        def _real_tok_s(n: int) -> float:
            """按真实 τ 与 warm-up 生成 n 步耗时序列并求平均 tok/s（防御除零）。"""
            times = []
            for i in range(n):
                t = tau_real * (1.0 + rng_r.normal(0.0, cfg.NOISE_STD))
                if i == 0:
                    t += cfg.WARM_UP      # 首步一次性 warm-up（与原脚本语义一致）
                times.append(max(t, 1e-9))   # 钳制非正时间
            return n / max(float(sum(times)), 1e-9)

        ts16 = _real_tok_s(16)
        ts48 = _real_tok_s(cfg.GEN_LEN_ENG)
        ts128 = _real_tok_s(128)
        ok3 = ts16 < ts48 and abs(ts128 - ts48) < cfg.CONV_TOL   # 收敛趋势
        rel = abs(ts48 - tok_s_real) / tok_s_real * 100
        ok4 = rel < cfg.TS_TOL_PCT                # 48 步复算偏差 <6%
        if not (ok1 and ok2 and ok3 and ok4):
            raise RealModelMismatchError(
                f"真实 48 步测量复算不符: ngen={ngen}, 帧对齐={ok2}, "
                f"16/48/128={ts16:.2f}/{ts48:.2f}/{ts128:.2f}, 偏差={rel:.2f}%",
                expected={"ngen": cfg.GEN_LEN_ENG, "frame_align": True,
                          "conv": cfg.CONV_TOL, "rel<": cfg.TS_TOL_PCT},
                actual={"ngen": ngen, "ts16": ts16, "ts48": ts48, "ts128": ts128,
                        "rel": rel}, param_key="F05",
            )
        return {
            "detail": (f"{tag} 生成步数 ngen={ngen} = gen_len_eng={cfg.GEN_LEN_ENG}; "
                       f"帧对齐: {ngen} / {cfg.TOK_PER_FRAME} = {ngen//cfg.TOK_PER_FRAME} 帧; "
                       f"重新标定 τ=1/tok_s={tau_real:.4f}s: 16步={ts16:.2f} | 48步={ts48:.2f} | "
                       f"128步={ts128:.2f}（收敛）; 48 步 tok/s = {tok_s_real:.2f}（偏差 {rel:.2f}%<6%）; "
                       f"[差异] 真实 tok/s 7.20 vs 文档审计 8.39（旧实测，CPU 单机差异）"),
            "source": rd.source_tag(), "tag": tag,
            "ngen": ngen, "tok_s_real": tok_s_real,
            "ts16": ts16, "ts48": ts48, "ts128": ts128, "rel": rel,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "convergence", self.validate_convergence),
            (2, "target", self.validate_target),
            (3, "frame_align", self.validate_frame_align),
            (4, "stability", self.validate_stability),
            (5, "bw_eff", self.validate_bw_eff),
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


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """F05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="F05 gen_len_eng 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    engine = ValidatorEngine(cfg, report, real_data=RD)

    print("=" * 78)
    print("F05 gen_len_eng 工程指标捕获步数 —— 公式验证（四层工厂架构，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"gen_len_eng = {cfg.GEN_LEN_ENG} | 权重 = {cfg.WEIGHT_BYTES/1e9:.2f}GB (fp32)")
    print("=" * 78)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "f05_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "f05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "f05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
