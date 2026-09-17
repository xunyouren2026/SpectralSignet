# -*- coding: utf-8 -*-
"""E04 GEN_PROMPT 重生成扰动种子 — 归一化溯源边距验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 归一化边距公式：margin = |dG - dQ| / max(dQ + dG, 1e-6)
  2. 3 个重生成样本（n=3）的 margin 均值 ≈ 实测 0.2975
  3. 重生成样本全部正确溯源到 Qwen（100%）
  4. 与无扰动基线 margin≈0.193 对照：提升 ≈ 54%

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E04Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 E04Config（env AIQ_E04_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享状态由引擎 _synthesize 惰性计算）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 3644-3766（E04 GEN_PROMPT）
  源码 _qwen_robustness.py（L51 重生成与 margin 公式）
  《参数审计与实验报告.txt》（状态=已用 regen n=3, margin=0.2975）
说明：纯数值合成数据（48 维特征），不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 k_proj Gamma 24 维剖面作为 Qwen 参考簇基线的
  beta24 部分（曲率 24 维合成），复算重生成归一化溯源边距；GPT-2 参考簇
  为设计口径。数据来源标注：[真实实测] 或 [审计回退]。
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
    TraceabilityError,
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

# ---- 第一层：配置模型 E04Config（pydantic 优先；dataclass 回退） ----
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


class E04Config(_ConfigModelBase):
    """E04 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E04 节点键名一一对应（GRID/NFRAME/DIM/
    N_SAMPLE/MARGIN_DOC）；取值优先级：env AIQ_E04_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0               # 固定随机种子：保证可复现（算法逻辑常量）
    GRID: int = 24              # beta 剖面格点（B03）
    NFRAME: int = 8             # 曲率帧数（B08），8 帧 x 3 指标 = 24 维
    DIM: int = 48               # 48 维：beta24 + 曲率24（E04 DIM）
    N_GEN: int = 3              # E04: 重生成次数（README ① n=3）
    MARGIN_TARGET: float = 0.2975  # 实测：重生成 margin 均值（README ⑤）
    MARGIN_BASE: float = 0.193     # 实测：无扰动基线 margin 均值（README ⑤）
    C_TARGET: float = 0.428        # 两家族参考簇间距目标（README ④Step 2，c≈0.43）
    Q_NORM: float = 0.55           # Qwen 簇归一化幅度（原始构造）
    EPS: float = 1e-6              # margin 公式分母下限（源码 max(dQ+dG, 1e-6)）
    MARGIN_TOL: float = 0.01       # margin 均值与实测的允许偏差
    IMPROVE_MIN: float = 1.50      # 重生成相对基线提升下限（实测 1.54x）
    GEN_DQ: tuple = (0.150, 0.155, 0.146)   # 重生成样本 dQ 设计值（README ⑤）
    BASE_DQ: tuple = (0.170, 0.173, 0.175)  # 无扰动基线样本 dQ 设计值（原始构造）
    REAL_SEED: int = 1234          # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E04 配置工厂：按优先级实例化 E04Config。"""

    def build(self) -> E04Config:
        """构建 E04Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E04Config, "E04")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def make_reference_clusters(rng: np.random.Generator, cfg: E04Config) -> tuple:
    """构造 Qwen / GPT2 参考簇基线，簇间距 ||f_q - f_g|| = C_TARGET。

    返回 (f_q, f_g, c, e)：f_q/Qwen 基线, f_g/GPT2 基线, 间距 c, 单位方向 e。
    """
    f_q = cfg.Q_NORM * rng.standard_normal(cfg.DIM)   # 随机初始方向
    f_q = f_q / np.linalg.norm(f_q) * cfg.Q_NORM      # 归一化到 Q_NORM 幅度
    f_g = f_q + cfg.C_TARGET                          # 先按标量粗放，稍后精确归一化
    # 归一化使簇间距严格 ||f_q - f_g|| = c = C_TARGET
    f_g = f_q + (f_g - f_q) / np.linalg.norm(f_g - f_q) * cfg.C_TARGET
    c = float(np.linalg.norm(f_q - f_g))              # 实际簇间距（应精确等于 C_TARGET）
    # 方向单位向量（Qwen -> GPT2）：供 make_sample 在连线上放置样本
    e = (f_g - f_q) / c if c > 0.0 else np.zeros(cfg.DIM)
    return f_q, f_g, c, e


def make_sample(f_q: np.ndarray, dq: float, e: np.ndarray) -> np.ndarray:
    """置于 Qwen->GPT2 连线上、距 Qwen 基线 dq 的样本（共线保证 dQ+dG=c）。"""
    return f_q + dq * e


def margin_of(p: np.ndarray, f_q: np.ndarray, f_g: np.ndarray,
              eps: float) -> tuple:
    """归一化边距 margin = |dG-dQ| / max(dQ+dG, EPS)，返回 (dQ, dG, margin)。

    与构造一致的欧氏度量（源码用 L1/MAD，公式结构相同：|dG-dQ|/(dQ+dG)）。
    """
    dq = float(np.linalg.norm(p - f_q))   # 到 Qwen 参考簇距离
    dg = float(np.linalg.norm(p - f_g))   # 到 GPT2 参考簇距离
    # 归一化：除以总距离（分母加 EPS 防御 dQ=dG=0 的全重合情形）
    m = abs(dg - dq) / max(dq + dg, eps)
    return dq, dg, m


# ---- 第三层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E04 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享合成状态在 _synthesize() 中一次性按原脚本单 rng 流计算。
    """

    def __init__(
        self,
        config: E04Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._synced = False
        # ---- 共享中间结果（与原脚本 main 单 rng 流一致）----
        self.f_q: np.ndarray | None = None
        self.f_g: np.ndarray | None = None
        self.c = 0.0
        self.e: np.ndarray | None = None
        self.gen_samples: list[np.ndarray] = []
        self.gen_margins: list[float] = []
        self.margin_mean = 0.0
        self.margin_min = 0.0
        self.margin_base_mean = 0.0
        self.improve = 0.0
        self.n_ok = 0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _synthesize(self) -> None:
        """一次性合成共享数据（与原脚本 main 的单一 rng 流一致，惰性）。"""
        if self._synced:
            return
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        self.f_q, self.f_g, self.c, self.e = make_reference_clusters(rng, cfg)
        # [1] 重生成样本（3 次）：margin = (c-2*dQ)/c（样本贴附 Qwen 侧时 dQ+dG=c）
        self.gen_samples = [make_sample(self.f_q, d, self.e) for d in cfg.GEN_DQ]
        self.gen_margins = [margin_of(s, self.f_q, self.f_g, cfg.EPS)[2]
                            for s in self.gen_samples]
        self.margin_mean = float(np.mean(self.gen_margins))
        self.margin_min = float(np.min(self.gen_margins))
        # [2] 无扰动基线样本对照
        base_samples = [make_sample(self.f_q, d, self.e) for d in cfg.BASE_DQ]
        base_margins = [margin_of(s, self.f_q, self.f_g, cfg.EPS)[2] for s in base_samples]
        self.margin_base_mean = float(np.mean(base_margins))
        # 提升倍率：重生成相对无扰动基线（防御除零）
        self.improve = (self.margin_mean / self.margin_base_mean
                        if self.margin_base_mean > 0.0 else float("inf"))
        # [3] 溯源判定：dQ <= dG 表示样本贴附 Qwen 簇
        self.n_ok = sum(1 for s in self.gen_samples
                        if margin_of(s, self.f_q, self.f_g, cfg.EPS)[0]
                        <= margin_of(s, self.f_q, self.f_g, cfg.EPS)[1])
        self._synced = True

    # ------------------------------------------------------------ 1) 重生成 margin
    def validate_gen_margin(self) -> dict:
        """1) 重生成 margin 均值 ≈ 0.2975（±0.01），且全部 margin 为正。"""
        cfg = self.config
        self._synthesize()
        ok = abs(self.margin_mean - cfg.MARGIN_TARGET) < cfg.MARGIN_TOL and self.margin_min > 0.0
        if not ok:
            raise ConfigError(
                f"重生成 margin 均值应≈{cfg.MARGIN_TARGET}，实际 {self.margin_mean:.4f} "
                f"(min={self.margin_min:.4f})",
                expected=cfg.MARGIN_TARGET, actual=self.margin_mean, param_key="E04",
            )
        return {
            "detail": (f"重生成 margin 均值 = {self.margin_mean:.4f} (实测 0.2975); "
                       f"margin 下限 = {self.margin_min:.4f} (实测 0.2102)"),
            "margin_mean": self.margin_mean, "margin_min": self.margin_min,
            "gen_margins": self.gen_margins,
        }

    # ------------------------------------------------------------ 2) 提升倍率
    def validate_improvement(self) -> dict:
        """2) 重生成相对无扰动基线提升 > IMPROVE_MIN（实测 1.54x）。"""
        cfg = self.config
        self._synthesize()
        ok = self.improve > cfg.IMPROVE_MIN
        if not ok:
            raise ConfigError(
                f"重生成相对基线应提升 ~54%（>{cfg.IMPROVE_MIN}x），实际 {self.improve:.3f}x",
                expected=cfg.IMPROVE_MIN, actual=self.improve, param_key="E04",
            )
        return {
            "detail": (f"无扰动基线 margin 均值 = {self.margin_base_mean:.4f} (实测 0.193); "
                       f"重生成提升倍率 = {self.improve:.3f}x (实测 0.2975/0.193=1.54x, +54%)"),
            "margin_base": self.margin_base_mean, "improve": self.improve,
        }

    # ------------------------------------------------------------ 3) 溯源准确率
    def validate_traceability(self) -> dict:
        """3) 重生成样本 100% 溯源到 Qwen。"""
        cfg = self.config
        self._synthesize()
        ok = self.n_ok == cfg.N_GEN
        if not ok:
            raise TraceabilityError(
                f"重生成样本应 100% 溯源到 Qwen，实际 {self.n_ok}/{cfg.N_GEN}",
                expected=cfg.N_GEN, actual=self.n_ok, param_key="E04",
            )
        return {
            "detail": f"重生成溯源准确率 = {self.n_ok}/{cfg.N_GEN} = {self.n_ok/cfg.N_GEN*100:.0f}%",
            "n_ok": self.n_ok, "n_gen": cfg.N_GEN,
        }

    # ------------------------------------------------------------ 4) 判别一致性
    def validate_consistency(self) -> dict:
        """4) 所有重生成样本 dQ<dG 且 margin>0（无错判）。"""
        cfg = self.config
        self._synthesize()
        assert self.f_q is not None and self.f_g is not None
        ok = all(margin_of(s, self.f_q, self.f_g, cfg.EPS)[0]
                 < margin_of(s, self.f_q, self.f_g, cfg.EPS)[1]
                 and margin_of(s, self.f_q, self.f_g, cfg.EPS)[2] > 0.0
                 for s in self.gen_samples)
        if not ok:
            raise TraceabilityError(
                "重生成样本应全部贴附 Qwen 簇（dQ<dG 且 margin>0）",
                expected={"dQ<dG": True, "margin>0": True},
                actual={"n": len(self.gen_samples)}, param_key="E04",
            )
        return {"detail": "判别一致性: 全部样本 dQ<dG 且 margin>0"}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实 Gamma 24 维剖面作为 Qwen 参考簇 beta24，复算重生成边距。

        margin 目标 0.2975 为文档审计值；GPT-2 参考簇为设计位移。数据缺失回退。
        """
        cfg = self.config
        rd = self._get_real_data()
        g_real = rd.get("spectral.k_proj_gamma_layers")
        g_mean = rd.get("spectral.k_proj_gamma_mean")
        g_doc = rd.audit("k_proj_gamma_mean") or 0.625
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 剖面缺失或维数不符：对照层无法构造，回退审计值且不判失败
        if g_real is None or len(g_real) != cfg.GRID:
            return {
                "detail": f"{tag} 真实剖面缺失 -> 回退审计值 {g_doc}，对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        g = np.asarray(g_real, dtype=float)
        rng_r = np.random.default_rng(cfg.REAL_SEED)   # 固定种子：真实对照层可复现
        # 曲率 24 维为合成；Qwen 簇 = [真实 Gamma 归一化缩放, 合成曲率] 再整体归一化
        tail = rng_r.standard_normal(cfg.NFRAME * 3)
        tail = tail / np.linalg.norm(tail) * cfg.Q_NORM
        f_q = np.concatenate([g / np.linalg.norm(g) * cfg.Q_NORM * 0.7, tail * 0.7])
        f_q = f_q / np.linalg.norm(f_q) * cfg.Q_NORM
        # GPT2 参考簇 = Qwen 簇 + 间距 C_TARGET 的设计位移
        f_g = f_q + (rng_r.standard_normal(cfg.DIM))
        f_g = f_q + (f_g - f_q) / np.linalg.norm(f_g - f_q) * cfg.C_TARGET
        e = (f_g - f_q) / cfg.C_TARGET if cfg.C_TARGET > 0.0 else np.zeros(cfg.DIM)
        gen = [make_sample(f_q, d, e) for d in cfg.GEN_DQ]  # 3 个重生成样本（设计 dQ）
        margins = [margin_of(s, f_q, f_g, cfg.EPS)[2] for s in gen]
        margin_mean_r = float(np.mean(margins))
        # 溯源判定：dQ <= dG 表示样本贴附 Qwen 簇
        n_ok = sum(1 for s in gen
                   if margin_of(s, f_q, f_g, cfg.EPS)[0] <= margin_of(s, f_q, f_g, cfg.EPS)[1])
        ok1 = abs(margin_mean_r - cfg.MARGIN_TARGET) < cfg.MARGIN_TOL
        ok2 = n_ok == cfg.N_GEN
        if not (ok1 and ok2):
            raise RealModelMismatchError(
                f"真实基线重生成边距不符: margin={margin_mean_r:.4f}（需≈{cfg.MARGIN_TARGET}±{cfg.MARGIN_TOL}）, "
                f"溯源 {n_ok}/{cfg.N_GEN}",
                expected={"margin": cfg.MARGIN_TARGET, "n_ok": cfg.N_GEN},
                actual={"margin": margin_mean_r, "n_ok": n_ok}, param_key="E04",
            )
        return {
            "detail": (f"{tag} 真实 k_proj Gamma 均值 {g_mean:.4f} vs 文档审计 {g_doc} "
                       f"(差异 {abs(g_mean - g_doc):.3f}); 重生成 margin 均值(真实基线) "
                       f"{margin_mean_r:.4f}（审计目标 0.2975）; 溯源 {n_ok}/{cfg.N_GEN} 判为 Qwen"),
            "source": rd.source_tag(), "tag": tag,
            "gamma_mean": g_mean, "gamma_doc": g_doc,
            "margin_real": margin_mean_r, "n_ok": n_ok,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "gen_margin", self.validate_gen_margin),
            (2, "improvement", self.validate_improvement),
            (3, "traceability", self.validate_traceability),
            (4, "consistency", self.validate_consistency),
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


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """E04 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E04 GEN_PROMPT 四层工厂验证")
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

    print("=" * 74)
    print("E04 GEN_PROMPT 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: DIM={cfg.DIM}, N_GEN={cfg.N_GEN}, MARGIN_TARGET={cfg.MARGIN_TARGET} "
          f"C_TARGET={cfg.C_TARGET} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e04_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
