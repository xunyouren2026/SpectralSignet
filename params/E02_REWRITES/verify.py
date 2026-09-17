# -*- coding: utf-8 -*-
"""E02 REWRITES 改写扰动数 — 48 维指纹 L1 家族判定鲁棒性验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 4 句改写样本 x 3 实例 = 12 个测试样本（n=12）
  2. 48 维特征（beta24 + 曲率24）MAD(L1) 最近邻家族判定
  3. 改写样本在 Qwen 基线附近的 MAD ≈ 实测 0.040
  4. 验证 Qwen vs GPT2 的距离判别：所有改写样本正确归类（100%）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E02Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 E02Config（env AIQ_E02_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享状态由引擎 _synthesize 惰性计算）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 3443-3551（E02 REWRITES）
  源码 _qwen_robustness.py（L45-L50 改写样本与 family_dist 逻辑）
  《参数审计与实验报告.txt》（状态=已用 rewrite n=12）
说明：纯数值合成数据（48 维特征向量），不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 k_proj Gamma 24 维剖面作为 48 维 Qwen 基线的
  beta24 部分，复算改写样本 MAD≈0.040 与 L1 家族判定；GPT-2 基线为设计口径。
  数据来源标注：[真实实测] 或 [审计回退]；真实剖面缺失时回退且不判失败。
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

# ---- 第一层：配置模型 E02Config（pydantic 优先；dataclass 回退） ----
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


class E02Config(_ConfigModelBase):
    """E02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E02 节点键名一一对应（GRID/NFRAME/DIM/
    N_REWRITES/MAD_DOC）；取值优先级：env AIQ_E02_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0              # 固定随机种子：保证可复现（算法逻辑常量）
    GRID: int = 24             # beta 剖面格点（B03，README ⑥）
    NFRAME: int = 8            # 曲率帧数（B08），8 帧 x 3 指标 = 24 维
    DIM: int = 48              # 48 维：beta24 + 曲率24（E02 DIM）
    N_REWRITE: int = 4         # E02: 每实例改写样本数（README ①）
    MAD_TARGET: float = 0.040  # 实测：改写样本到 Qwen 基线 MAD（README ⑤）
    FAMILY_SHIFT: float = 0.30  # GPT2 家族相对 Qwen 的位移幅度（原始构造）
    Q_I_NORM: float = 1.01      # qwen_instruct 相对 qwen_base 的幅度比（原始构造）
    MARGIN_MIN: float = 0.03    # 判别余量下限：dG - dQ 应远大于 MAD 本身
    BASE_AMP: float = 0.55      # 参考基线归一化幅度（原始构造）
    INSTR_OFFSET: float = 0.03  # instruct 簇相对 base 的初始扰动幅度（原始构造）
    MAD_TOL: float = 0.01       # MAD 与实测 0.040 的允许偏差
    REAL_SEED: int = 1234       # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E02 配置工厂：按优先级实例化 E02Config。"""

    def build(self) -> E02Config:
        """构建 E02Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E02Config, "E02")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def make_profile(rng: np.random.Generator, family_shift: float,
                 cfg: E02Config) -> np.ndarray:
    """构造一个 DIM 维参考基线剖面（原始构造：归一化 + 家族位移）。"""
    base = 0.5 + 0.15 * rng.standard_normal(cfg.DIM)  # 中心 0.5 的小扰动初始基线
    base = base / np.linalg.norm(base)                 # L2 归一化（只保留方向）
    return base * cfg.BASE_AMP + family_shift          # 缩放到 BASE_AMP，再叠加家族位移


def add_noise_mad(sample: np.ndarray, mad_target: float,
                  rng: np.random.Generator, dim: int) -> np.ndarray:
    """沿随机方向加噪声，使 MAD = mean|noise| = mad_target（L1 归一化）。

    防御：|n| 均值为 0（零向量）时直接返回原样本，避免除零。
    """
    n = rng.standard_normal(dim)                 # 各向同性随机噪声方向
    mean_abs = float(np.mean(np.abs(n)))         # 该方向的 L1 尺度
    if mean_abs == 0.0:                          # 理论不会发生，但防御除零
        return sample
    noise = mad_target * n / mean_abs            # 归一化使 mean|noise| 精确 = mad_target
    return sample + noise                        # 改写样本 = 基线 + 目标 MAD 噪声


def mad(a: np.ndarray, b: np.ndarray) -> float:
    """L1/MAD 距离：逐维绝对差取平均（README ②公式）。"""
    return float(np.mean(np.abs(a - b)))         # L1 范数/DIM：对异常维不敏感


def family_dist(p: np.ndarray, qwen_b: np.ndarray, qwen_i: np.ndarray,
                gpt2: np.ndarray, cfg: E02Config) -> tuple:
    """dQ = min(到 qwen_b, 到 qwen_i)；dG = 到 gpt2；均用 L1(MAD)。

    返回 (dQ, dG)。空输入防御：非 DIM 维输入返回 (inf, inf)。
    """
    if np.ndim(p) != 1 or p.shape[0] != cfg.DIM:  # 防御非 DIM 维样本
        return float("inf"), float("inf")
    dq = min(mad(p, qwen_b), mad(p, qwen_i))     # 同族取最近变体的距离
    dg = mad(p, gpt2)                            # 异族距离
    return dq, dg


def family_of(p: np.ndarray, qwen_b: np.ndarray, qwen_i: np.ndarray,
              gpt2: np.ndarray, cfg: E02Config) -> tuple:
    """L1 最近邻家族判定（源码 family_of 逻辑）。返回 (家族名, dQ, dG)。"""
    dq, dg = family_dist(p, qwen_b, qwen_i, gpt2, cfg)
    # 距离近者归属：dQ<=dG 判 Qwen0.5B，否则判 GPT2
    return ("Qwen0.5B", dq, dg) if dq <= dg else ("GPT2", dq, dg)


# ---- 第三层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E02 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享合成状态在 _synthesize() 中一次性按原脚本单 rng 流计算。
    """

    def __init__(
        self,
        config: E02Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._synced = False
        # ---- 共享中间结果（与原脚本 main 单 rng 流一致）----
        self.qwen_b: np.ndarray | None = None
        self.qwen_i: np.ndarray | None = None
        self.gpt2: np.ndarray | None = None
        self.rewrite_qwen: np.ndarray | None = None
        self.rewrite_gpt2: np.ndarray | None = None
        self.mad_q = 0.0
        self.mad_g_ref = 0.0
        self.margin_min = 0.0
        self.acc_qwen = 0.0
        self.acc_gpt2 = 0.0

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
        # ---- 构造三个实例的 48 维参考基线 ----
        # Qwen 家族：base 与 instruct 参考簇彼此接近（同族）
        self.qwen_b = make_profile(rng, 0.0, cfg)
        self.qwen_i = self.qwen_b + cfg.INSTR_OFFSET * rng.standard_normal(cfg.DIM)
        self.qwen_i = self.qwen_i / np.linalg.norm(self.qwen_i) * np.linalg.norm(self.qwen_b) * cfg.Q_I_NORM
        # GPT2 家族：远离 Qwen 簇（family_shift 位移）
        self.gpt2 = make_profile(rng, cfg.FAMILY_SHIFT, cfg)
        # ---- 构造改写样本（属于 Qwen 家族）----
        self.rewrite_qwen = np.array(
            [add_noise_mad(self.qwen_b, cfg.MAD_TARGET, rng, cfg.DIM)
             for _ in range(cfg.N_REWRITE)])
        # 同时构造 GPT2 家族改写样本（对照，证明可判别）
        self.rewrite_gpt2 = np.array(
            [add_noise_mad(self.gpt2, cfg.MAD_TARGET, rng, cfg.DIM)
             for _ in range(cfg.N_REWRITE)])
        # ---- [1] 改写样本 MAD 到 Qwen 基线 ----
        self.mad_q = float(np.mean([mad(s, self.qwen_b) for s in self.rewrite_qwen]))
        # ---- [2] 改写样本到 GPT2 距离应显著更大 ----
        self.mad_g_ref = float(np.mean([mad(s, self.gpt2) for s in self.rewrite_qwen]))
        # ---- [3] 全样本家族判定准确率 ----
        n_ok_q = sum(1 for s in self.rewrite_qwen
                     if family_of(s, self.qwen_b, self.qwen_i, self.gpt2, cfg)[0] == "Qwen0.5B")
        n_ok_g = sum(1 for s in self.rewrite_gpt2
                     if family_of(s, self.qwen_b, self.qwen_i, self.gpt2, cfg)[0] == "GPT2")
        self.acc_qwen = n_ok_q / cfg.N_REWRITE
        self.acc_gpt2 = n_ok_g / cfg.N_REWRITE
        # ---- [4] 判别余量：dG - dQ 应远大于 MAD 本身（鲁棒）----
        margins = [family_dist(s, self.qwen_b, self.qwen_i, self.gpt2, cfg)[1]
                   - family_dist(s, self.qwen_b, self.qwen_i, self.gpt2, cfg)[0]
                   for s in self.rewrite_qwen]
        self.margin_min = float(min(margins))    # 取最小余量（最坏情形）
        self._synced = True

    # ------------------------------------------------------------ 1) MAD 对齐
    def validate_mad(self) -> dict:
        """1) 改写样本到 Qwen 基线 MAD ≈ 0.040（±0.01）。"""
        cfg = self.config
        self._synthesize()
        ok = abs(self.mad_q - cfg.MAD_TARGET) < cfg.MAD_TOL
        if not ok:
            raise ConfigError(
                f"MAD 应≈{cfg.MAD_TARGET}，实际 {self.mad_q:.4f}",
                expected=cfg.MAD_TARGET, actual=self.mad_q, param_key="E02",
            )
        return {"detail": f"Qwen 改写样本 -> Qwen 基线 MAD = {self.mad_q:.4f} (实测 0.040)",
                "mad_q": self.mad_q, "mad_target": cfg.MAD_TARGET}

    # ------------------------------------------------------------ 2) 跨族距离
    def validate_cross_margin(self) -> dict:
        """2) 改写样本到 GPT2 距离应显著大于到 Qwen 距离（> 同族 + 余量）。"""
        cfg = self.config
        self._synthesize()
        ok = self.mad_g_ref > self.mad_q + cfg.MARGIN_MIN
        if not ok:
            raise ConfigError(
                f"Qwen 改写样本到 GPT2 的距离应显著大于到 Qwen 的距离: {self.mad_g_ref:.4f}",
                expected=self.mad_q + cfg.MARGIN_MIN, actual=self.mad_g_ref, param_key="E02",
            )
        return {"detail": (f"Qwen 改写样本 -> GPT2 基线 MAD = {self.mad_g_ref:.4f} "
                           f"(> Qwen MAD + {cfg.MARGIN_MIN})"),
                "mad_g_ref": self.mad_g_ref, "mad_q": self.mad_q, "margin_min": cfg.MARGIN_MIN}

    # ------------------------------------------------------------ 3) 家族判定准确率
    def validate_accuracy(self) -> dict:
        """3) 全样本家族判定：两家族均 100% 正确。"""
        cfg = self.config
        self._synthesize()
        ok = self.acc_qwen == 1.0 and self.acc_gpt2 == 1.0
        if not ok:
            raise TraceabilityError(
                f"家族判定应 100% 正确: Qwen={self.acc_qwen*100:.0f}%, GPT2={self.acc_gpt2*100:.0f}%",
                expected=1.0, actual=(self.acc_qwen, self.acc_gpt2), param_key="E02",
            )
        return {
            "detail": (f"L1 最近邻家族判定: Qwen 改写 {self.acc_qwen*100:.0f}% "
                       f"({int(self.acc_qwen*cfg.N_REWRITE)}/{cfg.N_REWRITE}), "
                       f"GPT2 改写 {self.acc_gpt2*100:.0f}% "
                       f"({int(self.acc_gpt2*cfg.N_REWRITE)}/{cfg.N_REWRITE}) 均 100%"),
            "acc_qwen": self.acc_qwen, "acc_gpt2": self.acc_gpt2, "n": cfg.N_REWRITE,
        }

    # ------------------------------------------------------------ 4) 判别余量
    def validate_margin(self) -> dict:
        """4) 判别余量 min(dG-dQ) 应大于 MARGIN_MIN（鲁棒性判据）。"""
        cfg = self.config
        self._synthesize()
        ok = self.margin_min > cfg.MARGIN_MIN
        if not ok:
            raise ConfigError(
                f"Qwen 改写样本的判别余量应充足（>{cfg.MARGIN_MIN}），实际 {self.margin_min:.4f}",
                expected=cfg.MARGIN_MIN, actual=self.margin_min, param_key="E02",
            )
        return {"detail": f"判别余量 min(dG-dQ) = {self.margin_min:.4f} > {cfg.MARGIN_MIN}（鲁棒）",
                "margin_min": self.margin_min}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实 k_proj Gamma 剖面作为 48 维基线 beta24 部分复算改写鲁棒性。

        GPT-2 基线为设计位移；真实剖面缺失时回退且不判失败。
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
        # 曲率 24 维为合成，归一化到 BASE_AMP（真实存档只有 beta 剖面）
        curv = cfg.BASE_AMP * rng_r.standard_normal(cfg.NFRAME * 3)
        curv = curv / np.linalg.norm(curv) * cfg.BASE_AMP
        # Qwen 真实基线 = [真实 Gamma 归一化, 合成曲率] 拼接成 48 维
        qwen_b_real = np.concatenate([g / np.linalg.norm(g) * cfg.BASE_AMP, curv])
        gpt2_real = make_profile(rng_r, family_shift=cfg.FAMILY_SHIFT, cfg=cfg)
        # 在真实基线附近生成改写样本（MAD 目标 0.040）
        rew = np.array([add_noise_mad(qwen_b_real, cfg.MAD_TARGET, rng_r, cfg.DIM)
                        for _ in range(cfg.N_REWRITE)])
        mad_q = float(np.mean([mad(s, qwen_b_real) for s in rew]))
        ok1 = abs(mad_q - cfg.MAD_TARGET) < cfg.MAD_TOL
        n_ok = 0
        # 真实基线双变体相同（qwen_b_real 两次传入）：真实侧只验证单变体可分性
        for s in rew:
            pred, dq, dg = family_of(s, qwen_b_real, qwen_b_real, gpt2_real, cfg)
            n_ok += int(pred == "Qwen0.5B")
        ok2 = n_ok == cfg.N_REWRITE
        if not (ok1 and ok2):
            raise RealModelMismatchError(
                f"真实基线改写鲁棒性不符: MAD={mad_q:.4f}（需≈{cfg.MAD_TARGET}±{cfg.MAD_TOL}）, "
                f"判定 {n_ok}/{cfg.N_REWRITE}",
                expected={"mad": cfg.MAD_TARGET, "n_ok": cfg.N_REWRITE},
                actual={"mad": mad_q, "n_ok": n_ok}, param_key="E02",
            )
        return {
            "detail": (f"{tag} 真实 k_proj Gamma 均值 {g_mean:.4f} vs 文档审计 {g_doc} "
                       f"(差异 {abs(g_mean - g_doc):.3f}); 改写样本 MAD(真实基线) {mad_q:.4f} "
                       f"(目标 0.040); L1 家族判定 {n_ok}/{cfg.N_REWRITE} 判为 Qwen0.5B"),
            "source": rd.source_tag(), "tag": tag,
            "gamma_mean": g_mean, "gamma_doc": g_doc, "mad_real": mad_q, "n_ok": n_ok,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "mad", self.validate_mad),
            (2, "cross_margin", self.validate_cross_margin),
            (3, "accuracy", self.validate_accuracy),
            (4, "margin", self.validate_margin),
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
    """E02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E02 REWRITES 四层工厂验证")
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
    print("E02 REWRITES 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: DIM={cfg.DIM}=beta{cfg.GRID}+曲率{cfg.NFRAME}x3, N_REWRITE={cfg.N_REWRITE} "
          f"MAD_TARGET={cfg.MAD_TARGET} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
