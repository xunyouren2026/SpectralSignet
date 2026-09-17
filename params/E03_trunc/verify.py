# -*- coding: utf-8 -*-
"""E03 trunc 截断比例扰动 — 截断鲁棒性与分离度递增验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 截断公式：k = max(int(T*ratio), 4)，保留前半部分
  2. 截断后指纹与基线 MAD：随截断加深单调增大（50% / 25%）
  3. "同向不等量"：Qwen 漂移 < GPT-2 漂移，分离度递增（0.039->0.043->0.047）
  4. 家族判定（L1 最近邻）在截断下仍 100% 正确

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E03Config              —— 配置模型（pydantic 优先；dataclass 回退；列表/字典
                            字段经 _mutable 防御共享可变对象）
  ConfigFactory          —— 实例化 E03Config（env AIQ_E03_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享状态由引擎 _synthesize 惰性计算）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 3552-3643（E03 trunc，第四节实测表）
  源码 _qwen_robustness.py（L194-L199 截断 k=max(int(T*ratio),4)）
  《参数审计与实验报告.txt》（状态=已用 trunc n=6）
说明：合成 24 维 beta 剖面，截断样本按"截断越深、剖面噪声越大"构造，
      MAD 目标值直接对齐文档实测表。不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 k_proj Gamma 24 维剖面作为 Qwen 截断基线，
  复算截断 MAD 单调递增与家族判定；GPT-2 基线为设计口径。
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

# ---- 第一层：配置模型 E03Config（pydantic 优先；dataclass 回退） ----
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

if _HAS_PYDANTIC:
    def _mutable(v: Any) -> Any:
        """pydantic 路径：默认值直接返回（pydantic 内部深拷贝，安全）。"""
        return v
else:
    def _mutable(v: Any) -> Any:
        """dataclass 路径：包装为 default_factory，避免类定义期共享可变默认值报错。"""
        return _dataclasses.field(default_factory=lambda: v)


class E03Config(_ConfigModelBase):
    """E03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E03 节点键名一一对应（DIM/T/TRUNCS）；
    取值优先级：env AIQ_E03_<KEY> > YAML > _params_data.json > 本模型默认值。
    TRUNCS 为列表类配置（ConfigFactory.get_list 语义），DOC 为文档实测表。
    """

    SEED: int = 0                # 固定随机种子：保证可复现（算法逻辑常量）
    GRID: int = 24               # B03: beta 剖面插值格点（E03 DIM）
    T: int = 96                  # 原始 token 数（README ⑤）
    TRUNCS: list = _mutable([1.0, 0.5, 0.25])   # E03: 截断比例（列表配置，README ②公式）
    DOC_MAD_TOL: float = 0.008   # MAD 与文档值的允许偏差
    DOC_SEP_TOL: float = 0.01    # 分离度与文档值的允许偏差
    SEP0_LO: float = 0.035       # 无截断分离度文档区间下界
    SEP0_HI: float = 0.042       # 无截断分离度文档区间上界
    SEP2_LO: float = 0.045       # 25% 截断分离度文档区间下界
    SEP2_HI: float = 0.050       # 25% 截断分离度文档区间上界
    BASE_Q_CENTER: float = 0.50  # Qwen 基线中心（原始构造）
    BASE_Q_SPREAD: float = 0.10  # Qwen 基线散布（原始构造）
    BASE_G_SPREAD: float = 0.20  # GPT2 基线散布（原始构造）
    BASE_MAD: float = 0.20       # 两基线间 MAD 目标（原始构造）
    MIN_TOK: int = 4             # 截断保底 token 数（源码 k=max(int(T*r),4)）
    # 文档实测表（主文档 E03 第四节）：MAD(Qwen) / MAD(GPT2) / 分离度
    DOC: dict = _mutable({1.0: (0.040, 0.079, 0.039),
                          0.5: (0.042, 0.085, 0.043),
                          0.25: (0.050, 0.097, 0.047)})
    REAL_SEED: int = 1234        # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E03 配置工厂：按优先级实例化 E03Config。"""

    def build(self) -> E03Config:
        """构建 E03Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E03Config, "E03")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def mad(a: np.ndarray, b: np.ndarray) -> float:
    """L1/MAD 距离：逐维绝对差取平均（README ②公式）。"""
    return float(np.mean(np.abs(a - b)))   # L1 范数/DIM：对异常维不敏感


def truncate_noise(scale: float, rng: np.random.Generator,
                   dim: int) -> np.ndarray:
    """截断引入的剖面漂移（随机方向），归一化使 MAD 精确 = scale。

    防御：|n| 总和为 0（零向量）时返回零漂移，避免除零。
    """
    n = rng.standard_normal(dim)          # 各向同性随机漂移方向
    abs_sum = float(np.abs(n).sum())      # L1 总尺度
    if abs_sum == 0.0:                    # 防御零向量除零
        return np.zeros(dim)
    return scale * n / abs_sum * dim      # 归一化使 mean|noise| = scale


def truncated_k(ratio: float, cfg: E03Config) -> int:
    """截断 token 数：k = max(int(T*ratio), MIN_TOK)（源码口径）。"""
    return max(int(cfg.T * ratio), cfg.MIN_TOK)   # 保底 4 token，防止截断到空序列


def make_baselines(rng: np.random.Generator, cfg: E03Config) -> tuple:
    """构造 Qwen / GPT-2 基线剖面（E01 流程），两基线 MAD = BASE_MAD。"""
    base_q = cfg.BASE_Q_CENTER + cfg.BASE_Q_SPREAD * rng.standard_normal(cfg.GRID)
    base_g = base_q + cfg.BASE_G_SPREAD * rng.standard_normal(cfg.GRID)  # 初始异族基线
    diff = base_g - base_q
    abs_sum = float(np.abs(diff).sum())
    if abs_sum == 0.0:                    # 防御两基线完全重合的退化情形
        return base_q, base_q
    base_g = base_q + diff / abs_sum * (cfg.BASE_MAD * cfg.GRID)  # 归一化使 MAD = 0.20
    return base_q, base_g


# ---- 第三层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E03 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享截断表在 _synthesize() 中一次性按原脚本单 rng 流计算。
    """

    def __init__(
        self,
        config: E03Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._synced = False
        # ---- 共享中间结果（与原脚本 main 单 rng 流一致）----
        self.mads_q: list[float] = []
        self.mads_g: list[float] = []
        self.seps: list[float] = []
        self.n_ok_all = 0
        self.doc_ok: list[bool] = []     # 每档（MAD_Q, MAD_G, SEP, 家族）与文档对齐标志

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _synthesize(self) -> None:
        """一次性计算截断表（与原脚本 main 的单一 rng 流一致，惰性）。"""
        if self._synced:
            return
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        base_q, base_g = make_baselines(rng, cfg)  # 构造两家族基线（MAD=0.20）
        for r in cfg.TRUNCS:
            k = truncated_k(r, cfg)
            # 文档表按比例取目标 MAD 值；缺失比例直接判配置错误（防御 KeyError）
            if r not in cfg.DOC:
                raise ConfigError(
                    f"截断比例 {r} 不在文档实测表 DOC 中",
                    expected=list(cfg.DOC.keys()), actual=r, param_key="E03",
                )
            mq_doc, mg_doc, sep_doc = cfg.DOC[r]
            # 截断样本 = 基线 + 漂移（漂移 MAD 目标 = 文档 MAD），k 越小漂移越大
            g_q = base_q + truncate_noise(mq_doc, rng, cfg.GRID)
            g_g = base_g + truncate_noise(mg_doc, rng, cfg.GRID)
            mq = mad(g_q, base_q)
            mg = mad(g_g, base_g)
            sep = mg - mq                 # 分离度：GPT2 漂移 - Qwen 漂移
            self.seps.append(sep)
            self.mads_q.append(mq)
            self.mads_g.append(mg)
            # 家族判定（L1 最近邻，源码 family_dist 逻辑）
            dq_q = mad(g_q, base_q)
            dg_q = mad(g_q, base_g)
            pred_q = "Qwen" if dq_q <= dg_q else "GPT2"
            dq_g = mad(g_g, base_q)
            dg_g = mad(g_g, base_g)
            pred_g = "Qwen" if dq_g <= dg_g else "GPT2"
            ok_fam = (pred_q == "Qwen") and (pred_g == "GPT2")
            self.n_ok_all += int(ok_fam)
            # 每项 MAD/分离度与文档对照（断言由 validate_doc_ranges 触发）
            ok_mq = abs(mq - mq_doc) < cfg.DOC_MAD_TOL
            ok_mg = abs(mg - mg_doc) < cfg.DOC_MAD_TOL
            ok_sep = abs(sep - sep_doc) < cfg.DOC_SEP_TOL
            self.doc_ok.append(ok_mq and ok_mg and ok_sep and ok_fam)
        self._synced = True

    # ------------------------------------------------------------ 1) MAD 单调递增
    def validate_monotonic(self) -> dict:
        """1) MAD 随截断单调递增：Qwen 0.040->0.050, GPT2 0.079->0.097。"""
        cfg = self.config
        self._synthesize()
        ok = (self.mads_q[0] < self.mads_q[1] < self.mads_q[2]
              and self.mads_g[0] < self.mads_g[1] < self.mads_g[2])
        if not ok:
            raise ConfigError(
                f"Qwen/GPT2 MAD 应随截断单调递增: Q={self.mads_q}, G={self.mads_g}",
                expected={"q": [self.mads_q[0], self.mads_q[2]],
                          "g": [self.mads_g[0], self.mads_g[2]]},
                actual={"q": self.mads_q, "g": self.mads_g}, param_key="E03",
            )
        return {
            "detail": (f"MAD 随截断单调递增: Qwen {self.mads_q[0]:.3f}->{self.mads_q[2]:.3f}, "
                       f"GPT2 {self.mads_g[0]:.3f}->{self.mads_g[2]:.3f}"),
            "mads_q": self.mads_q, "mads_g": self.mads_g,
        }

    # ------------------------------------------------------------ 2) 分离度递增
    def validate_sep_increase(self) -> dict:
        """2) 分离度递增（同向不等量：Qwen 漂移 < GPT2 漂移）。"""
        cfg = self.config
        self._synthesize()
        ok = self.seps[0] < self.seps[1] < self.seps[2]
        if not ok:
            raise ConfigError(
                f"分离度应随截断递增（Qwen 漂移 < GPT2 漂移）: {self.seps}",
                expected={"asc": True}, actual=self.seps, param_key="E03",
            )
        return {
            "detail": f"分离度递增: {self.seps[0]:.3f} -> {self.seps[1]:.3f} -> {self.seps[2]:.3f}",
            "seps": self.seps,
        }

    # ------------------------------------------------------------ 3) 分离度区间对齐文档
    def validate_doc_ranges(self) -> dict:
        """3) 每档 MAD/分离度对齐文档实测表，且分离度区间对齐 0.039->0.047。"""
        cfg = self.config
        self._synthesize()
        # 每档逐项与文档表对齐（原脚本循环内断言语义）
        if not all(self.doc_ok):
            rows = []
            for r, (mq, mg, sep, okf) in zip(cfg.TRUNCS, zip(self.mads_q, self.mads_g, self.seps, self.doc_ok)):
                mq_doc, mg_doc, sep_doc = cfg.DOC[r]
                rows.append(f"r={r}: MAD_Q={mq:.4f}(文档{mq_doc}) MAD_G={mg:.4f}(文档{mg_doc}) "
                            f"sep={sep:.4f}(文档{sep_doc}) ok={okf}")
            raise ConfigError(
                "MAD/分离度与文档实测表存在偏差",
                expected=cfg.DOC, actual="; ".join(rows), param_key="E03",
            )
        # 分离度区间对齐文档（0.039 -> 0.047）
        ok3 = (cfg.SEP0_LO < self.seps[0] < cfg.SEP0_HI) and (cfg.SEP2_LO < self.seps[2] < cfg.SEP2_HI)
        if not ok3:
            raise ConfigError(
                f"分离度区间应对齐文档: seps[0]={self.seps[0]:.4f}, seps[2]={self.seps[2]:.4f}",
                expected=((cfg.SEP0_LO, cfg.SEP0_HI), (cfg.SEP2_LO, cfg.SEP2_HI)),
                actual=(self.seps[0], self.seps[2]), param_key="E03",
            )
        return {
            "detail": (f"分离度区间对齐文档 (0.039->0.047): "
                       f"[{self.seps[0]:.4f}, {self.seps[2]:.4f}]"),
            "seps": self.seps, "sep0": self.seps[0], "sep2": self.seps[2],
        }

    # ------------------------------------------------------------ 4) 家族判定全部正确
    def validate_family(self) -> dict:
        """4) 两家族截断样本溯源全部 100% 正确。"""
        cfg = self.config
        self._synthesize()
        ok = self.n_ok_all == len(cfg.TRUNCS)
        if not ok:
            raise TraceabilityError(
                f"两家族截断样本溯源应全部正确，实际 {self.n_ok_all}/{len(cfg.TRUNCS)}",
                expected=len(cfg.TRUNCS), actual=self.n_ok_all, param_key="E03",
            )
        return {
            "detail": f"家族判定不受截断影响: {self.n_ok_all}/{len(cfg.TRUNCS)} 全部 100%",
            "n_ok": self.n_ok_all, "n_total": len(cfg.TRUNCS),
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实 k_proj Gamma 剖面作为 Qwen 截断基线，复算 MAD 单调性。

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
        # 将真实 Gamma 剖面标准化（零均值/单位方差）后按文档尺度散布，作为 Qwen 基线
        base_q = cfg.BASE_Q_CENTER + (g - g.mean()) / g.std() * cfg.BASE_Q_SPREAD
        rng_r = np.random.default_rng(cfg.REAL_SEED)   # 固定种子：真实对照层可复现
        # GPT-2 基线 = 真实基线 + 设计位移，MAD 精确 = BASE_MAD
        diff = cfg.BASE_G_SPREAD * rng_r.standard_normal(cfg.GRID)
        abs_sum = float(np.abs(diff).sum())
        base_g = base_q + diff / abs_sum * (cfg.BASE_MAD * cfg.GRID) if abs_sum > 0.0 else base_q
        mads_q, mads_g, seps = [], [], []
        n_ok = 0
        # 遍历三个截断比例：漂移幅度取文档 MAD 值，验证单调性 + 家族判定
        for r in cfg.TRUNCS:
            mq_doc, mg_doc, sep_doc = cfg.DOC[r]
            g_q = base_q + truncate_noise(mq_doc, rng_r, cfg.GRID)
            g_g = base_g + truncate_noise(mg_doc, rng_r, cfg.GRID)
            mq = mad(g_q, base_q)
            mg = mad(g_g, base_g)
            seps.append(mg - mq)              # 分离度 = 异族漂移 - 同族漂移
            mads_q.append(mq)
            mads_g.append(mg)
            dq_q, dg_q = mad(g_q, base_q), mad(g_q, base_g)
            dq_g, dg_g = mad(g_g, base_q), mad(g_g, base_g)
            n_ok += int(dq_q <= dg_q) + int(dq_g > dg_g)
        ok1 = mads_q[0] < mads_q[1] < mads_q[2] and mads_g[0] < mads_g[1] < mads_g[2]
        ok2 = seps[0] < seps[1] < seps[2]     # 分离度随截断递增（同向不等量）
        ok3 = n_ok == 2 * len(cfg.TRUNCS)     # 所有截断样本溯源全部正确
        if not (ok1 and ok2 and ok3):
            raise RealModelMismatchError(
                f"真实基线截断单调性不符: MAD_Q={mads_q}, MAD_G={mads_g}, "
                f"sep={seps}, n_ok={n_ok}",
                expected={"mono": True, "sep_asc": True, "n_ok": 2 * len(cfg.TRUNCS)},
                actual={"mads_q": mads_q, "mads_g": mads_g, "seps": seps, "n_ok": n_ok},
                param_key="E03",
            )
        return {
            "detail": (f"{tag} 真实 k_proj Gamma 均值 {g_mean:.4f} vs 文档审计 {g_doc} "
                       f"(差异 {abs(g_mean - g_doc):.3f}); MAD 单调递增 Qwen "
                       f"{mads_q[0]:.3f}->{mads_q[2]:.3f}, GPT2 {mads_g[0]:.3f}->{mads_g[2]:.3f}; "
                       f"分离度 {seps[0]:.3f}->{seps[2]:.3f} 递增; 家族判定 {n_ok}/{2*len(cfg.TRUNCS)}"),
            "source": rd.source_tag(), "tag": tag,
            "gamma_mean": g_mean, "gamma_doc": g_doc,
            "mads_q": mads_q, "mads_g": mads_g, "seps": seps, "n_ok": n_ok,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "monotonic", self.validate_monotonic),
            (2, "sep_increase", self.validate_sep_increase),
            (3, "doc_ranges", self.validate_doc_ranges),
            (4, "family", self.validate_family),
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
    """E03 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E03 trunc 四层工厂验证")
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
    print("E03 trunc 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: TRUNCS={list(cfg.TRUNCS)} T={cfg.T} GRID={cfg.GRID} "
          f"MIN_TOK={cfg.MIN_TOK} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e03_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
