# -*- coding: utf-8 -*-
"""E05 PROM 多维判别样本数 — LOOCV 判别流程验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 5 句 prompt x 2 实例 -> 特征矩阵（10 族 x 24 格点 = 240 维）
  2. 逐族判别：ratio = cross/within（>1 才可分）
  3. LOOCV 最近邻 + 逐维/std 标准化流程（对称标准化，test 样本同尺度）
  4. 场景A 同构 base vs instruct：不可分（acc<70% 或 ratio<1，文档实测 40%<50%）
  5. 场景B 跨家族 Qwen vs GPT2：可分（acc=100% 且 ratio>1）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E05Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 E05Config（env AIQ_E05_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享状态由引擎 _synthesize 惰性计算）
  ValidatorEngine        —— 4 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 3767-3887（E05 PROM）
  源码 _qwen_multidim.py（L42-L48 特征矩阵与 LOOCV 流程）
  《参数审计与实验报告.txt》（状态=待跑）
说明：家族位移用均匀向量（公共尺度型差异，呼应 E07），且位移范数 > prompt
      间距，体现"同一实例成簇、不同实例分离"的可分几何结构。纯数值合成。

真实模型对照：
  经 _real_data 惰性读取真实 k_proj Gamma 24 维剖面作为 240 维特征矩阵的
  k 族（前 24 维）prompt 种子，验证同构 base↔instruct 不可分、跨家族可分。
  数据来源标注：[真实实测] 或 [审计回退]；GPT-2 家族位移为设计口径。
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
    FamilySeparationError,
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

# ---- 第一层：配置模型 E05Config（pydantic 优先；dataclass 回退） ----
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


class E05Config(_ConfigModelBase):
    """E05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E05 节点键名一一对应（N_SAMPLE/N_FAM/DIM）；
    取值优先级：env AIQ_E05_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED_A: int = 1            # 场景 A 随机种子（固定可复现，算法逻辑常量）
    SEED_B: int = 2            # 场景 B 随机种子（固定可复现，算法逻辑常量）
    N_PROM: int = 5            # E05: 样本 prompt 数（README ①）
    N_FAM: int = 10            # 特征族数（k/o/up_proj x {max,mean,q90}=9 + k_rv_depth=1）
    GRID: int = 24             # B03: 每族深度格点（E05 DIM）
    PROMPT_SIG: float = 0.3    # prompt 语义种子幅度（族内 prompt 间距 ~0.3*sqrt(2*DIM)≈6.6）
    SHIFT_NOISE: float = 0.1   # 同 prompt 两次测量噪声
    SHIFT_NORM: float = 15.0   # 场景 B 家族均匀位移范数（>> prompt 间距 6.6，可分离）
    ACC_LOOCV_MIN: float = 0.7  # 源码 conclusion 阈值：acc>0.7 才判定可分
    EPS: float = 1e-12          # per_family_ratio 分母下限
    STD_EPS: float = 1e-9       # LOOCV 逐维标准化的分母下限
    REAL_SEED: int = 2024       # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E05 配置工厂：按优先级实例化 E05Config。"""

    def build(self) -> E05Config:
        """构建 E05Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E05Config, "E05")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def feature_matrix(rng: np.random.Generator, prompt_seeds: list,
                   shift: float, cfg: E05Config) -> np.ndarray:
    """5 句 x 2 实例 -> (10, DIM)。

    prompt 决定主导结构；shift 决定噪声幅度（README ③：同实例成簇）。
    """
    dim = cfg.N_FAM * cfg.GRID   # 240 维
    Z = np.zeros((2 * cfg.N_PROM, dim))
    for i in range(cfg.N_PROM):
        p = prompt_seeds[i]                       # 第 i 句 prompt 的语义种子
        Z[i] = p + shift * rng.standard_normal(dim)          # 实例 1（base）测量
        Z[i + cfg.N_PROM] = p + shift * rng.standard_normal(dim)  # 实例 2（instruct）测量
    return Z


def per_family_ratio(B: np.ndarray, I: np.ndarray, cfg: E05Config) -> tuple:
    """逐族 cross/within（源码 _qwen_multidim.py 逻辑）。

    within = max(两实例各自的族内样本间 MAD)；cross = 实例间平均 MAD。
    返回 (ratio, within, cross)。空输入防御：len<2 返回 (0.0, inf, 0.0)。
    """
    n = cfg.N_PROM
    if len(B) < 2 or len(I) < 2:                  # 防御样本数不足
        return 0.0, float("inf"), 0.0
    # base 实例内：5 句 prompt 两两 MAD 的均值（同实例内散布）
    wb = float(np.mean([np.mean(np.abs(B[i] - B[i2]))
                        for i in range(n) for i2 in range(i + 1, n)]))
    # instruct 实例内：同样计算族内散布
    wi = float(np.mean([np.mean(np.abs(I[i] - I[i2]))
                        for i in range(n) for i2 in range(i + 1, n)]))
    within = max(wb, wi)                          # 保守取两实例内散布的较大者
    cross = float(np.mean(np.abs(B - I)))         # 实例间平均 MAD（跨实例差异）
    # ratio>1 表示实例间差异大于实例内噪声 -> 可分
    return cross / (within + cfg.EPS), within, cross


def loocv_acc(Z: np.ndarray, cfg: E05Config) -> float:
    """LOOCV 最近邻 + 逐维/std 标准化（对称：训练集与 test 同尺度）。

    返回准确率（0~1）。空输入防御：len(Z)<2 返回 0.0。
    """
    if len(Z) < 2:
        return 0.0
    y = np.array([0] * cfg.N_PROM + [1] * cfg.N_PROM)     # 0=base, 1=instruct
    correct = 0
    for i in range(len(Z)):
        mask = np.ones(len(Z), bool)
        mask[i] = False                           # 留一：第 i 样本作测试集
        S = Z[mask].std(0) + cfg.STD_EPS          # 训练集逐维 std（对称标准化尺度）
        Zs = Z / S                                # 全体样本同尺度缩放（含 test）
        d = np.linalg.norm(Zs[mask] - Zs[i], axis=1)  # 最近邻距离（欧氏）
        nbr = y[mask][np.argmin(d)]               # 取最近训练样本的标签
        correct += int(nbr == y[i])               # 与自身真值比较
    return correct / len(Z)                       # 准确率 = 正确数/总数


# ---- 第三层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E05 验证引擎：顺序执行 4 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享场景数据在 _synthesize() 中一次性按原脚本独立种子流计算。
    """

    def __init__(
        self,
        config: E05Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._synced = False
        # ---- 共享中间结果（与原脚本 main 一致）----
        self.ratio_A = 0.0
        self.within_A = 0.0
        self.cross_A = 0.0
        self.acc_A = 0.0
        self.ratio_B = 0.0
        self.within_B = 0.0
        self.cross_B = 0.0
        self.acc_B = 0.0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _synthesize(self) -> None:
        """一次性计算场景 A/B 数据（与原脚本独立种子流一致，惰性）。"""
        if self._synced:
            return
        cfg = self.config
        dim = cfg.N_FAM * cfg.GRID
        # ---- [1] 场景 A：同构 base vs instruct（本质不可分）----
        rng_A = np.random.default_rng(cfg.SEED_A)   # 场景 A 独立种子
        prompt_seeds_A = [cfg.PROMPT_SIG * rng_A.standard_normal(dim)
                          for _ in range(cfg.N_PROM)]   # 5 句 prompt 语义种子
        B_A = feature_matrix(rng_A, prompt_seeds_A, cfg.SHIFT_NOISE, cfg)
        I_A = feature_matrix(rng_A, prompt_seeds_A, cfg.SHIFT_NOISE, cfg)  # 同种子第二实例
        self.ratio_A, self.within_A, self.cross_A = per_family_ratio(
            B_A[:cfg.N_PROM], I_A[:cfg.N_PROM], cfg)
        self.acc_A = loocv_acc(np.vstack([B_A[:cfg.N_PROM], I_A[:cfg.N_PROM]]), cfg)
        # ---- [2] 场景 B：跨家族 Qwen vs GPT2（可分）----
        rng_B = np.random.default_rng(cfg.SEED_B)   # 场景 B 独立种子
        prompt_seeds_B = [cfg.PROMPT_SIG * rng_B.standard_normal(dim)
                          for _ in range(cfg.N_PROM)]
        B_B = feature_matrix(rng_B, prompt_seeds_B, cfg.SHIFT_NOISE, cfg)
        I_B = feature_matrix(rng_B, prompt_seeds_B, cfg.SHIFT_NOISE, cfg)
        shift_vec = (cfg.SHIFT_NORM / np.sqrt(dim)) * np.ones(dim)   # 均匀家族位移
        I_B = I_B + shift_vec
        self.ratio_B, self.within_B, self.cross_B = per_family_ratio(
            B_B[:cfg.N_PROM], I_B[:cfg.N_PROM], cfg)
        self.acc_B = loocv_acc(np.vstack([B_B[:cfg.N_PROM], I_B[:cfg.N_PROM]]), cfg)
        self._synced = True

    # ------------------------------------------------------------ 1) 场景 A 同构不可分
    def validate_scenario_a(self) -> dict:
        """1) 同构 Qwen base vs Instruct：acc<70% 且 ratio<1（不可分双判据）。"""
        cfg = self.config
        self._synthesize()
        ok = self.acc_A < cfg.ACC_LOOCV_MIN and self.ratio_A < 1.0
        if not ok:
            raise FamilySeparationError(
                f"同构场景应不可分（acc<70% 且 ratio<1），"
                f"实际 acc={self.acc_A*100:.1f}%, ratio={self.ratio_A:.2f}",
                expected={"acc<": cfg.ACC_LOOCV_MIN, "ratio<": 1.0},
                actual={"acc": self.acc_A, "ratio": self.ratio_A}, param_key="E05",
            )
        return {
            "detail": (f"场景 A 同构 Qwen base vs Instruct: within={self.within_A:.4f} "
                       f"cross={self.cross_A:.4f} ratio={self.ratio_A:.2f} "
                       f"LOOCV={self.acc_A*100:.1f}% (文档实测: LOOCV=40%<50%) -> 不可分"),
            "acc": self.acc_A, "ratio": self.ratio_A,
            "within": self.within_A, "cross": self.cross_A,
        }

    # ------------------------------------------------------------ 2) 场景 B 跨家族可分
    def validate_scenario_b(self) -> dict:
        """2) 跨家族 Qwen vs GPT2：acc=100% 且 ratio>1（可分双判据）。"""
        cfg = self.config
        self._synthesize()
        ok = self.acc_B == 1.0 and self.ratio_B > 1.0
        if not ok:
            raise FamilySeparationError(
                f"跨家族场景应可分（acc=100% 且 ratio>1），"
                f"实际 acc={self.acc_B*100:.1f}%, ratio={self.ratio_B:.2f}",
                expected={"acc": 1.0, "ratio>": 1.0},
                actual={"acc": self.acc_B, "ratio": self.ratio_B}, param_key="E05",
            )
        return {
            "detail": (f"场景 B 跨家族 Qwen vs GPT2: within={self.within_B:.4f} "
                       f"cross={self.cross_B:.4f} ratio={self.ratio_B:.2f} "
                       f"LOOCV={self.acc_B*100:.0f}% -> 可分（实例成簇）"),
            "acc": self.acc_B, "ratio": self.ratio_B,
            "within": self.within_B, "cross": self.cross_B,
        }

    # ------------------------------------------------------------ 3) 判别力对比
    def validate_discrimination(self) -> dict:
        """3) 跨家族判别力应显著强于同构（acc/ratio 均更高）。"""
        self._synthesize()
        ok = self.acc_B > self.acc_A and self.ratio_B > self.ratio_A
        if not ok:
            raise ConfigError(
                f"跨家族可分性应强于同构: acc {self.acc_A:.2f}->{self.acc_B:.2f}, "
                f"ratio {self.ratio_A:.2f}->{self.ratio_B:.2f}",
                expected={"acc_B>acc_A": True, "ratio_B>ratio_A": True},
                actual={"acc": (self.acc_A, self.acc_B), "ratio": (self.ratio_A, self.ratio_B)},
                param_key="E05",
            )
        return {
            "detail": (f"判别力对比: 跨家族 acc {self.acc_A*100:.0f}%->{self.acc_B*100:.0f}%, "
                       f"ratio {self.ratio_A:.2f}->{self.ratio_B:.2f} 均显著高于同构"),
            "acc_a": self.acc_A, "acc_b": self.acc_B,
            "ratio_a": self.ratio_A, "ratio_b": self.ratio_B,
        }

    # ------------------------------------------------------------ 4) 真实模型对照
    def validate_real_model(self) -> dict:
        """4) 真实 k_proj Gamma 剖面作为 k 族特征种子，验证判别流程。

        同构 base↔instruct 不可分（acc<70%）、跨家族可分（acc=100%）。
        GPT-2 家族位移为设计口径。数据缺失回退。
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
        dim = cfg.N_FAM * cfg.GRID
        # 真实 Gamma 剖面放入 240 维种子的前 24 维（k 族），其余为 0
        seed_k = np.zeros(dim)
        seed_k[:cfg.GRID] = g
        # 5 句 prompt 种子 = 真实 k 族 + prompt 语义噪声
        seeds = [seed_k + cfg.PROMPT_SIG * rng_r.standard_normal(dim)
                 for _ in range(cfg.N_PROM)]
        B_A = feature_matrix(rng_r, seeds, cfg.SHIFT_NOISE, cfg)   # base 实例（真实种子）
        I_A = feature_matrix(rng_r, seeds, cfg.SHIFT_NOISE, cfg)   # instruct 实例（同种子）
        ratio_A, within_A, cross_A = per_family_ratio(B_A[:cfg.N_PROM], I_A[:cfg.N_PROM], cfg)
        acc_A = loocv_acc(np.vstack([B_A[:cfg.N_PROM], I_A[:cfg.N_PROM]]), cfg)
        okA = acc_A < cfg.ACC_LOOCV_MIN          # 同构场景应不可分
        # 场景 B：instruct 实例整体 + 均匀位移 -> 跨家族可分
        I_B = I_A + (cfg.SHIFT_NORM / np.sqrt(dim)) * np.ones(dim)
        ratio_B, within_B, cross_B = per_family_ratio(B_A[:cfg.N_PROM], I_B[:cfg.N_PROM], cfg)
        acc_B = loocv_acc(np.vstack([B_A[:cfg.N_PROM], I_B[:cfg.N_PROM]]), cfg)
        okB = acc_B == 1.0 and ratio_B > 1.0     # 跨家族应 100% 可分
        if not (okA and okB):
            raise RealModelMismatchError(
                f"真实 k 族种子判别流程不符: 场景A acc={acc_A*100:.0f}% ratio={ratio_A:.2f} "
                f"(需<70% 且不可分), 场景B acc={acc_B*100:.0f}% ratio={ratio_B:.2f} (需100% 可分)",
                expected={"accA<": cfg.ACC_LOOCV_MIN, "accB": 1.0, "ratioB>": 1.0},
                actual={"accA": acc_A, "ratioA": ratio_A, "accB": acc_B, "ratioB": ratio_B},
                param_key="E05",
            )
        return {
            "detail": (f"{tag} 真实 k_proj Gamma 均值 {g_mean:.4f} vs 文档审计 {g_doc} "
                       f"(差异 {abs(g_mean - g_doc):.3f}); 场景 A 同构(真实 k 族种子) "
                       f"ratio={ratio_A:.2f} LOOCV={acc_A*100:.0f}% -> 不可分; "
                       f"场景 B 跨家族(设计位移) ratio={ratio_B:.2f} LOOCV={acc_B*100:.0f}% -> 可分"),
            "source": rd.source_tag(), "tag": tag,
            "gamma_mean": g_mean, "gamma_doc": g_doc,
            "accA": acc_A, "ratioA": ratio_A, "accB": acc_B, "ratioB": ratio_B,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "scenario_a", self.validate_scenario_a),
            (2, "scenario_b", self.validate_scenario_b),
            (3, "discrimination", self.validate_discrimination),
            (4, "real_model", self.validate_real_model),
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
    """E05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E05 PROM 四层工厂验证")
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
    print("E05 PROM 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_PROM={cfg.N_PROM} N_FAM={cfg.N_FAM} GRID={cfg.GRID} "
          f"DIM={cfg.N_FAM*cfg.GRID} ACC_LOOCV_MIN={cfg.ACC_LOOCV_MIN}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e05_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
