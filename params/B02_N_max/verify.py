# -*- coding: utf-8 -*-
"""B02 N_max SPL 最大 Gegenbauer 谱阶 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. Gegenbauer 递推起点 C_0=1, C_1=2*alpha*x（对应源码 gegenbauer.py 三项递推）
  2. 高阶 C_n 与 scipy.special.eval_gegenbauer 交叉验证（n=0..8）
  3. C_2 闭式解校验
  4. N_max=8 -> (N_max+1)=9 个基函数，形状 (9, B)
  5. d=896, alpha=447 下：N_max=8 范数计算有限稳定（log 空间）；N_max=16 评估溢出风险
  6. 真实模型对照：真实 hidden=896 / KV 头维度 hd=64（arch 字段）→ alpha = hidden/2-1 = 447

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B02Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退；
                            ALPHA 由 D_HID 数学公式推导，禁止 torch/tensorflow，纯 numpy 递推）
  ConfigFactory           —— 实例化 B02Config（环境变量 AIQ_B02_<KEY> > YAML > _params_data.json > 默认）
  GegenbauerSynthesizer   —— Gegenbauer 递推求值（纯 numpy 三项递推，与源码 gegenbauer.py 同构）
  ValidatorEngine         —— 6 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 740-861（B02）
  《参数完整定义与公式.txt》B02 项
  源码 gegenbauer.py（L45-L54 三项递推；L96-L111 范数）
说明：纯 numpy 实现（禁止 torch/tensorflow），不加载任何大模型。运行时间数秒内。
  依赖 scipy.special.eval_gegenbauer 做交叉验证。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实 hidden=896 / KV 头维度 hd=64（arch 字段）→ alpha = hidden/2-1 = 447。
  如实呈现：常量 D_HID=896 与真实架构 hidden=896 一致（alpha=447 依据成立）。
=====================================================================
"""
import argparse
import io
import math
import os
import sys
import time
from typing import Any

import numpy as np
from scipy.special import eval_gegenbauer     # scipy 标准实现（交叉验证基准）

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 B02Config（pydantic 优先；dataclass 回退） ----
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


class B02Config(_ConfigModelBase):
    """B02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B02 节点键名一一对应；取值优先级：
    环境变量 AIQ_B02_<KEY> > YAML > _params_data.json > 本模型默认值。
    ALPHA 为派生值：由 D_HID 数学公式推导（超球面 S^{d-1}），由 ConfigFactory.build() 覆盖。
    """

    SEED: int = 0                    # H01 固定随机种子（本脚本确定性计算，仅供约定）——算法逻辑常量，保留
    N_MAX: int = 8                   # 最大 Gegenbauer 谱阶（决定基函数数量 = N_MAX+1）
    D_HID: int = 896                 # Qwen 0.5B 隐藏维度
    ALPHA: float = 447.0             # alpha = hidden/2-1（派生值，由 D_HID 计算覆盖）
    X_GRID: list = [-0.9, 0.9, 21]   # 自变量网格（超球面极角余弦的合理范围）
    ALPHA_SMALL: float = 2.5         # 测试用小 alpha（交叉验证避免大 alpha 数值病态）
    TOL_START: float = 1e-12         # 递推起点 C_0/C_1 误差容差——浮点容差，保留
    TOL_SCIPY: float = 1e-9          # 高阶 vs scipy 交叉验证误差容差——浮点容差，保留
    TOL_CLOSED: float = 1e-12        # C_2 闭式解误差容差——浮点容差，保留
    LARGE_NS: list = [0, 2, 8, 12, 16]   # d=896 下范数状态检查的谱阶
    N_PTS: int = 64                  # 基函数形状检查的采样点数量
    HD_REAL: int = 64                # 真实 KV 头维度（Qwen2.5-0.5B 架构断言基准）


# ---- 第二层：配置工厂 ConfigFactory（实例化 B02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B02 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B02Config。"""

    def build(self) -> B02Config:
        """构建 B02Config；ALPHA 由 D_HID 数学公式推导（alpha = D_HID/2 - 1）。"""
        cfg = self.build_model(B02Config, "B02")
        cfg.ALPHA = cfg.D_HID / 2.0 - 1.0   # alpha = 447（超球面 S^{d-1}）；数学公式，保留
        return cfg


# ---- 第三层：合成器 GegenbauerSynthesizer（算法与原脚本完全一致，纯 numpy） ----
class GegenbauerSynthesizer(_SynthBase):
    """B02 Gegenbauer 多项式求值合成器（纯 numpy 递推，与源码 gegenbauer.py 同构）。

    - eval(n, alpha, x)：统一求值入口，返回 numpy 数组；三项递推
      C_n = (2(n+α-1)x C_{n-1} - (n+2α-2) C_{n-2}) / n；
    - basis(n_max, alpha, xs)：沿新轴堆叠 n=0..n_max 基函数。
    """

    def __init__(self, cfg: B02Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def eval(self, n: int, alpha: float, x) -> np.ndarray:
        """纯 numpy 递推实现（与源码 gegenbauer.py 同构；禁止 torch/tensorflow）。"""
        x = np.asarray(x, dtype=float)
        if n < 0:
            raise ValueError("n 必须 >= 0")   # 防御：谱阶非负
        if n == 0:
            return np.ones_like(x)            # 起点 C_0 = 1
        if n == 1:
            return 2.0 * alpha * x            # 起点 C_1 = 2αx
        c_prev = np.ones_like(x)              # C_0
        c_curr = 2.0 * alpha * x              # C_1
        for k in range(1, n):
            # 三项递推公式（Gegenbauer 多项式标准递推，alpha 为参数）
            c_next = (2.0 * (k + alpha) * x * c_curr
                      - (k + 2.0 * alpha - 1.0) * c_prev) / (k + 1.0)
            c_prev, c_curr = c_curr, c_next
        return c_curr

    def basis(self, n_max: int, alpha: float, xs: np.ndarray) -> np.ndarray:
        """沿新轴堆叠 n=0..n_max 的基函数 → 形状 (n_max+1, len(xs))。"""
        return np.stack([self.eval(n, alpha, xs) for n in range(n_max + 1)], axis=0)

    def scalar(self, arr: np.ndarray) -> float:
        """把单元素 numpy 数组结果转为 Python 标量。"""
        return float(np.asarray(arr).reshape(-1)[0])


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def gegenbauer_norm_sq(n: int, a: float) -> float:
    """与源码 gegenbauer.py 同构的范数平方（朴素 float64 实现，可能溢出）。
    ||C_n||² = π·2^(1-2a)·Γ(n+2a) / (Γ(n+1)·(n+a)·Γ(a)²)。"""
    if a <= 0.0:
        return 1.0                       # 防御：非正 alpha 时范数退化为 1
    return (math.pi * 2.0 ** (1.0 - 2.0 * a)
            * math.exp(math.lgamma(n + 2.0 * a))          # Γ(n+2a) 可能巨大
            / (math.exp(math.lgamma(n + 1.0)) * (n + a)   # Γ(n+1)·(n+a)
               * math.exp(math.lgamma(a)) ** 2))          # Γ(a)²


def log_norm_sq(n: int, a: float) -> float:
    """log 空间计算范数平方，避免中间项溢出（数学上恒正且有限）。
    对 norm_sq 取对数：ln π + (1-2a)ln2 + lgamma(n+2a) - lgamma(n+1) - ln(n+a) - 2·lgamma(a)。"""
    return (math.log(math.pi) + (1.0 - 2.0 * a) * math.log(2.0)
            + math.lgamma(n + 2.0 * a) - math.lgamma(n + 1.0)
            - math.log(n + a) - 2.0 * math.lgamma(a))


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B02 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: B02Config,
        synth: GegenbauerSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 递推起点 C_0/C_1
    def validate_start(self) -> dict:
        """1) 递推起点：小 alpha(2.5) 与真实 alpha(447) 下 C_0=1、C_1=2αx。"""
        cfg = self.config
        ok1 = True
        rows = []
        # 在小 alpha（2.5）与真实 alpha（447）下都校验递推起点
        for a in [cfg.ALPHA_SMALL, cfg.ALPHA]:
            x0 = 0.3
            c0 = self.synth.scalar(self.synth.eval(0, a, np.array([x0])))       # C_0 实测
            c1 = self.synth.scalar(self.synth.eval(1, a, np.array([x0])))       # C_1 实测
            ok0 = abs(c0 - 1.0) < cfg.TOL_START            # C_0 须等于 1
            ok1c = abs(c1 - 2.0 * a * x0) < cfg.TOL_START  # C_1 须等于 2αx
            ok1 &= (ok0 and ok1c)
            rows.append(f"alpha={a:7.1f}: C_0={c0:.15f} (期望1) | C_1={c1:.10f} "
                        f"(期望2αx={2 * a * x0:.10f})")
        if not ok1:
            raise AIQValidationError(
                "递推起点 C_0/C_1 不符",
                expected={"C_0": 1.0, "C_1": "2αx"},
                actual=rows, param_key="B02",
            )
        return {"detail": "递推起点 C_0/C_1: " + "; ".join(rows)}

    # ------------------------------------------------------------ 2) 高阶交叉验证 vs scipy
    def validate_scipy(self) -> dict:
        """2) 高阶递推 vs scipy.special.eval_gegenbauer（alpha=2.5, n=0..8）。"""
        cfg = self.config
        x_pts = np.linspace(*cfg.X_GRID)   # 自变量网格（21 点）
        max_all = 0.0    # 所有阶最大误差
        ok2 = True
        rows = []
        for n in range(0, cfg.N_MAX + 1):                       # 扫描 n=0..8
            # 逐格点计算递推值（grid 上 21 个点）
            c_rec = np.array([self.synth.scalar(self.synth.eval(n, cfg.ALPHA_SMALL, np.array([xp])))
                              for xp in x_pts])
            c_scipy = eval_gegenbauer(n, cfg.ALPHA_SMALL, x_pts)   # scipy 参考值
            err = float(np.max(np.abs(c_rec - c_scipy)))       # 网格最大绝对误差
            max_all = max(max_all, err)
            one_ok = err < cfg.TOL_SCIPY
            ok2 &= one_ok
            rows.append(f"n={n:>3}: max|C_rec - C_scipy| = {err:.3e}")
        if not ok2:
            raise AIQValidationError(
                f"scipy 交叉验证最大误差过大: {max_all:.3e}",
                expected=cfg.TOL_SCIPY, actual=max_all, param_key="B02",
            )
        return {"detail": ("高阶递推 vs scipy.special.eval_gegenbauer "
                           f"(alpha={cfg.ALPHA_SMALL}, n=0..{cfg.N_MAX}): " + "; ".join(rows)),
                "max_err": max_all, "tol": cfg.TOL_SCIPY}

    # ------------------------------------------------------------ 3) C_2 闭式解
    def validate_closed_form(self) -> dict:
        """3) C_2 闭式解：递推值与 α(2(α+1)x²-1) 一致到机器精度。"""
        cfg = self.config
        x0 = 0.3
        c2_rec = self.synth.scalar(self.synth.eval(2, cfg.ALPHA_SMALL, np.array([x0])))   # 递推 C_2
        c2_closed = cfg.ALPHA_SMALL * (2 * (cfg.ALPHA_SMALL + 1) * x0 * x0 - 1)   # 闭式 C_2 = α(2(α+1)x²-1)
        err2 = abs(c2_rec - c2_closed)                       # 两口径偏差
        ok3 = err2 < cfg.TOL_CLOSED
        if not ok3:
            raise AIQValidationError(
                f"C_2 闭式解误差过大: {err2:.2e}",
                expected=cfg.TOL_CLOSED, actual=err2, param_key="B02",
            )
        return {"detail": (f"C_2 闭式解: 递推={c2_rec:.10f} 闭式α(2(α+1)x²-1)={c2_closed:.10f} "
                           f"误差={err2:.2e}"),
                "c2_rec": c2_rec, "c2_closed": c2_closed, "err": err2}

    # ------------------------------------------------------------ 4) N_max=8 -> 9 个基函数形状
    def validate_basis_shape(self) -> dict:
        """4) N_max=8 → 9 个基函数，形状 (9, N_PTS)。"""
        cfg = self.config
        n_pts = cfg.N_PTS                     # 采样点数量（数据层读取）
        xs = np.linspace(-0.9, 0.9, n_pts)                  # 自变量网格
        # 沿新轴堆叠 n=0..8 的基函数 → 形状 (9, 64)
        basis = self.synth.basis(cfg.N_MAX, cfg.ALPHA_SMALL, xs)
        shape = tuple(basis.shape)
        ok4 = (shape == (cfg.N_MAX + 1, n_pts))                  # 断言 (9, 64)
        if not ok4:
            raise AIQValidationError(
                f"基函数形状错误: {shape}",
                expected=(cfg.N_MAX + 1, n_pts), actual=shape, param_key="B02",
            )
        return {"detail": f"N_max={cfg.N_MAX} -> 基函数形状 {shape}（应为 ({cfg.N_MAX + 1}, {n_pts})）",
                "shape": shape}

    # ------------------------------------------------------------ 5) d=896 (alpha=447) 范数溢出状态
    def validate_norm_stability(self) -> dict:
        """5) d=896 (alpha=447) 下范数 ||C_n||^2 的 log 空间数值状态（须全部有限）。"""
        cfg = self.config
        ok5 = True
        rows = []
        for n in cfg.LARGE_NS:
            ln = log_norm_sq(n, cfg.ALPHA)                       # log 空间范数平方
            finite = math.isfinite(ln)                       # 须有限（log 空间策略的要点）
            ok5 &= finite
            try:
                naive = gegenbauer_norm_sq(n, cfg.ALPHA)     # 源码朴素 float64 实现
                naive_status = f"有限 = {naive:.6e}"
            except OverflowError:
                naive_status = "OverflowError（中间项 exp(lgamma) 超出 float64 上限 ~1.8e308）"
            rows.append(f"N_max={n:>2}: log||C_n||^2 = {ln:+.6f} | 朴素实现: {naive_status}")
        c8 = self.synth.scalar(self.synth.eval(8, cfg.ALPHA, np.array([0.3])))            # alpha=447 下 C_8
        c16 = self.synth.scalar(self.synth.eval(16, cfg.ALPHA, np.array([0.3])))          # alpha=447 下 C_16
        if not ok5:
            raise AIQValidationError(
                "d=896 下存在范数 log 空间不有限（溢出）",
                expected="全部 log 空间有限", actual=rows, param_key="B02",
            )
        return {"detail": (f"d={cfg.D_HID} (alpha={cfg.ALPHA:.0f}) 下范数状态: " + "; ".join(rows)
                           + f"; alpha=447, x=0.3: C_8={c8:.3e}, C_16={c16:.3e}"
                           + "（高阶递推值巨大，float32 会丢失精度）"),
                "c8": c8, "c16": c16}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 hidden=896 / hd=64 → alpha=447 公式成立。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        hidden_real = rd.get("arch.hidden")                  # 真实隐藏维度
        hd_real = rd.get("arch.hd")                          # 真实 KV 头维度
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        # 断言：真实 hidden 与常量一致、hd 为 HD_REAL(64)、alpha 公式成立
        ok6 = (hidden_real == cfg.D_HID) and (hd_real == cfg.HD_REAL) and (cfg.ALPHA == cfg.D_HID / 2.0 - 1.0)
        if not ok6:
            raise RealModelMismatchError(
                f"真实架构与常量不一致: hidden={hidden_real}, hd={hd_real}",
                expected={"hidden": cfg.D_HID, "hd": cfg.HD_REAL},
                actual={"hidden": hidden_real, "hd": hd_real},
                param_key="B02",
            )
        return {"detail": (f"{tag} 真实 hidden = {hidden_real}（与常量 D_HID={cfg.D_HID} 一致，"
                           f"alpha = hidden/2-1 = {cfg.ALPHA:.0f}，超球面参数成立）; "
                           f"真实 KV 头维度 hd = {hd_real}（Gegenbauer 基函数定义域基于真实隐藏维度）"),
                "source": rd.source_tag(), "tag": tag,
                "hidden_real": hidden_real, "hd_real": hd_real}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "start", self.validate_start),
            (2, "scipy_cross", self.validate_scipy),
            (3, "closed_form", self.validate_closed_form),
            (4, "basis_shape", self.validate_basis_shape),
            (5, "norm_stability", self.validate_norm_stability),
            (6, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except (AIQValidationError, ValueError) as e:  # 类型化异常 + 防御性 ValueError
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e),
                         "expected": getattr(e, "expected", None),
                         "actual": getattr(e, "actual", None),
                         "param_key": getattr(e, "param_key", None)}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # 结构化 JSON 日志（_logging 单例；每行可 json.loads）
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """B02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B02 N_max 四层工厂验证（纯 numpy）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = GegenbauerSynthesizer(cfg)                # ② 合成层（纯 numpy，禁止 torch/tensorflow）
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("B02 N_max 验证：Gegenbauer 递推 C_0=1, C_1=2*alpha*x  (backend=numpy, 禁 torch)")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_MAX={cfg.N_MAX} D_HID={cfg.D_HID} ALPHA={cfg.ALPHA:.0f} "
          f"ALPHA_SMALL={cfg.ALPHA_SMALL} N_PTS={cfg.N_PTS} HD_REAL={cfg.HD_REAL} "
          f"TOL_SCIPY={cfg.TOL_SCIPY:.0e} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b02_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
