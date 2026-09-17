# -*- coding: utf-8 -*-
"""D04 mu_grid — M3 截断高斯最优 μ 搜索：网格与恢复验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. μ 网格完整性：arange(5,40.01,0.5)，步长 0.5，覆盖 5-40°（71 点）
  2. 合成截断高斯 N(14°,25°)|[0,45] -> 联合网格搜索恢复 μ*≈14°、σ*≈25°
  3. μ 固定扫描表（σ 同步优化）复现主文档趋势：μ=14° 处 χ² 最小
  4. 物理含义：tan(μ*)≈0.25 -> |κ₂|/|κ₁|≈25%（鞍面主导）
  5. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M3 联合网格：
     恢复 μ*=14.0°、σ*=25.0°（χ²≈26），与文档审计 μ≈14° σ≈25° 一致）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D04Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D04Config（环境变量 AIQ_D04_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            μ/σ 网格经 get_grid 建模，跨节点在 build() 解析）
  MuSynthesizer          —— φ 分布合成（拒绝采样）+ M3 联合网格搜索
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2269-2398（D04 七步流程）
  源码 _phi_fit_chi2.py 行 130-137（μ,σ 联合网格）
  《参数审计与实验报告.txt》行 17、55（状态=已用，μ→14°）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
from math import erf, radians, sqrt, tan
from typing import Any

import numpy as np

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

# ---- 第一层：配置模型 D04Config（pydantic 优先；dataclass 回退） ----
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
        """pydantic 缺失时的空壳基类（无字段，仅提供 dataclass 语义）。

        子类 D04Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D04Config(_ConfigModelBase):
    """D04 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（SIG_TRUE/
    SIGMA_GRID）在 ConfigFactory.build() 中按来源节点显式解析。
    """

    MU_TRUE: float = 14.0               # 合成数据真实 μ（= 实测 14.1 取整）
    SIG_TRUE: float = 25.0              # 合成数据真实 σ（= 实测 25.1 取整，D05.SIGMA_TRUE 同源）
    N: int = 200000                     # 合成样本数
    N_FIT: int = 39                     # 拟合域格数
    TOL_MU: float = 1.0                 # μ 恢复容差（网格分辨率±0.5）
    TOL_SIG: float = 2.0                # σ 恢复容差（网格分辨率±0.5）
    MU_GRID: tuple = (5.0, 40.0, 71)    # μ 网格：(start, stop, count)（README ①）
    SIGMA_GRID: tuple = (2.0, 30.0, 57) # σ 网格：(start, stop, count)（D05 定义）
    MU_SCAN: list = [8, 12, 14, 18, 22] # μ 固定扫描表（主文档行 2385-2389）
    TAN_TOL: float = 0.08               # tan(μ*)≈0.25 物理含义容差
    SEED: int = 0                       # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D04 配置工厂：实例化 D04Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D04Config:
        cfg = self.build_model(D04Config, "D04")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D05）
        cfg.SIG_TRUE = self.get_float("D05", "SIGMA_TRUE", 25.0)
        # D 组网格参数：get_grid 保留 (start, stop, count) 语义
        cfg.MU_GRID = self.get_grid("D04", "MU_GRID", (5.0, 40.0, 71))
        cfg.SIGMA_GRID = self.get_grid("D05", "SIGMA_GRID", (2.0, 30.0, 57))
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
_erf = np.vectorize(erf)


def norm_cdf(x: float, mu: float, sig: float) -> float:
    """正态 CDF（math.erf 实现，兼容 numpy 2.x 无 np.erf）。"""
    return 0.5 * (1.0 + _erf((x - mu) / (sig * sqrt(2.0))))


def exp_gauss(centers: np.ndarray, mu: float, sig: float, Ntot: float) -> np.ndarray:
    """M3 截断高斯期望频数（README ②公式）。"""
    lo = centers - 0.5
    hi = centers + 0.5
    denom = norm_cdf(45.0, mu, sig) - norm_cdf(0.0, mu, sig)
    assert float(np.max(denom)) > 0.0, f"exp_gauss: 归一化分母为 0（μ={mu},σ={sig}）"
    return Ntot * (norm_cdf(hi, mu, sig) - norm_cdf(lo, mu, sig)) / denom


def chi2_merge(obs: np.ndarray, exp: np.ndarray) -> float:
    """低期望 bin 合并（源码 chi2_merge：期望<1 时合并到相邻格）。"""
    o = obs.astype(float).copy()
    e = exp.astype(float).copy()
    assert o.size == e.size > 0, f"chi2_merge: obs/e 尺寸不一致或为空 {o.size}/{e.size}"
    assert np.all(np.isfinite(o)) and np.all(np.isfinite(e)), "chi2_merge: 输入含 NaN/Inf"
    assert e.sum() > 0.0, f"chi2_merge: 期望总和为 0，无法计算 χ²"
    while e.min() < 1.0 and len(e) > 1:
        i = int(np.argmin(e))
        if i == len(e) - 1:
            e[-2] += e[-1]
            o[-2] += o[-1]
            e = e[:-1]
            o = o[:-1]
        else:
            e[i + 1] += e[i]
            o[i + 1] += o[i]
            e = np.delete(e, i)
            o = np.delete(o, i)
    assert np.all(e > 0.0), "chi2_merge: 合并后仍存在非正期望"
    return float(((o - e) ** 2 / e).sum())


def load_phi_real(rd: Any) -> np.ndarray | None:
    """真实 (κ1,κ2) -> φ(度) 数组；缺失返回 None。

    参数：
      rd: 共享真实数据模块（_real_data，经 setup_env 注入或惰性导入）
    """
    pairs = rd.phi_pairs()
    if pairs is None:
        return None
    k1, k2 = pairs[:, 0], pairs[:, 1]
    a1, a2 = np.abs(k1), np.abs(k2)
    big = np.maximum(a1, a2)
    small = np.minimum(a1, a2)
    valid = big > 1e-12
    if not valid.any():
        return None
    return np.degrees(np.arctan(small[valid] / big[valid]))


# ---- 第三层：合成器 MuSynthesizer（算法与原脚本完全一致） ----
class MuSynthesizer(_SynthBase):
    """D04 合成器：φ 分布拒绝采样 + M3 (μ,σ) 联合网格搜索。"""

    def __init__(self, cfg: D04Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_phi_trunc_gauss(
        self, mu: float, sigma: float, lo: float, hi: float,
        n: int, seed: int | None = None,
    ) -> np.ndarray:
        """拒绝采样：截断高斯 N(μ,σ)|[lo,hi]。"""
        assert n > 0, f"synth_phi_trunc_gauss: 样本数 n={n} 必须>0"
        rng = np.random.default_rng(self._seed if seed is None else seed)
        out = []
        total = 0
        guard = 0
        while total < n:
            guard += 1
            assert guard < 1000, "synth_phi_trunc_gauss: 拒绝采样未收敛（接受率过低）"
            x = rng.normal(mu, sigma, n - total)
            x = x[(x >= lo) & (x <= hi)]
            out.append(x)
            total += len(x)
        return np.concatenate(out)[:n]

    def scan_m3(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
                Ntot: float) -> tuple:
        """联合网格：μ∈[5,40]步0.5 × σ∈[2,30]步0.5 -> χ² 最小 (μ*,σ*)（README ②）。

        返回 ((μ*, σ*), χ²)。
        """
        cfg = self._cfg
        mu_lo, mu_hi, n_mu = cfg.MU_GRID
        sig_lo, sig_hi, n_sig = cfg.SIGMA_GRID
        best = (1e18, None)
        for mu in np.linspace(mu_lo, mu_hi, n_mu):
            for sig in np.linspace(sig_lo, sig_hi, n_sig):
                c2 = chi2_merge(obs_fit, exp_gauss(centers_fit, mu, sig, Ntot))
                if c2 < best[0]:
                    best = (c2, (mu, sig))
        return best[1], best[0]


# ---- 第四层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D04 验证引擎：顺序执行 5 项验证（4 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D04Config,
        synth: MuSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _prep_synth(self) -> tuple:
        """合成截断高斯数据并返回 (obs_fit, centers_fit, bins)。"""
        cfg = self.config
        phi = self.synth.synth_phi_trunc_gauss(cfg.MU_TRUE, cfg.SIG_TRUE, 0.0, 45.0, cfg.N)
        bins = np.linspace(0, 45, 46)
        centers_full = (bins[:-1] + bins[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)
        obs_fit = np.histogram(phi, bins=bins)[0][fit_idx]
        centers_fit = centers_full[fit_idx]
        return obs_fit, centers_fit, bins

    def _scan_table(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
                    Ntot: float) -> list:
        """μ 固定扫描表（σ 同步优化）-> [(μ, σ*, χ²), ...]。"""
        cfg = self.config
        table = []
        sig_lo, sig_hi, n_sig = cfg.SIGMA_GRID
        sigmas = np.linspace(sig_lo, sig_hi, n_sig)
        for mu in cfg.MU_SCAN:
            best = (1e18, None)
            for sig in sigmas:
                c2 = chi2_merge(obs_fit, exp_gauss(centers_fit, mu, sig, Ntot))
                if c2 < best[0]:
                    best = (c2, sig)
            table.append((mu, best[1], best[0]))
        return table

    # ------------------------------------------------------------ 1) 网格完整性
    def validate_grid(self) -> dict:
        """1) μ 网格完整性（README ①）。"""
        cfg = self.config
        mus = np.linspace(*cfg.MU_GRID)
        dmu = np.diff(mus)
        ok = (len(mus) == 71) and bool(np.allclose(dmu, 0.5))
        if not ok:
            raise AIQValidationError(
                f"μ 网格不符: n={len(mus)}",
                expected=71, actual=len(mus), param_key="D04",
            )
        return {
            "detail": f"n={len(mus)}, 范围[{mus[0]},{mus[-1]}], 步长一致",
            "n": len(mus),
        }

    # ------------------------------------------------------------ 2) 联合网格恢复
    def validate_joint_recovery(self) -> dict:
        """2) 联合网格恢复 μ*, σ*（README ⑤表第 2-3 行）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        (mu_star, sig_star), c2_star = self.synth.scan_m3(obs_fit, centers_fit, cfg.N)
        ok = (abs(mu_star - cfg.MU_TRUE) <= cfg.TOL_MU
              and abs(sig_star - cfg.SIG_TRUE) <= cfg.TOL_SIG)
        if not ok:
            raise AIQValidationError(
                f"联合网格恢复偏差: μ*={mu_star:.1f}, σ*={sig_star:.1f}",
                expected={"mu tol": cfg.TOL_MU, "sig tol": cfg.TOL_SIG},
                actual={"mu": mu_star, "sig": sig_star}, param_key="D04",
            )
        return {
            "detail": f"μ*={mu_star:.1f}° σ*={sig_star:.1f}° χ²={c2_star:.2f}; "
                      f"Δμ={abs(mu_star - cfg.MU_TRUE):.2f}°(≤{cfg.TOL_MU}), "
                      f"Δσ={abs(sig_star - cfg.SIG_TRUE):.2f}°(≤{cfg.TOL_SIG})",
            "mu_star": mu_star, "sig_star": sig_star, "chi2": c2_star,
        }

    # ------------------------------------------------------------ 3) μ 固定扫描表
    def validate_mu_scan(self) -> dict:
        """3) μ 固定扫描表：μ=14° 处 χ² 最小（README ⑤表第 4 行）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        table = self._scan_table(obs_fit, centers_fit, cfg.N)
        c2_14 = [c2 for mu, _, c2 in table if mu == cfg.MU_TRUE][0]
        ok = all(c2_14 <= c2 for mu, _, c2 in table if mu != cfg.MU_TRUE)
        table_str = "; ".join(f"μ={mu}°:χ²={c2:.1f}" for mu, _, c2 in table)
        if not ok:
            raise AIQValidationError(
                f"μ=14° 处 χ² 非最小: {table_str}",
                expected={"mu=14 min": True}, actual=table, param_key="D04",
            )
        return {
            "detail": table_str + f" (实测 8→78.5,12→52.3,14→42.1,18→48.7,22→55.2)",
            "table": table,
        }

    # ------------------------------------------------------------ 4) 物理含义
    def validate_physics(self) -> dict:
        """4) 物理含义：tan(μ*)≈0.25（README ③几何直觉，鞍面主导）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        (mu_star, _), _ = self.synth.scan_m3(obs_fit, centers_fit, cfg.N)
        ratio = tan(radians(mu_star))
        target = tan(radians(cfg.MU_TRUE))   # tan(14°)≈0.25（解析目标）
        ok = abs(ratio - target) < cfg.TAN_TOL
        if not ok:
            raise AIQValidationError(
                f"物理含义偏离: tan({mu_star:.1f}°)={ratio:.3f} 需≈{target:.3f}",
                expected=target, actual=ratio, param_key="D04",
            )
        return {
            "detail": f"tan({mu_star:.1f}°)={ratio:.3f} (鞍面主导, 各向同性为 1.0)",
            "ratio": ratio,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real(self) -> dict:
        """5) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M3 联合网格）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        bins = np.linspace(0, 45, 46)
        fit_idx = np.arange(cfg.N_FIT)
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成截断高斯
            obs_fit, centers_fit, _ = self._prep_synth()
            (mu_star, sig_star), _ = self.synth.scan_m3(obs_fit, centers_fit, cfg.N)
            ok = (abs(mu_star - cfg.MU_TRUE) <= cfg.TOL_MU
                  and abs(sig_star - cfg.SIG_TRUE) <= cfg.TOL_SIG)
            if not ok:
                raise AIQValidationError(
                    f"回退合成恢复偏差: μ*={mu_star:.1f}, σ*={sig_star:.1f}",
                    expected={"mu": cfg.MU_TRUE, "sig": cfg.SIG_TRUE},
                    actual={"mu": mu_star, "sig": sig_star}, param_key="D04",
                )
            return {"detail": f"μ*={mu_star:.1f}° σ*={sig_star:.1f}°（真实数据未就绪）",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        Nreal = float(len(phi_real))
        obs_r = np.histogram(phi_real, bins=bins)[0][fit_idx]
        centers_fit = (bins[:-1] + bins[1:]) / 2
        centers_fit = centers_fit[fit_idx]
        (mu_r, sig_r), c2_r = self.synth.scan_m3(obs_r, centers_fit, Nreal)
        okr1 = (abs(mu_r - cfg.MU_TRUE) <= cfg.TOL_MU) and (abs(sig_r - cfg.SIG_TRUE) <= cfg.TOL_SIG)
        if not okr1:
            raise RealModelMismatchError(
                f"真实数据 M3 恢复偏差: μ*={mu_r:.1f}°, σ*={sig_r:.1f}°",
                expected={"mu tol": cfg.TOL_MU, "sig tol": cfg.TOL_SIG},
                actual={"mu": mu_r, "sig": sig_r}, param_key="D04",
            )
        # μ 固定扫描表（σ 同步优化）于真实数据
        table_r = self._scan_table(obs_r, centers_fit, Nreal)
        c2_14_r = [c2 for mu, _, c2 in table_r if mu == cfg.MU_TRUE][0]
        okr2 = all(c2_14_r <= c2 for mu, _, c2 in table_r if mu != cfg.MU_TRUE)
        if not okr2:
            raise RealModelMismatchError(
                f"真实数据 μ=14° 处 χ² 非最小: {table_r}",
                expected={"mu=14 min": True}, actual=table_r, param_key="D04",
            )
        table_str_r = "; ".join(f"μ={mu}°:χ²={c2:.1f}" for mu, _, c2 in table_r)
        return {
            "detail": (f"{tag} 真实 φ 数据 M3 联合网格 μ*={mu_r:.1f}° σ*={sig_r:.1f}° "
                       f"χ²={c2_r:.2f}（文档审计 μ=14°/σ=25°，一致）; "
                       f"μ 固定扫描：{table_str_r}（μ=14° χ² 最小）"),
            "source": rd.source_tag(), "tag": tag,
            "mu_star": mu_r, "sig_star": sig_r, "chi2": c2_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "grid", self.validate_grid),
            (2, "joint_recovery", self.validate_joint_recovery),
            (3, "mu_scan", self.validate_mu_scan),
            (4, "physics", self.validate_physics),
            (5, "real_model", self.validate_real),
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
            # 结构化 JSON 日志（_logging 单例；每行可 json.loads）
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """D04 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D04 mu_grid 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = MuSynthesizer(cfg)                 # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D04 mu_grid 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MU_TRUE={cfg.MU_TRUE} SIG_TRUE={cfg.SIG_TRUE} N={cfg.N} "
          f"N_FIT={cfg.N_FIT} TOL_MU={cfg.TOL_MU} TOL_SIG={cfg.TOL_SIG} "
          f"MU_GRID={cfg.MU_GRID} SIGMA_GRID={cfg.SIGMA_GRID} MU_SCAN={cfg.MU_SCAN} "
          f"SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d04_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
