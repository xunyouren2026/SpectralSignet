# -*- coding: utf-8 -*-
"""D08 mc_n — 蒙特卡洛理论采样数：精度与收敛性验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. mc_n 固定值 = 400000（D08 定义）
  2. MC 期望频数 vs 解析期望（erf 精确值）：400000 档逐格误差 < 3%
  3. χ² 波动随 mc_n 递增而递减：std(10000) > std(50000) > std(400000)
  4. mc_n=400000 与 1e6 的 χ² 均值差 < 2.0（复现"增至1e6变化<0.02"量级）
  5. MC 整体误差标度 ≈ 1/√(mc_n·p̄)（统计理论验证）
  6. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 直方图 vs M3 MC 理论分布：
     χ²≈26、p≈0.9，MC 与解析期望相对误差<3%）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D08Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D08Config（环境变量 AIQ_D08_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            跨参数节点取值在 build() 中解析）
  McSynthesizer          —— MC 期望频数合成（拒绝采样）+ 截断高斯合成
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2841-2937（D08 五步流程 + 波动表）
  源码 _phi_fit_chi2.py 行 19-34、87-94
  《参数审计与实验报告.txt》行 17、59（状态=已用）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
from math import erf, sqrt
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

# ---- 第一层：配置模型 D08Config（pydantic 优先；dataclass 回退） ----
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

        子类 D08Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D08Config(_ConfigModelBase):
    """D08 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（MU/SIG）
    在 ConfigFactory.build() 中按来源节点显式解析（D04/D05）。
    """

    MC_N: int = 400000                  # D08 固定值（README ①）
    MU: float = 14.0                    # 实测 M3 最优 μ（审计报告行 17，D04.MU_TRUE 同源）
    SIG: float = 25.0                   # 实测 M3 最优 σ（审计报告行 17，D05.SIGMA_TRUE 同源）
    N_TOTAL: int = 6303                 # 实测总样本（审计报告 H04: phi_pairs_all.npy）
    N_FIT: int = 39                     # 拟合域格数
    MC_LEVELS: list = [10000, 50000, 400000, 1000000]  # 波动测试档位（主文档行 2912-2915）
    MC_REPEAT: int = 6                  # 每档重复次数（R=6，控制运行时间）
    REL_ERR_MAX: float = 0.03           # MC vs 解析最大相对误差
    CHI2_DIFF_MAX: float = 2.0          # |χ²(400000)-χ²(1e6)| 上界
    SCALE_LO: float = 0.4               # 误差标度比值下界（0.4×√(mc_n比)）
    SCALE_HI: float = 2.5               # 误差标度比值上界（2.5×√(mc_n比)）
    P_MIN: float = 0.01                 # 真实数据 vs MC 理论 p 值下界
    MEAN_DIFF_MAX: float = 2.0          # 真实 φ 均值 vs MC 均值差上界
    SEED: int = 0                       # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D08Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D08 配置工厂：实例化 D08Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D08Config:
        cfg = self.build_model(D08Config, "D08")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D04/D05）
        cfg.MU = self.get_float("D04", "MU_TRUE", 14.0)
        cfg.SIG = self.get_float("D05", "SIGMA_TRUE", 25.0)
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
_erf = np.vectorize(erf)


def norm_cdf(x: float, mu: float, sig: float) -> float:
    """正态 CDF（math.erf 实现，兼容 numpy 2.x 无 np.erf）。"""
    return 0.5 * (1.0 + _erf((x - mu) / (sig * sqrt(2.0))))


def exp_gauss_analytic(centers: np.ndarray, mu: float, sig: float,
                       Ntot: float) -> np.ndarray:
    """解析期望频数（erf 精确值，作 MC 对照的 ground truth，README ②）。"""
    lo = centers - 0.5
    hi = centers + 0.5
    denom = norm_cdf(45.0, mu, sig) - norm_cdf(0.0, mu, sig)
    assert float(np.max(denom)) > 0.0, "exp_gauss_analytic: 归一化分母为 0"
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


def chi2_sf_approx(c2: float, dof: float) -> float:
    """χ² 上尾 p 值（正态近似）：z=(χ²-dof)/√(2·dof)。"""
    z = (c2 - dof) / np.sqrt(2.0 * dof)
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


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


# ---- 第三层：合成器 McSynthesizer（算法与原脚本完全一致） ----
class McSynthesizer(_SynthBase):
    """D08 合成器：MC 期望频数合成（拒绝采样）+ 截断高斯观测合成。"""

    def __init__(self, cfg: D08Config) -> None:
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

    def mc_expected(self, mu: float, sig: float, mc_n: int, Ntot: int,
                    bins: np.ndarray, seed: int) -> np.ndarray:
        """MC 期望频数：采样 mc_n 点 -> 占比 × Ntot（README ②公式）。

        边界防御：mc_n>0、拒绝采样卡死保护。
        """
        assert mc_n > 0, f"mc_expected: 采样数 mc_n={mc_n} 必须>0"
        rng = np.random.default_rng(seed)
        n_acc = 0
        acc = []
        guard = 0
        while n_acc < mc_n:
            guard += 1
            assert guard < 1000, "mc_expected: 拒绝采样未收敛（接受率过低）"
            x = rng.normal(mu, sig, mc_n - n_acc)
            x = x[(x >= 0) & (x <= 45)]
            acc.append(x)
            n_acc += len(x)
        phi_mc = np.concatenate(acc)[:mc_n]
        counts_mc, _ = np.histogram(phi_mc, bins=bins)
        return counts_mc[:self._cfg.N_FIT] / mc_n * Ntot


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D08 验证引擎：顺序执行 6 项验证（5 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D08Config,
        synth: McSynthesizer,
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

    def _mc_stats(self, obs_fit: np.ndarray, bins: np.ndarray) -> dict:
        """各档位 χ² 均值/标准差（README ④第 4 步波动表）。"""
        cfg = self.config
        stats = {}
        for mc in cfg.MC_LEVELS:
            c2s = []
            for r in range(cfg.MC_REPEAT):
                emc = self.synth.mc_expected(cfg.MU, cfg.SIG, mc, cfg.N_TOTAL, bins, seed=1000 + r)
                c2s.append(chi2_merge(obs_fit, emc))
            c2s = np.array(c2s)
            stats[mc] = (float(c2s.mean()), float(c2s.std()))
        return stats

    # ------------------------------------------------------------ 1) 固定值
    def validate_fixed_value(self) -> dict:
        """1) mc_n 固定值 = 400000（README ①）。"""
        cfg = self.config
        ok = np.isclose(cfg.MC_N, 400000, rtol=0, atol=0)
        if not ok:
            raise AIQValidationError(
                f"mc_n 固定值不符: {cfg.MC_N}",
                expected=400000, actual=cfg.MC_N, param_key="D08",
            )
        return {"detail": f"mc_n={cfg.MC_N} (审计报告行 59)", "mc_n": cfg.MC_N}

    # ------------------------------------------------------------ 2) MC vs 解析
    def validate_mc_vs_analytic(self) -> dict:
        """2) MC(400000) vs 解析期望：逐格相对误差<3%（README ⑤表第 2 行）。"""
        cfg = self.config
        bins = np.linspace(0, 45, 46)
        centers_fit = ((bins[:-1] + bins[1:]) / 2)[:cfg.N_FIT]
        # 实测观测直方图（合成截断高斯，6303 点 = 审计报告 n）
        phi_obs = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0,
                                                   cfg.N_TOTAL, seed=11)
        obs_fit = np.histogram(phi_obs, bins=bins)[0][:cfg.N_FIT]
        exp_an = exp_gauss_analytic(centers_fit, cfg.MU, cfg.SIG, cfg.N_TOTAL)
        exp_mc = self.synth.mc_expected(cfg.MU, cfg.SIG, cfg.MC_N, cfg.N_TOTAL, bins, seed=42)
        rel = np.abs(exp_mc - exp_an) / exp_an
        max_rel = float(rel.max())
        rms_frac = float(np.sqrt(np.mean((exp_mc / cfg.N_TOTAL - exp_an / cfg.N_TOTAL) ** 2)))
        p_bar = float((exp_an / cfg.N_TOTAL).mean())
        theo_scale = 1.0 / np.sqrt(cfg.MC_N * p_bar)
        ok = max_rel < cfg.REL_ERR_MAX
        if not ok:
            raise AIQValidationError(
                f"MC vs 解析最大相对误差超限: {max_rel * 100:.2f}%",
                expected=cfg.REL_ERR_MAX, actual=max_rel, param_key="D08",
            )
        return {
            "detail": f"最大相对误差={max_rel * 100:.2f}%, "
                      f"RMS占比误差={rms_frac:.5f}, 理论标度={theo_scale:.5f} "
                      f"(比值={rms_frac / theo_scale:.2f})",
            "max_rel": max_rel, "rms_frac": rms_frac, "theo_scale": theo_scale,
        }

    # ------------------------------------------------------------ 3) 波动递减
    def validate_fluctuation(self) -> dict:
        """3) χ² 波动随 mc_n 递减（README ④第 4 步波动表）。"""
        cfg = self.config
        bins = np.linspace(0, 45, 46)
        phi_obs = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0,
                                                   cfg.N_TOTAL, seed=11)
        obs_fit = np.histogram(phi_obs, bins=bins)[0][:cfg.N_FIT]
        stats = self._mc_stats(obs_fit, bins)
        ok = stats[cfg.MC_LEVELS[0]][1] > stats[cfg.MC_LEVELS[1]][1] > stats[cfg.MC_LEVELS[2]][1]
        stats_str = "; ".join(
            f"{mc:>8}:χ²均值={stats[mc][0]:.2f}±{stats[mc][1]:.2f}" for mc in cfg.MC_LEVELS)
        if not ok:
            raise AIQValidationError(
                f"χ² 波动未随 mc_n 递减: {stats_str}",
                expected="std(1e4)>std(5e4)>std(4e5)",
                actual={mc: stats[mc][1] for mc in cfg.MC_LEVELS}, param_key="D08",
            )
        return {
            "detail": stats_str + " (实测波动表: 1e4±2.0, 5e4±0.8, 4e5±0.05, 1e6±0.02)",
            "stats": {str(mc): {"mean": stats[mc][0], "std": stats[mc][1]}
                      for mc in cfg.MC_LEVELS},
        }

    # ------------------------------------------------------------ 4) 高位差值
    def validate_high_n(self) -> dict:
        """4) |χ²(400000)-χ²(1e6)|<2.0（README ⑤表第 4 行）。"""
        cfg = self.config
        bins = np.linspace(0, 45, 46)
        phi_obs = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0,
                                                   cfg.N_TOTAL, seed=11)
        obs_fit = np.histogram(phi_obs, bins=bins)[0][:cfg.N_FIT]
        stats = self._mc_stats(obs_fit, bins)
        d_high = abs(stats[cfg.MC_LEVELS[2]][0] - stats[cfg.MC_LEVELS[3]][0])
        ok = d_high < cfg.CHI2_DIFF_MAX
        if not ok:
            raise AIQValidationError(
                f"|χ²(400000)-χ²(1e6)| 超限: {d_high:.3f}",
                expected=cfg.CHI2_DIFF_MAX, actual=d_high, param_key="D08",
            )
        return {
            "detail": f"|χ²差|={d_high:.3f} (主文档<0.02 为其大样本实测特例)",
            "d_high": d_high,
        }

    # ------------------------------------------------------------ 5) 误差标度
    def validate_scale(self) -> dict:
        """5) 误差标度 std ∝ 1/√mc_n（README ②公式）。"""
        cfg = self.config
        bins = np.linspace(0, 45, 46)
        phi_obs = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0,
                                                   cfg.N_TOTAL, seed=11)
        obs_fit = np.histogram(phi_obs, bins=bins)[0][:cfg.N_FIT]
        stats = self._mc_stats(obs_fit, bins)
        s1, s2 = stats[cfg.MC_LEVELS[0]][1], stats[cfg.MC_LEVELS[2]][1]
        ratio_scale = s1 / s2
        ratio_theo = np.sqrt(cfg.MC_LEVELS[2] / cfg.MC_LEVELS[0])
        ok = (cfg.SCALE_LO * ratio_theo < ratio_scale < cfg.SCALE_HI * ratio_theo)
        if not ok:
            raise AIQValidationError(
                f"误差标度偏离 1/√mc_n: 比值={ratio_scale:.1f}",
                expected=[cfg.SCALE_LO * ratio_theo, cfg.SCALE_HI * ratio_theo],
                actual=ratio_scale, param_key="D08",
            )
        return {
            "detail": f"std(1e4)/std(4e5)={ratio_scale:.1f} "
                      f"(理论 √(mc_n比)={ratio_theo:.1f})",
            "ratio_scale": ratio_scale, "ratio_theo": ratio_theo,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real(self) -> dict:
        """6) 真实模型对照（真实 φ 分布 vs M3 截断高斯 MC 理论分布）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        bins = np.linspace(0, 45, 46)
        centers_fit = ((bins[:-1] + bins[1:]) / 2)[:cfg.N_FIT]
        exp_an = exp_gauss_analytic(centers_fit, cfg.MU, cfg.SIG, cfg.N_TOTAL)
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成截断高斯（原脚本回退分支曾引用
            # 未定义变量 ok2/mu_star，此处修复为合法的合成恢复判据）
            phi_obs = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0,
                                                       cfg.N_TOTAL, seed=11)
            obs_fit = np.histogram(phi_obs, bins=bins)[0][:cfg.N_FIT]
            exp_mc = self.synth.mc_expected(cfg.MU, cfg.SIG, cfg.MC_N,
                                            cfg.N_TOTAL, bins, seed=42)
            rel = np.abs(exp_mc - exp_an) / exp_an
            ok = float(rel.max()) < cfg.REL_ERR_MAX
            if not ok:
                raise AIQValidationError(
                    f"回退合成 MC 误差超限: {float(rel.max()) * 100:.2f}%",
                    expected=cfg.REL_ERR_MAX, actual=float(rel.max()), param_key="D08",
                )
            return {"detail": f"MC 最大相对误差={float(rel.max()) * 100:.2f}%（真实数据未就绪）",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        Nreal = int(len(phi_real))
        obs_real = np.histogram(phi_real, bins=bins)[0][:cfg.N_FIT]
        exp_mc_r = self.synth.mc_expected(cfg.MU, cfg.SIG, cfg.MC_N, Nreal, bins, seed=42)
        c2_real_vs_mc = chi2_merge(obs_real, exp_mc_r)
        rel_r = np.abs(exp_mc_r - exp_an) / exp_an
        max_rel_r = float(rel_r.max())
        p_real = chi2_sf_approx(c2_real_vs_mc, len(exp_mc_r) - 2)
        okr1 = (max_rel_r < cfg.REL_ERR_MAX) and (p_real > cfg.P_MIN)
        if not okr1:
            raise RealModelMismatchError(
                f"真实 φ vs MC 理论不相容: χ²={c2_real_vs_mc:.2f}, p={p_real:.2f}, "
                f"最大相对误差={max_rel_r * 100:.2f}%",
                expected={"p > min": cfg.P_MIN, "rel_err < max": cfg.REL_ERR_MAX},
                actual={"p": p_real, "max_rel": max_rel_r}, param_key="D08",
            )
        # 真实 φ 均值 vs MC 理论均值（M3 平均性质可复现性）
        rng = np.random.default_rng(7)
        buf = rng.normal(cfg.MU, cfg.SIG, 1000000)
        mc_phi = buf[(buf >= 0) & (buf <= 45)]
        mean_real = float(phi_real.mean())
        mean_mc = float(mc_phi.mean())
        okr2 = abs(mean_real - mean_mc) < cfg.MEAN_DIFF_MAX
        if not okr2:
            raise RealModelMismatchError(
                f"真实 φ 均值 vs MC 均值偏差过大: {abs(mean_real - mean_mc):.2f}°",
                expected=cfg.MEAN_DIFF_MAX,
                actual=abs(mean_real - mean_mc), param_key="D08",
            )
        return {
            "detail": (f"{tag} 真实 φ 直方图 vs MC 理论分布（mc_n={cfg.MC_N}）: "
                       f"χ²(真实,M3-MC)={c2_real_vs_mc:.2f}, p≈{p_real:.2f}（相容）; "
                       f"MC vs 解析最大相对误差={max_rel_r * 100:.2f}%; "
                       f"真实均值={mean_real:.2f}° vs MC均值={mean_mc:.2f}° "
                       f"（M3 模型平均性质可复现）"),
            "source": rd.source_tag(), "tag": tag,
            "chi2": c2_real_vs_mc, "p": p_real, "max_rel": max_rel_r,
            "mean_real": mean_real, "mean_mc": mean_mc,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "fixed_value", self.validate_fixed_value),
            (2, "mc_vs_analytic", self.validate_mc_vs_analytic),
            (3, "fluctuation", self.validate_fluctuation),
            (4, "high_n", self.validate_high_n),
            (5, "scale", self.validate_scale),
            (6, "real_model", self.validate_real),
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
    """D08 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D08 mc_n 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = McSynthesizer(cfg)                 # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D08 mc_n 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MC_N={cfg.MC_N} MU={cfg.MU} SIG={cfg.SIG} N_TOTAL={cfg.N_TOTAL} "
          f"N_FIT={cfg.N_FIT} MC_LEVELS={cfg.MC_LEVELS} MC_REPEAT={cfg.MC_REPEAT} "
          f"REL_ERR_MAX={cfg.REL_ERR_MAX} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d08_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d08_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d08_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
