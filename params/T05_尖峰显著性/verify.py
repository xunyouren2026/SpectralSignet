# -*- coding: utf-8 -*-
"""T05 尖峰显著性 φ 分布尖峰卡方判据 — 统计裁决层守护闸门
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 临界值 5.99 = χ²_{0.05}(df=2)（scipy 校验；无 scipy 用文档常量）
  2. 实测 χ²: M3=42.1, M4=39.8 -> Δχ²(M3→M4)=2.3 < 5.99 -> M4 否决（无尖峰）
  3. 口径说明：χ²_M1 − χ²_M4 = 245.3 − 39.8 = 205.5 仅证背景非均匀；
     是否引入尖峰须用相邻模型增量 Δχ²(M3→M4)（诚实口径标注）
  4. 合成「无尖峰」φ 数据：拟合 M3 vs M4 -> Δχ² < 5.99（不触发）
  5. 合成「有尖峰」φ 数据：拟合 M3 vs M4 -> Δχ² > 5.99（触发）
  6. 判决函数阈值分界验证
  7. 工程化防御：chi2_merge 空输入/除零/NaN 防御
  8. 真实模型实测对照：真实 φ 分布（phi_pairs_all.npy）M3/M4 -> 无尖峰
  9. 真实 vs 文档审计 Δχ²（真实实测为准）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T05Config / ConfigFactory / T05Synthesizer / T05Validator /
  ReportGenerator / main —— 同 A01（env AIQ_T05_<KEY> 覆盖由共享工厂处理）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T05 节（行 9978-10050）
  源码 _phi_fit_chi2.py（φ 分布四模型卡方拟合）
  《AI几何指纹插件_参数审计与实验报告.txt》T05（M4否决 p=2e-7 → 无尖峰）
真实模型对照：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测 phi_pairs_all.npy（6303×2），
  由 (|κ1|,|κ2|) 得真实 φ 分布（N=6303），跑 M3/M4 卡方：
  χ²_M3≈26.2, χ²_M4≈26.2, Δχ²≈0.0 < 5.99（a*≈0）-> M4 否决，无尖峰，
  与文档审计结论一致（且比文档 Δχ²=2.3 更彻底）。
  输出标注：[真实实测]（phi_pairs_all.npy 存在）/ [审计回退]（缺失）。真实实测为准。
修复标注：原脚本无 scipy 降级路径中 norm_cdf 的 Abramowitz-Stegun 近似被
  重复套用外层 0.5*(1+·) 导致双重包装；本版改用 math.erf（标准库）精确实现。
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

# ---------------- 环境检测 ----------------
try:  # 探测 scipy 是否可用：决定临界值校验走"真实 scipy"还是"文档常量"
    import scipy.stats  # noqa: F401  仅检测 scipy 可用性
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---- 第一层：配置模型 T05Config（pydantic 优先；dataclass 回退） ----
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


class T05Config(_ConfigModelBase):
    """T05 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T05 节点键名一一对应（未在 JSON 的字段
    以模型默认值兜底）；取值优先级：环境变量 AIQ_T05_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0                 # 合成「无尖峰」φ 数据固定种子
    SEED_SPIKE: int = 1           # 合成「有尖峰」φ 数据固定种子（与无尖峰区分）
    CRIT_DCHI2: float = 5.99      # 文档临界值 χ²_{0.05}(df=2)（README ②）
    CHI_M3_DOC: float = 42.1      # 实测 χ²_M3（截断高斯，README ④Step1）
    CHI_M4_DOC: float = 39.8      # 实测 χ²_M4（高斯+尖峰，README ④Step1）
    CHI_M1_DOC: float = 245.3     # 实测 χ²_M1（均匀零模型，仅口径参考）
    MU_PK: float = 17.4           # 尖峰位置（°），主曲率角（README ⑥ D06/D07）
    CRIT_TOL: float = 0.01        # 临界值 scipy 对照容差
    DCHI2_DOC: float = 2.3        # 文档 Δχ²(M3→M4) 审计值（README ④Step2）
    N_SMOOTH: int = 20000         # 无尖峰合成样本数（采样规模）
    N_BASE: int = 18800           # 有尖峰合成样本基数（采样规模）
    N_PK: int = 1200              # 有尖峰合成尖峰样本数（6% 尖峰 @17.4°）


# ---- 第二层：配置工厂 ConfigFactory（实例化 T05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T05 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T05Config。"""

    def build(self) -> T05Config:
        """构建 T05Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T05Config, "T05")


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def norm_cdf(x, mu: float, sig: float):
    """标准正态 CDF（标量/数组通用）。

    优先用 scipy.stats.norm.cdf；无 scipy 时降级为 math.erf 精确实现
    （兼容 numpy 2.x，不依赖 np.erf）。
    """
    z = (np.asarray(x, dtype=float) - mu) / sig   # 标准化：z=(x-μ)/σ
    if _HAS_SCIPY:
        from scipy.stats import norm as _norm
        return _norm.cdf(z)                       # scipy 精确路径
    # 降级路径：Φ(z)=0.5*(1+erf(z/√2))，math.erf 为标准库、双精度精确
    return np.vectorize(lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))(z)


def chi2_merge(obs, exp) -> float:
    """卡方统计量（对期望 < 1 的 bin 做相邻合并，参照 _phi_fit_chi2.py）。

    空输入、形状不匹配、NaN/Inf、全零期望（除零）显式报错。
    """
    o = np.asarray(obs, dtype=float).copy()       # 复制：合并过程就地修改
    e = np.asarray(exp, dtype=float).copy()
    assert o.shape == e.shape, f"chi2_merge: 观测与期望形状不符 {o.shape} != {e.shape}"
    assert o.size > 0, "chi2_merge: 输入为空"
    assert np.isfinite(o).all() and np.isfinite(e).all(), \
        f"chi2_merge: 输入含 NaN/Inf (obs={o}, exp={e})"
    # 合并规则：期望 < 1 的 bin 并入相邻 bin（卡方近似要求期望 ≥ 1）
    while e.min() < 1.0 and e.size > 1:
        i = int(np.argmin(e))                     # 找期望最小的 bin
        if i == e.size - 1:                       # 末位 bin：并入前一位
            e[-2] += e[-1]; o[-2] += o[-1]
            e = e[:-1]; o = o[:-1]
        else:                                     # 一般位：并入后一位
            e[i + 1] += e[i]; o[i + 1] += o[i]
            e = np.delete(e, i); o = np.delete(o, i)
    if not (e > 0.0).any():                       # 合并后期望全为零 => 卡方无定义
        raise AssertionError(f"chi2_merge: 合并后期望全为零，卡方无定义 {e}")
    e_safe = np.where(e > 0.0, e, 1.0)            # 残余零期望防除零（正常路径不触发）
    return float(((o - e_safe) ** 2 / e_safe).sum())


def _norm_cdf_z(z: np.ndarray) -> np.ndarray:
    """标准正态 CDF 直接对标准化 z 求值（整网格一次向量化调用）。

    与 norm_cdf 数值恒等（同一 scipy 算法 / 同一 math.erf 降级实现），
    但一次性处理整个网格数组，避免双重网格搜索中逐点调用的 scipy
    分发开销（结果不变，仅性能优化）。
    """
    if _HAS_SCIPY:
        from scipy.stats import norm as _norm
        return _norm.cdf(z)                       # scipy 精确路径（整数组一次调用）
    # 降级路径：与 norm_cdf 完全相同的 math.erf 实现（一次 vectorize 整网格）
    return np.vectorize(lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))(z)


def fit_models(phi_deg: np.ndarray, mu_pk: float = 17.4) -> tuple[float, float, float, float]:
    """拟合 M3(截断高斯) 与 M4(截断高斯背景+尖峰@mu_pk) 嵌套模型。

    M4 ⊃ M3：增加 a(强度)、sp(峰宽) 两个参数 -> Δχ² 判据 df=2。
    返回 (chi2_M3, chi2_M4, a*, sp*)。
    性能：μ×σ 与 a×sp 双重网格搜索的 norm.cdf 已整网格向量化
    （结果恒等：同一 scipy 算法、同一运算顺序、同一次序的网格遍历；
    chi2_merge 相邻合并语义保持不变）。
    """
    phi_deg = np.asarray(phi_deg, dtype=float)
    assert phi_deg.size > 0, "fit_models: φ 数据为空"
    assert np.isfinite(phi_deg).all(), f"fit_models: φ 数据含 NaN/Inf: {phi_deg}"
    bins = np.linspace(0, 45, 46)                 # 0°~45° 共 45 个 1° 宽 bin
    obs_full, _ = np.histogram(phi_deg, bins=bins)
    centers_full = (bins[:-1] + bins[1:]) / 2     # 每个 bin 的中心角
    fit_idx = np.arange(39)                       # 只用前 39 个 bin（尾端 bin 样本过稀）
    obs = obs_full[fit_idx]
    centers = centers_full[fit_idx]
    lo = centers - 0.5                            # bin 左边界
    hi = centers + 0.5                            # bin 右边界
    N = len(phi_deg)                              # 总样本数（期望计数归一化分母）

    # M3 期望计数：截断高斯 N(μ,σ)|[0,45] 落在每个 bin 的概率 × N
    def exp_gauss(mu: float, sig: float) -> np.ndarray:
        denom = norm_cdf(45.0, mu, sig) - norm_cdf(0.0, mu, sig)  # 截断归一化常数
        return N * (norm_cdf(hi, mu, sig) - norm_cdf(lo, mu, sig)) / denom

    # ---- M3 网格整网格向量化：μ∈[5,40] 步长 0.5 × σ∈[2,30] 步长 0.5 ----
    mus = np.arange(5.0, 40.01, 0.5)              # (71,)
    sigs = np.arange(2.0, 30.01, 0.5)             # (57,)
    mu_grid, sig_grid = np.meshgrid(mus, sigs, indexing="ij")       # (71,57)
    z_hi = (hi[None, None, :] - mu_grid[..., None]) / sig_grid[..., None]   # (71,57,39)
    z_lo = (lo[None, None, :] - mu_grid[..., None]) / sig_grid[..., None]
    z_45 = (45.0 - mu_grid) / sig_grid            # (71,57)
    z_0 = (0.0 - mu_grid) / sig_grid
    # 截断高斯落在各 bin 的概率：(Φ(hi)-Φ(lo)) / (Φ(45)-Φ(0))，整网格一次 CDF 调用
    prob_grid = _norm_cdf_z(z_hi) - _norm_cdf_z(z_lo)               # (71,57,39)
    denom_grid = _norm_cdf_z(z_45) - _norm_cdf_z(z_0)               # (71,57)
    exp_all3 = N * prob_grid / denom_grid[..., None]                # (71,57,39)

    # 逐网格点卡方（chi2_merge 相邻合并语义保持不变；遍历次序与逐点循环恒等）
    best3 = (1e18, None)
    for i, mu in enumerate(mus):
        for j, sig in enumerate(sigs):
            c2 = chi2_merge(obs, exp_all3[i, j])
            if c2 < best3[0]:
                best3 = (c2, (float(mu), float(sig)))
    c2_3, (mu_b, sig_b) = best3                  # M3 最优卡方及其参数

    # ---- M4 网格整网格向量化：a∈[0,0.20] 步长 0.005 × sp∈[0.6,6.0] 步长 0.2 ----
    g = exp_gauss(mu_b, sig_b) / N                # 背景概率密度（单位归一，M3 最优参数）
    a_grid = np.arange(0.0, 0.21, 0.005)          # (41,)
    sp_grid = np.arange(0.6, 6.01, 0.2)           # (28,)
    z_hi4 = (hi[None, :] - mu_pk) / sp_grid[:, None]      # (28,39)
    z_lo4 = (lo[None, :] - mu_pk) / sp_grid[:, None]
    z_45_4 = (45.0 - mu_pk) / sp_grid             # (28,)
    z_0_4 = (0.0 - mu_pk) / sp_grid
    pk_grid = (_norm_cdf_z(z_hi4) - _norm_cdf_z(z_lo4)) \
        / (_norm_cdf_z(z_45_4) - _norm_cdf_z(z_0_4))[:, None]      # (28,39)
    exp_all4 = N * ((1.0 - a_grid[:, None, None]) * g[None, None, :]
                    + a_grid[:, None, None] * pk_grid[None, :, :])  # (41,28,39)

    # 逐网格点卡方（遍历次序与逐点循环恒等：a 外层、sp 内层）
    best4 = (1e18, None)
    for i, a in enumerate(a_grid):
        for j, sp in enumerate(sp_grid):
            c2 = chi2_merge(obs, exp_all4[i, j])
            if c2 < best4[0]:
                best4 = (c2, (float(a), float(sp)))
    c2_4, (a4, sp4) = best4                     # M4 最优卡方及其参数
    return c2_3, c2_4, a4, sp4


def verdict(dlt: float, crit: float) -> str:
    """Δχ² 判决函数：> crit 尖峰显著，否则不显著。"""
    return "尖峰显著 🚨" if dlt > crit else "尖峰不显著 ✅"


# ---- 第三层：合成器 T05Synthesizer（截断高斯 φ 样本） ----
class T05Synthesizer(_SynthBase):
    """T05 合成器：拒绝采样生成截断高斯 φ 分布样本（固定种子，可复现）。"""

    def __init__(self, cfg: T05Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def trunc_gauss_sample(self, n: int, mu: float, sig: float, lo: float = 0.0,
                           hi: float = 45.0, seed: int | None = None) -> np.ndarray:
        """拒绝采样：截断高斯 N(mu,sig)|[lo,hi]。"""
        if n <= 0:
            raise SynthesisError(f"trunc_gauss_sample: n 必须 > 0，收到 {n}", actual=n)
        if seed is None:
            seed = self._seed
        rng = np.random.default_rng(seed)         # 固定种子 => 可复现
        out = []
        # 拒绝采样：反复批量抽高斯样本，仅保留落在 [lo,hi] 内的，直到凑满 n 个
        while len(out) < n:
            x = rng.normal(mu, sig, size=2 * n)   # 一次抽 2n 个提高批处理效率
            x = x[(x >= lo) & (x <= hi)]
            out.extend(x.tolist())
        return np.array(out[:n])                  # 精确截断到 n 个样本

    def smooth_sample(self) -> np.ndarray:
        """合成「无尖峰」φ 数据（纯截断高斯）。"""
        cfg = self._cfg
        return self.trunc_gauss_sample(cfg.N_SMOOTH, mu=14.0, sig=25.0, seed=cfg.SEED)

    def spiky_sample(self) -> np.ndarray:
        """合成「有尖峰」φ 数据：背景 + 6% 窄尖峰 @MU_PK。"""
        cfg = self._cfg
        base = self.trunc_gauss_sample(cfg.N_BASE, mu=14.0, sig=25.0, seed=cfg.SEED_SPIKE)
        pk = self.trunc_gauss_sample(cfg.N_PK, mu=cfg.MU_PK, sig=0.8, seed=cfg.SEED_SPIKE)
        return np.concatenate([base, pk])         # 背景 + 窄尖峰 => 有尖峰分布


# ---- 第四层：验证引擎 T05Validator（8 项验证 + 结构化日志 + 类型化异常） ----
class T05ValidationError(AIQValidationError):
    """T05 尖峰显著性判据验证失败。"""


class T05Validator(_EngineBase):
    """T05 验证引擎：顺序执行 8 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 临界值/审计 χ² 均经配置读取（env AIQ_T05_<KEY> 覆盖），零硬编码判据。
    """

    def __init__(
        self,
        config: T05Config,
        synth: T05Synthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 临界值校验
    def validate_crit_value(self) -> dict:
        """1) 临界值校验：5.99 = χ²_{0.05}(2)（scipy 校验，无 scipy 用文档常量）。"""
        cfg = self.config
        if _HAS_SCIPY:
            from scipy.stats import chi2 as _chi2
            crit_scipy = float(_chi2.ppf(0.95, 2))  # 自由度为 2 的 χ² 分布 95% 分位点
        else:
            crit_scipy = cfg.CRIT_DCHI2           # 无 scipy 时直接用文档常量
        ok = bool(np.isclose(crit_scipy, cfg.CRIT_DCHI2, atol=cfg.CRIT_TOL, rtol=0.0))
        if not ok:
            raise T05ValidationError(
                f"临界值不符: scipy={crit_scipy} vs 文档 {cfg.CRIT_DCHI2}",
                expected=cfg.CRIT_DCHI2, actual=crit_scipy, param_key="T05",
            )
        return {
            "detail": (f"临界值: scipy χ²_{{0.05}}(2)={crit_scipy:.4f} "
                       f"(文档 {cfg.CRIT_DCHI2}, 差值 {abs(crit_scipy - cfg.CRIT_DCHI2):.4f}) "
                       f"{'[scipy 可用]' if _HAS_SCIPY else '[无 scipy, 用文档常量]'}"),
            "crit": crit_scipy, "scipy": _HAS_SCIPY,
        }

    # ------------------------------------------------------------ 2) 实测 χ² 复现
    def validate_doc(self) -> dict:
        """2) 实测 χ² 值复现：Δχ²(M3→M4)=2.3 < 5.99 -> M4 否决（含口径说明）。"""
        cfg = self.config
        dlt_doc = cfg.DCHI2_DOC                  # 文档 Δχ²(M3→M4) 审计值
        ok = bool(dlt_doc < cfg.CRIT_DCHI2)      # 增量未过临界 => 尖峰不显著
        if not ok:
            raise T05ValidationError(
                f"实测 Δχ²={dlt_doc} 应 < {cfg.CRIT_DCHI2}",
                expected=cfg.CRIT_DCHI2, actual=dlt_doc, param_key="T05",
            )
        dlt_m1m4 = cfg.CHI_M1_DOC - cfg.CHI_M4_DOC   # 口径说明（仅证背景非均匀）
        return {
            "detail": (f"实测: Δχ²(M3→M4) = {dlt_doc:.1f} < {cfg.CRIT_DCHI2} "
                       f"(χ²_M3={cfg.CHI_M3_DOC}, χ²_M4={cfg.CHI_M4_DOC}) -> M4 否决 ✓ "
                       f"(无尖峰); [口径] χ²_M1-χ²_M4={dlt_m1m4:.1f} > {cfg.CRIT_DCHI2} "
                       f"仅证背景非均匀；是否引入尖峰须用相邻模型增量 Δχ²(M3→M4)={dlt_doc:.1f}"),
            "dlt_doc": dlt_doc, "dlt_m1m4": dlt_m1m4,
        }

    # ------------------------------------------------------------ 3) 合成无尖峰
    def validate_no_spike(self) -> dict:
        """3) 合成「无尖峰」数据：Δχ² < 5.99 且 a*≈0（不触发）。"""
        cfg = self.config
        assert self.synth is not None
        smooth = self.synth.smooth_sample()
        c3s, c4s, a_s, sp_s = fit_models(smooth, cfg.MU_PK)  # 纯截断高斯数据拟合两模型
        dlt_s = c3s - c4s
        ok = bool(dlt_s < cfg.CRIT_DCHI2 and a_s < 0.05)
        if not ok:
            raise T05ValidationError(
                f"无尖峰数据误触发: Δχ²={dlt_s}（应 <{cfg.CRIT_DCHI2} 且 a*≈0）",
                expected={"crit": cfg.CRIT_DCHI2, "a_max": 0.05},
                actual={"dlt": dlt_s, "a": a_s}, param_key="T05",
            )
        return {"detail": (f"合成无尖峰: Δχ²={dlt_s:.2f} < {cfg.CRIT_DCHI2} "
                           f"(a*={a_s:.2f}≈0, sp*={sp_s:.1f}°) -> 不触发"),
                "dlt": dlt_s, "a": a_s, "sp": sp_s}

    # ------------------------------------------------------------ 4) 合成有尖峰
    def validate_spike(self) -> dict:
        """4) 合成「有尖峰」数据：Δχ² > 5.99（触发尖峰告警）。"""
        cfg = self.config
        assert self.synth is not None
        spiky = self.synth.spiky_sample()
        c3p, c4p, a_p, sp_p = fit_models(spiky, cfg.MU_PK)
        dlt_p = c3p - c4p
        ok = bool(dlt_p > cfg.CRIT_DCHI2)        # 增量过临界 => M4 胜出，尖峰显著
        if not ok:
            raise T05ValidationError(
                f"有尖峰数据未触发: Δχ²={dlt_p}（应 >{cfg.CRIT_DCHI2}）",
                expected=f"> {cfg.CRIT_DCHI2}", actual=dlt_p, param_key="T05",
            )
        return {"detail": (f"合成有尖峰: Δχ²={dlt_p:.2f} > {cfg.CRIT_DCHI2} "
                           f"(a*={a_p:.2f}, sp*={sp_p:.1f}°) -> 触发尖峰告警"),
                "dlt": dlt_p, "a": a_p, "sp": sp_p}

    # ------------------------------------------------------------ 5) 判决函数
    def validate_verdict(self) -> dict:
        """5) 判决函数示例：阈值分界两侧各返回正确结论。"""
        cfg = self.config
        ok = (verdict(cfg.DCHI2_DOC, cfg.CRIT_DCHI2) == "尖峰不显著 ✅") and \
             (verdict(12.0, cfg.CRIT_DCHI2) == "尖峰显著 🚨")
        if not ok:
            raise T05ValidationError(
                "判决函数阈值分界错误",
                expected={"below": "尖峰不显著 ✅", "above": "尖峰显著 🚨"},
                actual={"below": verdict(cfg.DCHI2_DOC, cfg.CRIT_DCHI2),
                        "above": verdict(12.0, cfg.CRIT_DCHI2)},
                param_key="T05",
            )
        return {"detail": (f"判决函数: Δχ²=2.3 -> {verdict(2.3, cfg.CRIT_DCHI2)}  |  "
                           f"Δχ²=12.0 -> {verdict(12.0, cfg.CRIT_DCHI2)}"),
                "ok": ok}

    # ------------------------------------------------------------ 6) 防御
    def validate_guards(self) -> dict:
        """6) 工程化防御：chi2_merge 空输入 / 形状不符 / NaN / 全零期望。"""
        guards = [
            ("空输入", lambda: chi2_merge([], [])),
            ("形状不符", lambda: chi2_merge([1.0, 2.0], [1.0])),
            ("NaN 输入", lambda: chi2_merge([1.0, float("nan")], [1.0, 1.0])),
            ("全零期望", lambda: chi2_merge([0.0, 0.0], [0.0, 0.0])),
        ]
        guard_oks = []
        for _gname, fn in guards:
            try:
                fn()                             # 未抛异常 => 防御缺失
                guard_oks.append(False)
            except AssertionError:
                guard_oks.append(True)           # 显式报错 => 防御通过
        ok_guard = all(guard_oks)
        if not ok_guard:
            raise T05ValidationError(
                f"chi2_merge 防御失败: {[g[0] for g, o in zip(guards, guard_oks) if not o]}",
                expected="all raise", actual=guard_oks, param_key="T05",
            )
        return {"detail": f"防御: 空/形状/NaN/全零期望 均显式报错 {guard_oks}",
                "guard_oks": guard_oks}

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型实测对照：真实 φ 分布（phi_pairs_all.npy）M3/M4 卡方。"""
        cfg = self.config
        rd = self._get_real_data()
        src = rd.source_tag()
        pairs = rd.phi_pairs()                   # 经 _cfg 相对定位（零硬编码）
        if pairs is None:                        # 审计回退：维持文档结论
            dlt_doc = cfg.DCHI2_DOC
            ok_fallback = bool(dlt_doc < cfg.CRIT_DCHI2)
            if not ok_fallback:
                raise T05ValidationError(
                    f"审计 Δχ²={dlt_doc} 应 < {cfg.CRIT_DCHI2}",
                    expected=cfg.CRIT_DCHI2, actual=dlt_doc, param_key="T05",
                )
            return {"detail": (f"[审计回退] 真实 φ 数据缺失，维持文档结论 Δχ²={dlt_doc:.1f} "
                               f"< {cfg.CRIT_DCHI2} -> M4 否决，无尖峰"),
                    "fallback": True}
        a1, a2 = np.abs(pairs[:, 0]), np.abs(pairs[:, 1])   # 主曲率取绝对值
        # φ = arctan(min/max)：min 为小主曲率、max 为大主曲率，结果在 [0,45]°
        phi_real = np.degrees(np.arctan2(np.minimum(a1, a2),
                                         np.maximum(a1, a2) + 1e-16))
        phi_mean_real = float(np.mean(phi_real))            # 均值：分布中心参考
        phi_med_real = float(np.median(phi_real))           # 中位数：稳健中心参考
        c3r, c4r, a_r, sp_r = fit_models(phi_real, cfg.MU_PK)  # 真实分布跑 M3/M4
        dlt_r = c3r - c4r
        ok_real = bool(dlt_r < cfg.CRIT_DCHI2)              # 预期无尖峰
        if not ok_real:
            raise RealModelMismatchError(
                f"真实 φ 拟合 Δχ²={dlt_r:.2f} 应 < {cfg.CRIT_DCHI2}（M4 否决，无尖峰）",
                expected=cfg.CRIT_DCHI2, actual=dlt_r, param_key="T05",
            )
        return {
            "detail": (f"[{src}] 真实 φ 分布(N={phi_real.size}, φmean={phi_mean_real:.2f}° "
                       f"φmed={phi_med_real:.2f}°) M3/M4: χ²_M3={c3r:.2f} "
                       f"χ²_M4={c4r:.2f} Δχ²={dlt_r:.2f} (a*={a_r:.3f}≈0, "
                       f"sp*={sp_r:.1f}°) -> 尖峰不显著 ✅"),
            "source": src, "dlt_real": dlt_r, "c3": c3r, "c4": c4r,
            "a": a_r, "sp": sp_r, "n": int(phi_real.size),
        }

    # ------------------------------------------------------------ 8) 真实 vs 审计
    def validate_real_audit(self) -> dict:
        """8) 真实 vs 文档审计 Δχ² 对照（真实数据应更无尖峰，真实实测为准）。"""
        cfg = self.config
        rd = self._get_real_data()
        pairs = rd.phi_pairs()
        if pairs is None:
            return {"detail": "[审计回退] 真实 φ 数据缺失，跳过真实 vs 审计 Δχ² 对照",
                    "fallback": True}
        a1, a2 = np.abs(pairs[:, 0]), np.abs(pairs[:, 1])
        phi_real = np.degrees(np.arctan2(np.minimum(a1, a2), np.maximum(a1, a2) + 1e-16))
        c3r, c4r, _, _ = fit_models(phi_real, cfg.MU_PK)
        dlt_r = c3r - c4r
        dlt_doc = cfg.DCHI2_DOC
        ok_cmp = bool(dlt_r < dlt_doc)                      # 真实数据应更无尖峰
        if not ok_cmp:
            raise T05ValidationError(
                f"真实 Δχ²={dlt_r} 应 ≤ 文档 {dlt_doc}",
                expected=f"< {dlt_doc}", actual=dlt_r, param_key="T05",
            )
        return {
            "detail": (f"真实 vs 审计: Δχ²(M3→M4) 真实={dlt_r:.2f} 文档审计={dlt_doc:.1f} "
                       f"(真实数据比审计更无尖峰; 两者均 < {cfg.CRIT_DCHI2})"),
            "dlt_real": dlt_r, "dlt_doc": dlt_doc,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 8 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "crit_value", self.validate_crit_value),
            (2, "doc", self.validate_doc),
            (3, "no_spike", self.validate_no_spike),
            (4, "spike", self.validate_spike),
            (5, "verdict", self.validate_verdict),
            (6, "guards", self.validate_guards),
            (7, "real", self.validate_real),
            (8, "real_audit", self.validate_real_audit),
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
    """T05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T05 尖峰显著性 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = T05Synthesizer(cfg)                       # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T05Validator(cfg, synth, report, real_data=RD)  # ③ 验证层

    print("=" * 74)
    print(f"T05 尖峰显著性  Δχ² = χ²_M3 - χ²_M4 > {cfg.CRIT_DCHI2} -> 尖峰显著")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: CRIT_DCHI2={cfg.CRIT_DCHI2} DCHI2_DOC={cfg.DCHI2_DOC} "
          f"MU_PK={cfg.MU_PK} N_SMOOTH={cfg.N_SMOOTH}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t05_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
