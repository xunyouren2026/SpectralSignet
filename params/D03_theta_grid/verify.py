# -*- coding: utf-8 -*-
"""D03 theta_grid — M2 截断均匀最优 θ 搜索：网格与搜索逻辑验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. θ 网格完整性：arange(25,45.01,0.2)，步长 0.2，覆盖 25-45°（101 点）
  2. 合成截断高斯数据 -> M2 最优 θ* 偏向高值（≥40°，M2 退化为 M1）
     -> χ²(M2) 较高，M3 截断高斯 χ² 更低 -> 复现"M2 否决"
  3. 合成截断均匀数据 U[0,30°] -> M2 最优 θ* ≈ 30°（正控制：搜索逻辑正确）
  4. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M2 网格搜索：
     θ*≈41.8°、χ²(M2)≈108 远超 χ²(M3)≈26 -> 真实数据确认"M2 否决"）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D03Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D03Config（环境变量 AIQ_D03_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            θ/M3 网格经 get_grid 建模，跨节点在 build() 解析）
  ThetaSynthesizer       —— φ 分布合成（高斯/均匀）+ M2 θ 网格搜索 + M3 对照拟合
  ValidatorEngine        —— 4 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2158-2268（D03 六步流程）
  源码 _phi_fit_chi2.py 行 122-128（exp_trunc, θ 扫描）
  《参数审计与实验报告.txt》行 54（状态=已用，M2 否决）
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

# ---- 第一层：配置模型 D03Config（pydantic 优先；dataclass 回退） ----
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

        子类 D03Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D03Config(_ConfigModelBase):
    """D03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（MU/SIG/
    MU_GRID/SIGMA_GRID）在 ConfigFactory.build() 中按来源节点显式解析。
    """

    N: int = 200000                     # 合成样本数
    N_FIT: int = 39                     # 拟合域格数（源码 np.arange(39)）
    MU: float = 14.0                    # 实测 M3 最优 μ（D04.MU_TRUE 同源）
    SIG: float = 25.0                   # 实测 M3 最优 σ（D05.SIGMA_TRUE 同源）
    THETA_GRID: tuple = (25.0, 45.0, 101)   # θ 网格：(start, stop, count)（主文档行 2158）
    MU_GRID: tuple = (5.0, 40.0, 71)    # M3 联合网格 μ（D04 定义）
    SIGMA_GRID: tuple = (2.0, 30.0, 57) # M3 联合网格 σ（D05 定义）
    THETA_STAR_MIN: float = 40.0        # 高斯数据 θ* 下界（M2 推向边界判据）
    RECOVER_TOL: float = 1.0            # 正控制 U[0,30°] θ* 恢复容差
    SEED: int = 0                       # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D03 配置工厂：实例化 D03Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D03Config:
        cfg = self.build_model(D03Config, "D03")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D04/D05）
        cfg.MU = self.get_float("D04", "MU_TRUE", 14.0)
        cfg.SIG = self.get_float("D05", "SIGMA_TRUE", 25.0)
        # D 组网格参数：get_grid 保留 (start, stop, count) 语义
        cfg.THETA_GRID = self.get_grid("D03", "THETA_GRID", (25.0, 45.0, 101))
        cfg.MU_GRID = self.get_grid("D04", "MU_GRID", (5.0, 40.0, 71))
        cfg.SIGMA_GRID = self.get_grid("D05", "SIGMA_GRID", (2.0, 30.0, 57))
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
_erf = np.vectorize(erf)


def norm_cdf(x: float, mu: float, sig: float) -> float:
    """正态 CDF（math.erf 实现，兼容 numpy 2.x 无 np.erf）。"""
    return 0.5 * (1.0 + _erf((x - mu) / (sig * sqrt(2.0))))


def exp_trunc(centers: np.ndarray, theta: float, Ntot: float) -> np.ndarray:
    """M2 截断均匀期望：E=N·clip(bin宽,[0,θ])/θ（源码 exp_trunc，README ②）。"""
    lo = centers - 0.5
    hi = centers + 0.5
    c = np.clip(hi, 0, theta) - np.clip(lo, 0, theta)
    return Ntot * c / theta


def exp_gauss(centers: np.ndarray, mu: float, sig: float, Ntot: float) -> np.ndarray:
    """M3 截断高斯期望频数（用于对照）。"""
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


# ---- 第三层：合成器 ThetaSynthesizer（算法与原脚本完全一致） ----
class ThetaSynthesizer(_SynthBase):
    """D03 合成器：φ 分布合成（高斯/均匀）+ M2 θ 网格搜索 + M3 对照拟合。

    - synth_phi_trunc_gauss / synth_phi_uniform：分布合成；
    - scan_m2：θ∈[25,45] 步 0.2 网格搜索（经 get_grid (start,stop,count) 展开）；
    - fit_m3：M3 截断高斯 (μ,σ) 网格（对照用）。
    """

    def __init__(self, cfg: D03Config) -> None:
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

    def synth_phi_uniform(self, lo: float, hi: float, n: int, seed: int = 3) -> np.ndarray:
        """合成截断均匀 U[lo,hi]。"""
        assert n > 0, f"synth_phi_uniform: 样本数 n={n} 必须>0"
        rng = np.random.default_rng(seed)
        return rng.uniform(lo, hi, n)

    def scan_m2(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
                Ntot: float) -> tuple:
        """D03：θ∈[25,45] 步 0.2 网格搜索，取 χ² 最小（README ②公式）。

        返回 (θ*, χ²_M2, thetas)。
        """
        cfg = self._cfg
        thetas = np.linspace(*cfg.THETA_GRID)   # (start, stop, count) 展开
        best = (1e18, None)
        for th in thetas:
            c2 = chi2_merge(obs_fit, exp_trunc(centers_fit, th, Ntot))
            if c2 < best[0]:
                best = (c2, th)
        return best[1], best[0], thetas

    def fit_m3(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
               Ntot: float) -> tuple:
        """M3 截断高斯 (μ,σ) 网格（用于对照，μ∈[5,40], σ∈[2,30]）。"""
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


# ---- 第四层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D03 验证引擎：顺序执行 4 项验证（3 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D03Config,
        synth: ThetaSynthesizer,
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

    def _bins_centers(self) -> tuple:
        """45 格直方图（np.linspace(0,45,46)，源码口径）+ 格心数组。"""
        bins = np.linspace(0, 45, 46)
        centers_full = (bins[:-1] + bins[1:]) / 2
        return bins, centers_full

    # ------------------------------------------------------------ 1) 网格完整性
    def validate_grid(self) -> dict:
        """1) θ 网格完整性（README ⑤表，源码口径不含 45.2 越界点）。"""
        cfg = self.config
        thetas = np.linspace(*cfg.THETA_GRID)
        dth = np.diff(thetas)
        ok = (len(thetas) == 101) and bool(np.allclose(dth, 0.2))
        if not ok:
            raise AIQValidationError(
                f"θ 网格不符: n={len(thetas)}",
                expected=101, actual=len(thetas), param_key="D03",
            )
        return {
            "detail": f"n={len(thetas)}, 范围[{thetas[0]},{thetas[-1]}], 步长一致",
            "n": len(thetas),
        }

    # ------------------------------------------------------------ 2) 高斯数据 M2 vs M3
    def validate_gauss_m2_vs_m3(self) -> dict:
        """2) 截断高斯数据：M3 更优 -> M2 否决（README ④第 5 步）。"""
        cfg = self.config
        bins, centers_full = self._bins_centers()
        fit_idx = np.arange(cfg.N_FIT)
        centers_fit = centers_full[fit_idx]
        phi_g = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
        obs_g = np.histogram(phi_g, bins=bins)[0][fit_idx]
        th_star, c2_m2, _ = self.synth.scan_m2(obs_g, centers_fit, cfg.N)
        (mu3, sig3), c2_m3 = self.synth.fit_m3(obs_g, centers_fit, cfg.N)
        ok = (c2_m3 < c2_m2) and (th_star >= cfg.THETA_STAR_MIN)
        if not ok:
            raise AIQValidationError(
                f"M2 否决判据不满足: M2 θ*={th_star:.1f}° χ²={c2_m2:.2f}; "
                f"M3 χ²={c2_m3:.2f}",
                expected={"c2_m3 < c2_m2": True, "theta* >= min": cfg.THETA_STAR_MIN},
                actual={"c2_m2": c2_m2, "c2_m3": c2_m3, "theta_star": th_star},
                param_key="D03",
            )
        return {
            "detail": f"M2 θ*={th_star:.1f}° χ²={c2_m2:.2f}; "
                      f"M3 μ={mu3:.1f}° σ={sig3:.1f}° χ²={c2_m3:.2f}; "
                      f"Δχ²(M2-M3)={c2_m2 - c2_m3:+.2f}",
            "theta_star": th_star, "c2_m2": c2_m2, "c2_m3": c2_m3,
        }

    # ------------------------------------------------------------ 3) 正控制
    def validate_positive_control(self) -> dict:
        """3) 正控制：截断均匀 U[0,30] -> θ* ≈ 30（README ⑤表第 3 行）。"""
        cfg = self.config
        bins, centers_full = self._bins_centers()
        fit_idx = np.arange(cfg.N_FIT)
        centers_fit = centers_full[fit_idx]
        phi_u = self.synth.synth_phi_uniform(0.0, 30.0, cfg.N)
        obs_u = np.histogram(phi_u, bins=bins)[0][fit_idx]
        th_u, c2_u, _ = self.synth.scan_m2(obs_u, centers_fit, cfg.N)
        ok = abs(th_u - 30.0) <= cfg.RECOVER_TOL
        if not ok:
            raise AIQValidationError(
                f"正控制恢复失败: θ*={th_u:.1f}° 应≈30°",
                expected=30.0, actual=th_u, param_key="D03",
            )
        return {
            "detail": f"θ*={th_u:.1f}° χ²={c2_u:.2f} (真实 θ=30°, 允许±{cfg.RECOVER_TOL}°)",
            "theta_star": th_u, "c2": c2_u,
        }

    # ------------------------------------------------------------ 4) 真实模型对照
    def validate_real(self) -> dict:
        """4) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M2 搜索）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        bins, centers_full = self._bins_centers()
        fit_idx = np.arange(cfg.N_FIT)
        centers_fit = centers_full[fit_idx]
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成高斯
            phi_g = self.synth.synth_phi_trunc_gauss(cfg.MU, cfg.SIG, 0.0, 45.0, cfg.N)
            obs_g = np.histogram(phi_g, bins=bins)[0][fit_idx]
            _, c2_m2, _ = self.synth.scan_m2(obs_g, centers_fit, cfg.N)
            (_, _), c2_m3 = self.synth.fit_m3(obs_g, centers_fit, cfg.N)
            ok = c2_m3 < c2_m2
            if not ok:
                raise AIQValidationError(
                    f"回退数据 M3 应优于 M2: M3={c2_m3:.2f}, M2={c2_m2:.2f}",
                    expected={"c2_m3 < c2_m2": True},
                    actual={"c2_m2": c2_m2, "c2_m3": c2_m3}, param_key="D03",
                )
            return {"detail": f"M3 χ²={c2_m3:.2f} < M2 χ²={c2_m2:.2f}",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        Nreal = float(len(phi_real))
        obs_r = np.histogram(phi_real, bins=bins)[0][fit_idx]
        th_r, c2_r_m2, _ = self.synth.scan_m2(obs_r, centers_fit, Nreal)
        (mu3_r, sig3_r), c2_r_m3 = self.synth.fit_m3(obs_r, centers_fit, Nreal)
        ok = (th_r >= cfg.THETA_STAR_MIN) and (c2_r_m3 < c2_r_m2)
        if not ok:
            raise RealModelMismatchError(
                f"真实数据 M2 否决判据不满足: θ*={th_r:.1f}°, M2 χ²={c2_r_m2:.2f}, "
                f"M3 χ²={c2_r_m3:.2f}",
                expected={"theta* >= min": cfg.THETA_STAR_MIN, "c2_m3 < c2_m2": True},
                actual={"theta_star": th_r, "c2_m2": c2_r_m2, "c2_m3": c2_r_m3},
                param_key="D03",
            )
        return {
            "detail": (f"{tag} 真实 φ 数据 M2 θ*={th_r:.1f}° χ²={c2_r_m2:.2f}; "
                       f"M3 μ={mu3_r:.1f}° σ={sig3_r:.1f}° χ²={c2_r_m3:.2f}; "
                       f"Δχ²(M2-M3)={c2_r_m2 - c2_r_m3:+.2f}（真实数据确认 M2 否决）"),
            "source": rd.source_tag(), "tag": tag,
            "theta_star": th_r, "c2_m2": c2_r_m2, "c2_m3": c2_r_m3,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "grid", self.validate_grid),
            (2, "gauss_m2_vs_m3", self.validate_gauss_m2_vs_m3),
            (3, "positive_control", self.validate_positive_control),
            (4, "real_model", self.validate_real),
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
    """D03 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D03 theta_grid 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = ThetaSynthesizer(cfg)              # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D03 theta_grid 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N={cfg.N} N_FIT={cfg.N_FIT} MU={cfg.MU} SIG={cfg.SIG} "
          f"THETA_GRID={cfg.THETA_GRID} MU_GRID={cfg.MU_GRID} "
          f"SIGMA_GRID={cfg.SIGMA_GRID} THETA_STAR_MIN={cfg.THETA_STAR_MIN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d03_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
