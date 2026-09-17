# -*- coding: utf-8 -*-
"""D05 sigma_grid — M3 截断高斯最优 σ 搜索：网格与恢复验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. σ 网格完整性：arange(2,30.01,0.5)，步长 0.5，覆盖 2-30°（57 点）
  2. 合成截断高斯 N(14°,25°)|[0,45]，固定 μ=14 扫 σ -> 恢复 σ*≈25°
  3. σ 固定扫描表复现主文档趋势：σ=25° 处 χ² 最小
  4. σ<10° 时 χ² 严重恶化（σ=8° >> σ=25°）
  5. 物理含义：68% 区间 [μ-σ*, μ+σ*]，σ/μ≈1.8（弥散度）
  6. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布：固定 μ=14° 扫 σ
     恢复 σ*=25.0°（χ²≈26），σ=25° 处 χ² 最小——正向确认）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D05Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D05Config（环境变量 AIQ_D05_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            σ 网格经 get_grid 建模）
  SigmaSynthesizer       —— φ 分布合成（拒绝采样）+ σ 扫描 χ² 表
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2399-2528（D05 七步流程）
  源码 _phi_fit_chi2.py 行 130-137（σ 维度扫描）
  《参数审计与实验报告.txt》行 17、56（状态=已用，σ→25°）
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

# ---- 第一层：配置模型 D05Config（pydantic 优先；dataclass 回退） ----
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

        子类 D05Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D05Config(_ConfigModelBase):
    """D05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 D05 节点键名一一对应。
    """

    MU_FIXED: float = 14.0          # 固定 μ（实测最优 14.1°）
    SIG_TRUE: float = 25.0          # 合成数据真实 σ（= 实测 25.1° 取整）
    N: int = 200000                 # 合成样本数
    N_FIT: int = 39                 # 拟合域格数
    TOL_SIG: float = 2.0            # σ 恢复容差（网格分辨率±0.5）
    SIGMA_GRID: tuple = (2.0, 30.0, 57)   # σ 网格：(start, stop, count)（README ①）
    SIG_SCAN: list = [8, 15, 20, 25, 28, 30]  # σ 固定扫描表（主文档行 2515-2520）
    DEGRADE_MULT: float = 3.0       # σ<10° 恶化倍数判据（χ²(8°)>3×χ²(25°)）
    RATIO_TOL: float = 0.4          # σ/μ 物理含义容差
    SEED: int = 0                   # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D05 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 D05Config。"""

    def build(self) -> D05Config:
        cfg = self.build_model(D05Config, "D05")   # pydantic/dataclass 回退
        # D 组网格参数：get_grid 保留 (start, stop, count) 语义
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


# ---- 第三层：合成器 SigmaSynthesizer（算法与原脚本完全一致） ----
class SigmaSynthesizer(_SynthBase):
    """D05 合成器：φ 分布拒绝采样 + σ 扫描 χ² 表。"""

    def __init__(self, cfg: D05Config) -> None:
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

    def sigma_chi2_table(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
                         Ntot: float, mu_fixed: float) -> dict:
        """固定 μ 扫 σ 网格 -> {σ: χ²}（README ②公式）。"""
        cfg = self._cfg
        sig_lo, sig_hi, n_sig = cfg.SIGMA_GRID
        table = {}
        for sig in np.linspace(sig_lo, sig_hi, n_sig):
            table[sig] = chi2_merge(obs_fit, exp_gauss(centers_fit, mu_fixed, sig, Ntot))
        return table


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D05 验证引擎：顺序执行 6 项验证（5 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D05Config,
        synth: SigmaSynthesizer,
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
        """合成截断高斯数据 -> (obs_fit, centers_fit, bins)。"""
        cfg = self.config
        phi = self.synth.synth_phi_trunc_gauss(cfg.MU_FIXED, cfg.SIG_TRUE, 0.0, 45.0, cfg.N)
        bins = np.linspace(0, 45, 46)
        centers_full = (bins[:-1] + bins[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)
        obs_fit = np.histogram(phi, bins=bins)[0][fit_idx]
        centers_fit = centers_full[fit_idx]
        return obs_fit, centers_fit, bins

    # ------------------------------------------------------------ 1) 网格完整性
    def validate_grid(self) -> dict:
        """1) σ 网格完整性（README ①）。"""
        cfg = self.config
        sigmas = np.linspace(*cfg.SIGMA_GRID)
        dsg = np.diff(sigmas)
        ok = (len(sigmas) == 57) and bool(np.allclose(dsg, 0.5))
        if not ok:
            raise AIQValidationError(
                f"σ 网格不符: n={len(sigmas)}",
                expected=57, actual=len(sigmas), param_key="D05",
            )
        return {
            "detail": f"n={len(sigmas)}, 范围[{sigmas[0]},{sigmas[-1]}], 步长一致",
            "n": len(sigmas),
        }

    # ------------------------------------------------------------ 2) σ 恢复
    def validate_sigma_recovery(self) -> dict:
        """2) 固定 μ=14 扫 σ -> 最优 σ*≈25（README ⑤表第 2 行）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        chi2_table = self.synth.sigma_chi2_table(obs_fit, centers_fit, cfg.N, cfg.MU_FIXED)
        sig_star = min(chi2_table, key=chi2_table.get)
        ok = abs(sig_star - cfg.SIG_TRUE) <= cfg.TOL_SIG
        if not ok:
            raise AIQValidationError(
                f"σ 恢复偏差: σ*={sig_star:.1f}° 需≈{cfg.SIG_TRUE}°",
                expected=cfg.SIG_TRUE, actual=sig_star, param_key="D05",
            )
        return {
            "detail": f"σ*={sig_star:.1f}° (真实 {cfg.SIG_TRUE}°, 实测 25.1°, "
                      f"允许±{cfg.TOL_SIG})",
            "sig_star": sig_star,
        }

    # ------------------------------------------------------------ 3) σ 固定扫描表
    def validate_scan_table(self) -> dict:
        """3) σ 固定扫描表：σ=25° 处 χ² 最小（README ⑤表第 3 行）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        chi2_table = self.synth.sigma_chi2_table(obs_fit, centers_fit, cfg.N, cfg.MU_FIXED)
        scan_str = "; ".join(f"σ={s}°:χ²={chi2_table[s]:.1f}" for s in cfg.SIG_SCAN)
        ok = chi2_table[25] <= min(chi2_table[s] for s in cfg.SIG_SCAN)
        if not ok:
            raise AIQValidationError(
                f"σ=25° 处 χ² 非最小: {scan_str}",
                expected={"sigma=25 min": True},
                actual={s: chi2_table[s] for s in cfg.SIG_SCAN}, param_key="D05",
            )
        return {
            "detail": scan_str + " (实测 8→156.3,15→78.5,20→52.1,25→42.1,28→44.7,30→48.2)",
            "table": {s: chi2_table[s] for s in cfg.SIG_SCAN},
        }

    # ------------------------------------------------------------ 4) σ<10° 恶化
    def validate_degradation(self) -> dict:
        """4) σ<10° 时 χ² 严重恶化（README ⑤表第 4 行）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        chi2_table = self.synth.sigma_chi2_table(obs_fit, centers_fit, cfg.N, cfg.MU_FIXED)
        c2_25 = chi2_table[25]
        c2_8 = chi2_table[8]
        ok = c2_8 > cfg.DEGRADE_MULT * c2_25
        if not ok:
            raise AIQValidationError(
                f"σ<10° 恶化不足: χ²(8°)={c2_8:.2f} vs χ²(25°)={c2_25:.2f}",
                expected={"ratio >": cfg.DEGRADE_MULT},
                actual=c2_8 / c2_25, param_key="D05",
            )
        return {
            "detail": f"χ²(8°)={c2_8:.2f} vs χ²(25°)={c2_25:.2f}, "
                      f"比值={c2_8 / c2_25:.1f}×>{cfg.DEGRADE_MULT}",
            "ratio": c2_8 / c2_25,
        }

    # ------------------------------------------------------------ 5) 物理含义
    def validate_physics(self) -> dict:
        """5) 物理含义：68% 区间与 σ/μ≈1.8（README ③几何直觉）。"""
        cfg = self.config
        obs_fit, centers_fit, _ = self._prep_synth()
        chi2_table = self.synth.sigma_chi2_table(obs_fit, centers_fit, cfg.N, cfg.MU_FIXED)
        sig_star = min(chi2_table, key=chi2_table.get)
        spread_lo = cfg.MU_FIXED - sig_star
        spread_hi = cfg.MU_FIXED + sig_star
        ratio_sm = sig_star / cfg.MU_FIXED
        ratio_ref = cfg.SIG_TRUE / cfg.MU_FIXED   # ≈25/14≈1.79（弥散度解析目标）
        ok = abs(ratio_sm - ratio_ref) < cfg.RATIO_TOL
        if not ok:
            raise AIQValidationError(
                f"σ/μ 偏离: {ratio_sm:.2f} 需≈{ratio_ref:.2f}",
                expected=ratio_ref, actual=ratio_sm, param_key="D05",
            )
        return {
            "detail": f"σ*={sig_star:.1f}° -> 68%区间 [{spread_lo:.0f}°, {spread_hi:.0f}°]"
                      f"（截断[0,45]）, σ/μ={ratio_sm:.2f} (实测≈1.8)",
            "sig_star": sig_star, "ratio": ratio_sm,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real(self) -> dict:
        """6) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 σ 扫描）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        bins = np.linspace(0, 45, 46)
        fit_idx = np.arange(cfg.N_FIT)
        centers_fit = ((bins[:-1] + bins[1:]) / 2)[fit_idx]
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成截断高斯（与原脚本语义一致）
            obs_fit, centers_fit_s, _ = self._prep_synth()
            chi2_tbl = self.synth.sigma_chi2_table(obs_fit, centers_fit_s, cfg.N, cfg.MU_FIXED)
            sig_star = min(chi2_tbl, key=chi2_tbl.get)
            ok = abs(sig_star - cfg.SIG_TRUE) <= cfg.TOL_SIG
            if not ok:
                raise AIQValidationError(
                    f"回退合成 σ 恢复偏差: σ*={sig_star:.1f}°",
                    expected=cfg.SIG_TRUE, actual=sig_star, param_key="D05",
                )
            return {"detail": f"σ*={sig_star:.1f}°（真实数据未就绪）",
                    "tag": "[审计回退]", "sig_star": sig_star, "fallback": True}
        tag = "[真实实测]"
        Nreal = float(len(phi_real))
        obs_r = np.histogram(phi_real, bins=bins)[0][fit_idx]
        chi2_r = self.synth.sigma_chi2_table(obs_r, centers_fit, Nreal, cfg.MU_FIXED)
        sig_star_r = min(chi2_r, key=chi2_r.get)
        okr1 = abs(sig_star_r - cfg.SIG_TRUE) <= cfg.TOL_SIG
        if not okr1:
            raise RealModelMismatchError(
                f"真实数据 σ 恢复偏差: σ*={sig_star_r:.1f}°",
                expected=cfg.SIG_TRUE, actual=sig_star_r, param_key="D05",
            )
        scan_r = "; ".join(f"σ={s}°:χ²={chi2_r[s]:.1f}" for s in cfg.SIG_SCAN)
        okr2 = chi2_r[25] <= min(chi2_r[s] for s in cfg.SIG_SCAN)
        if not okr2:
            raise RealModelMismatchError(
                f"真实数据 σ=25° 处 χ² 非最小: {scan_r}",
                expected={"sigma=25 min": True},
                actual={s: chi2_r[s] for s in cfg.SIG_SCAN}, param_key="D05",
            )
        return {
            "detail": (f"{tag} 真实 φ 数据固定 μ=14° 扫 σ: σ*={sig_star_r:.1f}° "
                       f"χ²={chi2_r[sig_star_r]:.2f}（文档审计 σ≈25°，一致）; "
                       f"{scan_r}（σ=25° χ² 最小）"),
            "source": rd.source_tag(), "tag": tag,
            "sig_star": sig_star_r, "chi2": chi2_r[sig_star_r],
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "grid", self.validate_grid),
            (2, "sigma_recovery", self.validate_sigma_recovery),
            (3, "scan_table", self.validate_scan_table),
            (4, "degradation", self.validate_degradation),
            (5, "physics", self.validate_physics),
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
    """D05 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D05 sigma_grid 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = SigmaSynthesizer(cfg)              # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D05 sigma_grid 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MU_FIXED={cfg.MU_FIXED} SIG_TRUE={cfg.SIG_TRUE} N={cfg.N} "
          f"N_FIT={cfg.N_FIT} TOL_SIG={cfg.TOL_SIG} SIGMA_GRID={cfg.SIGMA_GRID} "
          f"SIG_SCAN={cfg.SIG_SCAN} DEGRADE_MULT={cfg.DEGRADE_MULT} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d05_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
