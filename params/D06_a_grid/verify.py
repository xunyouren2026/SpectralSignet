# -*- coding: utf-8 -*-
"""D06 a_grid — M4 尖峰模型强度 a 搜索：负/正控制验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. a 网格完整性：arange(0,0.61,0.01)，61 点，步长 0.01
  2. 负控制：纯截断高斯数据 -> 最优 a≈0，Δχ²(M3->M4)<5.99 -> M4 否决（复现实测）
  3. 正控制：背景 + 0°尖峰(a=0.30, sp=1.5°) -> 最优 a≈0.30，Δχ²>5.99
     -> 尖峰显著（算法能检出）
  4. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M4 (a,sp) 联合网格：
     真实最优 a*=0.0、Δχ²=0.0 < 5.99 -> 无 0° 尖峰、M4 否决）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D06Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D06Config（环境变量 AIQ_D06_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            a/sp 网格经 get_grid 建模，跨节点在 build() 解析）
  SpikeSynthesizer       —— 混合分布合成（截断高斯+尖峰）+ M4 (a,sp) 联合网格搜索
  ValidatorEngine        —— 4 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2529-2676（D06 六步流程，尖峰在 φ≈0°）
  源码 _phi_fit_chi2.py 行 139-146、164-166（Δχ² 判定）
  《参数审计与实验报告.txt》行 57、122、140（状态=已用，M4 否决）
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

# ---- 第一层：配置模型 D06Config（pydantic 优先；dataclass 回退） ----
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

        子类 D06Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D06Config(_ConfigModelBase):
    """D06 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（SP_GRID）
    在 ConfigFactory.build() 中按来源节点显式解析（D07）。
    """

    MU_BG: float = 14.0             # M3 背景参数 μ（实测 14.1）
    SIG_BG: float = 25.0            # M3 背景参数 σ（实测 25.1）
    N: int = 200000                 # 合成样本数
    N_FIT: int = 39                 # 拟合域格数
    CRIT: float = 5.99              # T05: df=2, α=0.05（D06.CRIT_DCHI2 同源）
    ACC_BG: float = 0.605           # N(14,25) 在 [0,45] 理论接受率 = Φ(1.24)-Φ(-0.56)
    A_GRID: tuple = (0.0, 0.6, 61)  # a 网格：(start, stop, count)（README ①）
    SP_GRID: tuple = (0.6, 6.0, 28) # sp 网格：(start, stop, count)（D07 定义）
    A_TOL: float = 0.05             # a* 恢复容差（负/正控制判据）
    SEED: int = 0                   # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D06Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D06 配置工厂：实例化 D06Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D06Config:
        cfg = self.build_model(D06Config, "D06")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D07）
        cfg.SP_GRID = self.get_grid("D07", "SP_GRID", (0.6, 6.0, 28))
        # D 组网格参数：get_grid 保留 (start, stop, count) 语义
        cfg.A_GRID = self.get_grid("D06", "A_GRID", (0.0, 0.6, 61))
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
_erf = np.vectorize(erf)


def norm_cdf(x: float, mu: float, sig: float) -> float:
    """正态 CDF（math.erf 实现，兼容 numpy 2.x 无 np.erf）。"""
    return 0.5 * (1.0 + _erf((x - mu) / (sig * sqrt(2.0))))


def pdf_bg(centers: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """截断高斯背景 bin 密度（格宽 1°，README ②）。"""
    lo = centers - 0.5
    hi = centers + 0.5
    denom = norm_cdf(45.0, mu, sig) - norm_cdf(0.0, mu, sig)
    assert float(np.max(denom)) > 0.0, f"pdf_bg: 归一化分母为 0（μ={mu},σ={sig}）"
    return (norm_cdf(hi, mu, sig) - norm_cdf(lo, mu, sig)) / denom


def pdf_spike(centers: np.ndarray, sp: float) -> np.ndarray:
    """0° 尖峰：p∝exp(-φ²/2sp²)，截断[0,45]归一化（主文档行 2600-2601）。

    边界防御：sp>0 且归一化分母为正。
    """
    assert sp > 0.0, f"pdf_spike: 峰宽 sp={sp} 必须>0"
    p = np.exp(-0.5 * (centers / sp) ** 2)
    assert p.sum() > 0.0, f"pdf_spike: 归一化分母为 0（sp={sp}）"
    return p / p.sum()


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


# ---- 第三层：合成器 SpikeSynthesizer（算法与原脚本完全一致） ----
class SpikeSynthesizer(_SynthBase):
    """D06 合成器：混合分布合成（截断高斯+尖峰）+ M4 (a,sp) 联合网格搜索。"""

    def __init__(self, cfg: D06Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_mix(self, a_true: float, sp_true: float, n: int,
                  seed: int | None = None) -> np.ndarray:
        """混合采样：(1-a)截断高斯 + a 尖峰（一次性超量采样，避免循环累积）。"""
        cfg = self._cfg
        assert n > 0, f"synth_mix: 样本数 n={n} 必须>0"
        assert 0.0 <= a_true <= 1.0, f"synth_mix: 尖峰强度 a={a_true} 超出 [0,1]"
        rng = np.random.default_rng(self._seed if seed is None else seed)
        n_pk = int(rng.binomial(n, a_true))
        n_bg = n - n_pk
        buf = rng.normal(cfg.MU_BG, cfg.SIG_BG, int(n_bg / cfg.ACC_BG) + 1000)
        out_bg = buf[(buf >= 0) & (buf <= 45)][:n_bg]
        assert out_bg.size == n_bg, f"synth_mix: 背景采样不足 {out_bg.size}/{n_bg}"
        pk = np.abs(rng.normal(0.0, sp_true, n_pk))
        pk = pk[pk <= 45]
        guard = 0
        while len(pk) < n_pk:
            guard += 1
            assert guard < 1000, "synth_mix: 尖峰拒绝采样未收敛（接受率过低）"
            add = np.abs(rng.normal(0.0, sp_true, n_pk))
            pk = np.concatenate([pk, add[add <= 45]])
        return np.concatenate([out_bg, pk[:n_pk]])

    def scan_m4(self, obs_fit: np.ndarray, centers_fit: np.ndarray,
                Ntot: float, bg_params: tuple) -> tuple:
        """(a,sp) 联合网格：a∈[0,0.6]步0.01 × sp∈[0.6,6]步0.2（README ②公式）。

        返回 (χ²_M4, (a*,sp*), χ²_M3)。
        """
        cfg = self._cfg
        a_lo, a_hi, n_a = cfg.A_GRID
        sp_lo, sp_hi, n_sp = cfg.SP_GRID
        a_grid = np.linspace(a_lo, a_hi, n_a)
        sp_grid = np.linspace(sp_lo, sp_hi, n_sp)
        pb = pdf_bg(centers_fit, *bg_params)
        best = (1e18, None)
        for a in a_grid:
            for sp in sp_grid:
                pmix = (1 - a) * pb + a * pdf_spike(centers_fit, sp)
                c2 = chi2_merge(obs_fit, Ntot * pmix)
                if c2 < best[0]:
                    best = (c2, (a, sp))
        c2_m3 = chi2_merge(obs_fit, Ntot * pb)   # M3 基线（a=0）
        return best[0], best[1], c2_m3


# ---- 第四层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D06 验证引擎：顺序执行 4 项验证（3 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D06Config,
        synth: SpikeSynthesizer,
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

    def _bins_fit(self, phi: np.ndarray) -> tuple:
        """φ 数组 -> (obs_fit, centers_fit)（45 格直方图，取前 N_FIT 格）。"""
        cfg = self.config
        bins = np.linspace(0, 45, 46)
        centers_full = (bins[:-1] + bins[1:]) / 2
        fit_idx = np.arange(cfg.N_FIT)
        return np.histogram(phi, bins=bins)[0][fit_idx], centers_full[fit_idx]

    # ------------------------------------------------------------ 1) a 网格完整性
    def validate_grid(self) -> dict:
        """1) a 网格完整性（README ①）。"""
        cfg = self.config
        a_grid = np.linspace(*cfg.A_GRID)
        da = np.diff(a_grid)
        ok = (len(a_grid) == 61) and bool(np.allclose(da, 0.01))
        if not ok:
            raise AIQValidationError(
                f"a 网格不符: n={len(a_grid)}",
                expected=61, actual=len(a_grid), param_key="D06",
            )
        return {
            "detail": f"n={len(a_grid)}, 范围[{a_grid[0]},{a_grid[-1]}], 步长一致",
            "n": len(a_grid),
        }

    # ------------------------------------------------------------ 2) 负控制
    def validate_negative_control(self) -> dict:
        """2) 负控制：纯截断高斯（无尖峰，README ⑤表第 2 行）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        buf = rng.normal(cfg.MU_BG, cfg.SIG_BG, int(cfg.N / cfg.ACC_BG) + 1000)
        phi_bg = buf[(buf >= 0) & (buf <= 45)][:cfg.N]
        obs_fit, centers_fit = self._bins_fit(phi_bg)
        c2_neg, (a_neg, sp_neg), c2_m3_neg = self.synth.scan_m4(
            obs_fit, centers_fit, cfg.N, (cfg.MU_BG, cfg.SIG_BG))
        d_neg = c2_m3_neg - c2_neg
        ok = (a_neg <= cfg.A_TOL) and (d_neg < cfg.CRIT)
        if not ok:
            raise AIQValidationError(
                f"负控制不满足: a*={a_neg:.2f}, Δχ²={d_neg:.2f}",
                expected={"a* <= tol": cfg.A_TOL, "dchi2 < crit": cfg.CRIT},
                actual={"a_star": a_neg, "dchi2": d_neg}, param_key="D06",
            )
        return {
            "detail": f"a*={a_neg:.2f}, sp*={sp_neg:.1f}°, χ²(M4)={c2_neg:.2f}, "
                      f"χ²(M3)={c2_m3_neg:.2f}, Δχ²={d_neg:.2f} "
                      f"(实测对照: a*=0.010, Δχ²=2.3)",
            "a_star": a_neg, "sp_star": sp_neg, "dchi2": d_neg,
        }

    # ------------------------------------------------------------ 3) 正控制
    def validate_positive_control(self) -> dict:
        """3) 正控制：背景 + 尖峰(a=0.30, sp=1.5，README ⑤表第 3 行)。"""
        cfg = self.config
        # 合成构造值（a=0.30, sp=1.5°）为"合成构造参数"（规范 7b 允许保留）
        phi_pos = self.synth.synth_mix(0.30, 1.5, cfg.N, seed=7)
        obs_fit, centers_fit = self._bins_fit(phi_pos)
        c2_pos, (a_pos, sp_pos), c2_m3_pos = self.synth.scan_m4(
            obs_fit, centers_fit, cfg.N, (cfg.MU_BG, cfg.SIG_BG))
        d_pos = c2_m3_pos - c2_pos
        ok = (d_pos > cfg.CRIT) and (abs(a_pos - 0.30) <= cfg.A_TOL)
        if not ok:
            raise AIQValidationError(
                f"正控制不满足: a*={a_pos:.2f}, Δχ²={d_pos:.2f}",
                expected={"dchi2 > crit": cfg.CRIT, "a*≈0.30": cfg.A_TOL},
                actual={"a_star": a_pos, "dchi2": d_pos}, param_key="D06",
            )
        return {
            "detail": f"a*={a_pos:.2f}, sp*={sp_pos:.1f}°, χ²(M4)={c2_pos:.2f}, "
                      f"χ²(M3)={c2_m3_pos:.2f}, Δχ²={d_pos:.2f}, "
                      f"a 恢复误差={abs(a_pos - 0.30):.2f}",
            "a_star": a_pos, "sp_star": sp_pos, "dchi2": d_pos,
        }

    # ------------------------------------------------------------ 4) 真实模型对照
    def validate_real(self) -> dict:
        """4) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 φ 分布 M4 搜索）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        phi_real = load_phi_real(rd)
        if phi_real is None:
            # 审计回退：真实数据缺失时回退合成纯高斯（与原脚本语义一致）
            rng = np.random.default_rng(cfg.SEED)
            buf = rng.normal(cfg.MU_BG, cfg.SIG_BG, int(cfg.N / cfg.ACC_BG) + 1000)
            phi_bg = buf[(buf >= 0) & (buf <= 45)][:cfg.N]
            obs_fit, centers_fit = self._bins_fit(phi_bg)
            c2_neg, (a_neg, _), c2_m3_neg = self.synth.scan_m4(
                obs_fit, centers_fit, cfg.N, (cfg.MU_BG, cfg.SIG_BG))
            d_neg = c2_m3_neg - c2_neg
            ok = (a_neg <= cfg.A_TOL) and (d_neg < cfg.CRIT)
            if not ok:
                raise AIQValidationError(
                    f"回退纯高斯负控制不满足: a*={a_neg:.2f}, Δχ²={d_neg:.2f}",
                    expected={"a* <= tol": cfg.A_TOL, "dchi2 < crit": cfg.CRIT},
                    actual={"a_star": a_neg, "dchi2": d_neg}, param_key="D06",
                )
            return {"detail": f"a*={a_neg:.2f} Δχ²={d_neg:.2f}（真实数据未就绪）",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        Nreal = float(len(phi_real))
        obs_r, centers_fit_r = self._bins_fit(phi_real)
        c2_r4, (a_r, sp_r), c2_r3 = self.synth.scan_m4(
            obs_r, centers_fit_r, Nreal, (cfg.MU_BG, cfg.SIG_BG))
        d_r = c2_r3 - c2_r4
        ok = (a_r <= cfg.A_TOL) and (d_r < cfg.CRIT)
        if not ok:
            raise RealModelMismatchError(
                f"真实数据 M4 否决判据不满足: a*={a_r:.2f}, Δχ²={d_r:.2f}",
                expected={"a* <= tol": cfg.A_TOL, "dchi2 < crit": cfg.CRIT},
                actual={"a_star": a_r, "dchi2": d_r}, param_key="D06",
            )
        return {
            "detail": (f"{tag} 真实 φ 数据 M4: a*={a_r:.2f}, sp*={sp_r:.1f}°, "
                       f"χ²(M4)={c2_r4:.2f}, χ²(M3)={c2_r3:.2f}, Δχ²={d_r:.2f} < {cfg.CRIT} "
                       f"（真实数据确认审计 M4 否决，K<0%=45.5% 双曲主导）"),
            "source": rd.source_tag(), "tag": tag,
            "a_star": a_r, "sp_star": sp_r, "dchi2": d_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "grid", self.validate_grid),
            (2, "negative_control", self.validate_negative_control),
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
    """D06 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D06 a_grid 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = SpikeSynthesizer(cfg)              # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D06 a_grid 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MU_BG={cfg.MU_BG} SIG_BG={cfg.SIG_BG} N={cfg.N} N_FIT={cfg.N_FIT} "
          f"CRIT={cfg.CRIT} ACC_BG={cfg.ACC_BG} A_GRID={cfg.A_GRID} "
          f"SP_GRID={cfg.SP_GRID} A_TOL={cfg.A_TOL} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d06_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d06_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d06_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
