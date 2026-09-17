# -*- coding: utf-8 -*-
"""T06 闭环判据 λ↔E[DEFF] 反解-实测一致性 — 曲率幅度模型自洽校验
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 重建 λ→E[DEFF] MC 映射，理论锚点 λ=1 -> E[DEFF]=1+2/π（各向同性）
  2. 反解 λ*(target_deff=1.5711) 落入审计报告预期区间 [0.41, 0.52]
  3. 判据 |λ*-λ_实测|/λ_实测 < 15%：实测 λ=0.852 下判「未闭环」✗
  4. 判据函数有效性：NS 单目标 λ*=0.853 vs 0.8516 -> 0.16% 闭环；
     λ*=0.910 vs 0.8516 -> 6.9% 闭环（文档标注已偏大）
  5. MC 采样误差标注（λ=0.45 处 5 次重采 std，附注）
  6. 工程化防御：除零（lam_meas=0）/空输入/非法网格参数显式报错
  7. 真实模型实测对照：真实 λ=0.8516、DEFF=1.5920 代入闭环判据 -> 未闭环
  8. 真实 vs 文档审计 λ 差异（舍入级 < 0.01，真实实测为准）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T06Config / ConfigFactory / T06MappingSynthesizer / T06Validator /
  ReportGenerator / main —— 同 A01（env AIQ_T06_<KEY> 覆盖由共享工厂处理）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T06 节（行 10051-10090）
  源码 _phi_reverse_v2.py（λ↔E[DEFF] MC 反解）
  《AI几何指纹插件_参数审计与实验报告.txt》306 λ=σ2/σ1（实测 0.852，未闭环）
真实模型对照：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测 phi_pairs_all.npy 得
  λ=0.8516、DEFF=1.5920（_real_metrics.json curvature），代入闭环判据：
  反解 λ*(DEFF=1.5920)≈0.519，|λ*-λ_实测|/λ_实测≈39% > 15% -> 未闭环 ✗，
  与文档审计「未闭环」结论一致（审计 λ=0.852，真实 λ=0.8516，偏差 0.0004）。
  输出标注：[真实实测]（数据齐全）/ [审计回退]（缺失）。真实实测为准。
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

# ---- 第一层：配置模型 T06Config（pydantic 优先；dataclass 回退） ----
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


class T06Config(_ConfigModelBase):
    """T06 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T06 节点键名一一对应（未在 JSON 的字段
    以模型默认值兜底）；取值优先级：环境变量 AIQ_T06_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 7                # λ→E[DEFF] 映射 MC 固定种子（README ④Step1）
    MC_N: int = 80000            # 轻量 MC 采样规模（源码用 400000，误差更大）
    GRID: int = 60               # lam_grid 网格点数（README ④Step1 为 40 点，取 60 更密）
    CLOSURE_THR: float = 0.15    # 闭环判据阈值 15%（D11 closure_thr）
    LAM_MEAS_AUDIT: float = 0.852   # 306 λ=σ2/σ1 实测（审计报告口径）
    TARGET_DEFF: float = 1.5711     # D10 平台中心目标（NS 平台，README ④Step3）
    LAM_STAR_LO: float = 0.41       # 审计报告预期反解区间下限（README ⑤）
    LAM_STAR_HI: float = 0.52       # 审计报告预期反解区间上限（README ⑤）
    ISO_TOL: float = 0.002          # 理论锚点 MC 对照容差
    NS_LAM_STAR: float = 0.853      # NS 单目标反解（README ④Step4 表格口径）
    LAM_MEAS: float = 0.8516        # 真实/审计 λ=0.8516（数据层 LAM_MEAS）
    AI_LAM_STAR: float = 0.910      # AI 单目标反解（README ④Step4 表格口径）
    DEFF_REF: float = 1.5920        # 真实 DEFF 审计参考（真实对照回退用）


# ---- 第二层：配置工厂 ConfigFactory（实例化 T06Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T06 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T06Config。"""

    def build(self) -> T06Config:
        """构建 T06Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T06Config, "T06")


# ---------------- 算法逻辑常量（数学公式常量） ----------------
THEORY_ISO = 1 + 2 / np.pi   # λ=1（各向同性）理论 E[DEFF]（README ④Step1，数学公式）


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def closure_check(lam_star: float, lam_meas: float, thr: float) -> tuple[bool, float]:
    """闭环判据：返回 (是否闭环, 相对偏差)。

    相对偏差 = |λ* - λ_实测| / |λ_实测|；λ_实测 为零时除零防御显式报错。
    """
    if lam_meas == 0.0:
        raise ValueError(f"closure_check: 实测 λ 为零，相对偏差无定义（除零防御）")
    dev = abs(lam_star - lam_meas) / abs(lam_meas)   # 相对偏差 = 归一化到实测值
    return dev < thr, dev                            # 偏差小于阈值 => 模型自洽闭环


def phi_from_pairs(kk: np.ndarray) -> np.ndarray:
    """逐对主曲率 (|κ₁|,|κ₂|) -> 主曲率角 φ = arctan(min/max)（度）。"""
    kk = np.asarray(kk, dtype=float)
    assert kk.ndim == 2 and kk.shape[1] == 2, f"phi_from_pairs: 需 (N,2) 数组，收到 {kk.shape}"
    assert kk.shape[0] > 0, "phi_from_pairs: 输入为空（无曲率对）"
    assert np.isfinite(kk).all(), f"phi_from_pairs: 输入含 NaN/Inf: {kk}"
    a = np.abs(kk[:, 0])                    # 第一主曲率绝对值
    b = np.abs(kk[:, 1])                    # 第二主曲率绝对值
    bb = np.maximum(a, b)                   # 大主曲率（分母）
    ss = np.minimum(a, b)                   # 小主曲率（分子）
    return np.degrees(np.arctan(ss / np.maximum(bb, 1e-12)))   # φ∈[0,45]°，ε 防除零


# ---- 第三层：合成器 T06MappingSynthesizer（λ→E[DEFF] MC 映射重建） ----
class T06MappingSynthesizer(_SynthBase):
    """T06 映射合成器：重建 λ→E[DEFF] 查找表（各向异性高斯 MC，参照 _phi_reverse_v2.py）。"""

    def __init__(self, cfg: T06Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def build_mapping(self, n: int | None = None, seed: int | None = None,
                      grid: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """重建 λ→E[DEFF] 映射（各向异性高斯 MC）。

        对每个 λ 用同一随机种子生成同一基础样本路径，再按 λ 缩放第二个分量，
        使不同 λ 共享随机路径、映射更平滑单调（有意设计，README ④Step1）。
        返回 (lam_grid, deffs)，二者长度均为 grid 且 deffs 单调递增。
        """
        cfg = self._cfg
        if n is None:
            n = cfg.MC_N
        if seed is None:
            seed = self._seed
        if grid is None:
            grid = cfg.GRID
        assert n > 0, f"build_mapping: MC 采样数 n 必须 > 0，收到 {n}"
        assert grid >= 2, f"build_mapping: 网格点数 grid 必须 >= 2，收到 {grid}"
        lam_grid = np.linspace(0.05, 1.0, grid)   # λ 网格：从近退化到各向同性
        deffs = []
        for lam in lam_grid:
            rng = np.random.default_rng(seed)     # 每个 λ 都用同一种子 => 共享随机路径
            # 各向异性高斯协方差 diag(1, λ²)：λ 控制第二分量方差（长轴=1）
            k = rng.multivariate_normal([0, 0], [[1.0, 0], [0, lam ** 2]], size=n)
            ph = phi_from_pairs(k)                # 主曲率角
            # E[DEFF] 估计：DEFF 逐点公式在 φ 域的期望 = mean(1+sin 2φ)
            deffs.append(float(np.mean(1 + np.sin(np.radians(2 * ph)))))
        out = np.array(deffs)
        if not np.isfinite(out).all():
            raise SynthesisError("build_mapping: 映射含 NaN/Inf", actual=out.shape)
        return lam_grid, out


# ---- 第四层：验证引擎 T06Validator（8 项验证 + 结构化日志 + 类型化异常） ----
class T06ValidationError(AIQValidationError):
    """T06 闭环判据验证失败。"""


class T06Validator(_EngineBase):
    """T06 验证引擎：顺序执行 8 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 映射表在步骤 1 重建并缓存（self.lam_grid/self.deffs），供后续步骤复用。
    """

    def __init__(
        self,
        config: T06Config,
        synth: T06MappingSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.lam_grid: np.ndarray | None = None   # λ 网格（步骤 1 重建）
        self.deffs: np.ndarray | None = None      # E[DEFF] 映射（步骤 1 重建）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 映射锚点
    def validate_anchor(self) -> dict:
        """1) 重建映射 + 理论锚点校验 λ=1 -> E[DEFF]=1+2/π（README ④Step1）。"""
        cfg = self.config
        assert self.synth is not None
        lam_grid, deffs = self.synth.build_mapping()
        self.lam_grid, self.deffs = lam_grid, deffs
        deff_iso = deffs[-1]                      # 末点即 λ=1（各向同性）情形
        ok = bool(np.isclose(deff_iso, THEORY_ISO, atol=cfg.ISO_TOL, rtol=0.0))
        if not ok:
            raise T06ValidationError(
                f"λ=1 锚点偏差过大: {deff_iso} vs 理论 {THEORY_ISO}",
                expected=THEORY_ISO, actual=deff_iso, param_key="T06",
            )
        return {"detail": (f"映射锚点 λ=1.0 -> E[DEFF]={deff_iso:.4f} "
                           f"(理论 1+2/π={THEORY_ISO:.4f}, 偏差 "
                           f"{abs(deff_iso - THEORY_ISO):.4f})"),
                "deff_iso": deff_iso, "theory_iso": THEORY_ISO,
                "range": [float(deffs[0]), float(deffs[-1])]}

    # ------------------------------------------------------------ 2) 反解 λ*
    def validate_invert(self) -> dict:
        """2) 反解 λ*（target_deff 须落在映射范围内，否则显式报错）。"""
        cfg = self.config
        assert self.deffs is not None and self.lam_grid is not None
        deffs, lam_grid = self.deffs, self.lam_grid
        in_range = bool(deffs[0] < cfg.TARGET_DEFF < deffs[-1])
        if not in_range:
            raise T06ValidationError(
                f"target_deff={cfg.TARGET_DEFF} 不在映射范围 "
                f"E[DEFF]∈[{deffs[0]:.3f},{deffs[-1]:.3f}]",
                expected=[deffs[0], deffs[-1]], actual=cfg.TARGET_DEFF, param_key="T06",
            )
        lam_star = float(np.interp(cfg.TARGET_DEFF, deffs, lam_grid))  # 线性插值反解
        if not np.isfinite(lam_star):
            raise T06ValidationError(
                f"反解 λ* 非有限: {lam_star}",
                expected="finite", actual=lam_star, param_key="T06",
            )
        ok = bool(cfg.LAM_STAR_LO <= lam_star <= cfg.LAM_STAR_HI)   # 落入审计预期区间
        if not ok:
            raise T06ValidationError(
                f"反解 λ*={lam_star} 不在预期区间 [{cfg.LAM_STAR_LO}, {cfg.LAM_STAR_HI}]",
                expected=[cfg.LAM_STAR_LO, cfg.LAM_STAR_HI], actual=lam_star,
                param_key="T06",
            )
        return {"detail": (f"反解 λ* (target_deff={cfg.TARGET_DEFF}) = {lam_star:.3f} "
                           f"(映射范围 E[DEFF]∈[{deffs[0]:.3f},{deffs[-1]:.3f}]) "
                           f"落入审计报告预期区间 [{cfg.LAM_STAR_LO}, {cfg.LAM_STAR_HI}]"),
                "lam_star": lam_star, "target_deff": cfg.TARGET_DEFF}

    # ------------------------------------------------------------ 3) 判据：未闭环
    def validate_not_closed(self) -> dict:
        """3) 判据判定：审计实测 λ=0.852 应判「未闭环」（README ④Step3）。"""
        cfg = self.config
        assert self.deffs is not None and self.lam_grid is not None
        lam_star = float(np.interp(cfg.TARGET_DEFF, self.deffs, self.lam_grid))
        closed, dev = closure_check(lam_star, cfg.LAM_MEAS_AUDIT, cfg.CLOSURE_THR)
        ok = (not closed) and dev > cfg.CLOSURE_THR   # 审计实测 λ=0.852 应判未闭环
        if not ok:
            raise T06ValidationError(
                f"应判未闭环: dev={dev * 100:.1f}% 应 >{cfg.CLOSURE_THR * 100:.0f}%",
                expected=f"> {cfg.CLOSURE_THR}", actual=dev, param_key="T06",
            )
        return {"detail": (f"判据: |{lam_star:.3f}-{cfg.LAM_MEAS_AUDIT}|/"
                           f"{cfg.LAM_MEAS_AUDIT} = {dev * 100:.1f}% > 15% "
                           f"-> 未闭环 ✗"),
                "lam_star": lam_star, "lam_meas": cfg.LAM_MEAS_AUDIT, "dev": dev}

    # ------------------------------------------------------------ 4a) 判据有效性 NS
    def validate_closure_ns(self) -> dict:
        """4a) 判据函数有效性：NS 单目标口径下应判闭环（自洽情形）。"""
        cfg = self.config
        ok, dev_a = closure_check(cfg.NS_LAM_STAR, cfg.LAM_MEAS, cfg.CLOSURE_THR)
        if not (ok and dev_a < cfg.CLOSURE_THR):
            raise T06ValidationError(
                f"NS 口径应判闭环: dev={dev_a * 100:.2f}% 应 <{cfg.CLOSURE_THR * 100:.0f}%",
                expected=f"< {cfg.CLOSURE_THR}", actual=dev_a, param_key="T06",
            )
        return {"detail": (f"对照: λ*={cfg.NS_LAM_STAR} vs {cfg.LAM_MEAS} -> 偏差 "
                           f"{dev_a * 100:.2f}% < 15% -> 闭环 ✅（判据能识别自洽情形）"),
                "dev": dev_a, "lam_star": cfg.NS_LAM_STAR, "lam_meas": cfg.LAM_MEAS}

    # ------------------------------------------------------------ 4b) 判据有效性 AI
    def validate_closure_ai(self) -> dict:
        """4b) 判据函数有效性：AI 单目标口径下应判闭环（文档标注已偏大）。"""
        cfg = self.config
        ok, dev_b = closure_check(cfg.AI_LAM_STAR, cfg.LAM_MEAS, cfg.CLOSURE_THR)
        if not (ok and dev_b < cfg.CLOSURE_THR):
            raise T06ValidationError(
                f"AI 口径应判闭环: dev={dev_b * 100:.2f}% 应 <{cfg.CLOSURE_THR * 100:.0f}%",
                expected=f"< {cfg.CLOSURE_THR}", actual=dev_b, param_key="T06",
            )
        return {"detail": (f"对照: λ*={cfg.AI_LAM_STAR} vs {cfg.LAM_MEAS} -> 偏差 "
                           f"{dev_b * 100:.2f}% < 15% -> 闭环（但已偏大, 文档标注）"),
                "dev": dev_b, "lam_star": cfg.AI_LAM_STAR, "lam_meas": cfg.LAM_MEAS}

    # ------------------------------------------------------------ 5) MC 误差标注（附注）
    def validate_mc_error(self) -> dict:
        """5) MC 采样误差标注（附注，不参与判定；README ④Step5）。"""
        cfg = self.config
        reps = []
        for i in range(5):                     # 5 次独立重采的标准差量化 MC 噪声量级
            kk = np.random.default_rng(100 + i).multivariate_normal(
                [0, 0], [[1.0, 0], [0, 0.45 ** 2]], size=cfg.MC_N)
            reps.append(np.mean(1 + np.sin(np.radians(2 * phi_from_pairs(kk)))))
        sem = float(np.std(reps))
        return {"detail": (f"[注] λ=0.45 处 E[DEFF] 的 5 次重采 std ≈ {sem:.4f} "
                           f"（映射表 MC 采样噪声量级；源码 MC_n=400000 时更小）"),
                "sem": sem}

    # ------------------------------------------------------------ 6) 防御
    def validate_guards(self) -> dict:
        """6) 工程化防御：除零 / 空输入 / 非法网格参数显式报错。"""
        guards = [
            ("实测 λ=0 (除零)", lambda: closure_check(1.0, 0.0, 0.15)),
            ("MC n=0", lambda: self.synth.build_mapping(n=0)),
            ("网格 grid=1", lambda: self.synth.build_mapping(grid=1)),
            ("phi 空输入", lambda: phi_from_pairs(np.empty((0, 2)))),
        ]
        guard_oks = []
        for _gname, fn in guards:
            try:
                fn()                             # 未抛异常 => 防御缺失
                guard_oks.append(False)
            except (ValueError, AssertionError):
                guard_oks.append(True)           # 显式报错 => 防御通过
        ok_guard = all(guard_oks)
        if not ok_guard:
            raise T06ValidationError(
                f"边界防御失败: {[g[0] for g, o in zip(guards, guard_oks) if not o]}",
                expected="all raise", actual=guard_oks, param_key="T06",
            )
        return {"detail": f"防御: 除零/空输入/非法网格 均显式报错 {guard_oks}",
                "guard_oks": guard_oks}

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型实测对照：真实 λ/DEFF 代入闭环判据 -> 未闭环（真实实测为准）。"""
        cfg = self.config
        assert self.deffs is not None and self.lam_grid is not None
        rd = self._get_real_data()
        src = rd.source_tag()
        deffs, lam_grid = self.deffs, self.lam_grid
        # 真实值优先，缺失回退审计值（数据层 DEFF_REF / LAM_MEAS）—— 保证缺数据环境可运行
        deff_real = rd.get("curvature.DEFF_plat", rd.audit("DEFF_plat", cfg.DEFF_REF))
        lam_real = rd.get("curvature.lambda_ratio", rd.audit("lambda_ratio", cfg.LAM_MEAS))
        in_range_r = bool(deffs[0] < deff_real < deffs[-1])   # 真实 DEFF 须在映射范围内
        if not in_range_r:                     # 审计回退：真实 DEFF 超出映射范围
            lam_star = float(np.interp(cfg.TARGET_DEFF, deffs, lam_grid))
            closed_audit, _ = closure_check(lam_star, cfg.LAM_MEAS_AUDIT, cfg.CLOSURE_THR)
            ok_fallback = not closed_audit
            if not ok_fallback:
                raise T06ValidationError(
                    "审计口径应判未闭环",
                    expected="not closed", actual="closed", param_key="T06",
                )
            return {"detail": (f"[{src}] 真实 DEFF={deff_real} 不在映射范围 "
                               f"[{deffs[0]:.3f},{deffs[-1]:.3f}], 维持审计未闭环结论"),
                    "fallback": True, "deff_real": deff_real}
        lam_star_r = float(np.interp(deff_real, deffs, lam_grid))  # 真实反解
        if not np.isfinite(lam_star_r):
            raise T06ValidationError(
                f"真实反解 λ* 非有限: {lam_star_r}",
                expected="finite", actual=lam_star_r, param_key="T06",
            )
        closed_r, dev_r = closure_check(lam_star_r, lam_real, cfg.CLOSURE_THR)
        ok_real = (not closed_r) and dev_r > cfg.CLOSURE_THR   # 预期真实模型同样未闭环
        if not ok_real:
            raise RealModelMismatchError(
                f"真实口径应判未闭环: dev={dev_r * 100:.1f}% 应 >{cfg.CLOSURE_THR * 100:.0f}%",
                expected=f"> {cfg.CLOSURE_THR}", actual=dev_r, param_key="T06",
            )
        return {
            "detail": (f"[{src}] 真实反解 λ*(DEFF={deff_real:.4f})={lam_star_r:.4f} "
                       f"vs 真实实测 λ={lam_real:.4f}: 相对偏差 {dev_r * 100:.1f}% "
                       f"> {cfg.CLOSURE_THR * 100:.0f}% -> 未闭环 ✗"),
            "source": src, "deff_real": deff_real, "lam_real": lam_real,
            "lam_star_r": lam_star_r, "dev": dev_r,
        }

    # ------------------------------------------------------------ 8) 真实 vs 审计
    def validate_real_audit(self) -> dict:
        """8) 真实 vs 文档审计 λ 差异：舍入级 < 0.01，未闭环结论一致。"""
        cfg = self.config
        rd = self._get_real_data()
        lam_real = rd.get("curvature.lambda_ratio", None)
        if lam_real is None:                     # 真实缺失：回退审计口径，视为一致
            return {"detail": "[审计回退] 真实 λ 缺失，跳过真实 vs 审计 λ 对照",
                    "fallback": True}
        diff_lam = abs(lam_real - cfg.LAM_MEAS_AUDIT)   # 两口径 λ 差
        ok = bool(diff_lam < 0.01)               # 舍入级差异
        if not ok:
            raise T06ValidationError(
                f"真实 λ={lam_real} 与审计 {cfg.LAM_MEAS_AUDIT} 偏差 {diff_lam} 超 0.01",
                expected=0.01, actual=diff_lam, param_key="T06",
            )
        return {
            "detail": (f"真实 vs 审计: λ 真实={lam_real:.4f} 审计={cfg.LAM_MEAS_AUDIT:.3f} "
                       f"(|Δ|={diff_lam:.4f} < 0.01; 未闭环结论两者一致, 真实实测为准)"),
            "lam_real": lam_real, "lam_audit": cfg.LAM_MEAS_AUDIT, "diff": diff_lam,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 8 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "anchor", self.validate_anchor),
            (2, "invert", self.validate_invert),
            (3, "not_closed", self.validate_not_closed),
            (4, "closure_ns", self.validate_closure_ns),
            (5, "closure_ai", self.validate_closure_ai),
            (6, "mc_error", self.validate_mc_error),
            (7, "guards", self.validate_guards),
            (8, "real", self.validate_real),
            (9, "real_audit", self.validate_real_audit),
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
    """T06 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T06 闭环判据 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = T06MappingSynthesizer(cfg)                # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T06Validator(cfg, synth, report, real_data=RD)  # ③ 验证层

    print("=" * 74)
    print(f"T06 闭环判据  |λ*-λ_实测|/λ_实测 < {cfg.CLOSURE_THR * 100:.0f}%")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: CLOSURE_THR={cfg.CLOSURE_THR} TARGET_DEFF={cfg.TARGET_DEFF} "
          f"MC_N={cfg.MC_N} GRID={cfg.GRID}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t06_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t06_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t06_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
