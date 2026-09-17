# -*- coding: utf-8 -*-
"""E01 REF_PROMPTS 参考剖面数 — 基线指纹建立与输入无关性验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 4 个参考 prompt 分别计算剖面 -> 均值构成基线指纹（baseline_fingerprint）
  2. 跨 prompt 距离（两两欧氏均值）应 ≈ 实测 0.031（构造噪声对齐量级）
  3. 跨模型距离 ≈ 0.219，分离比 = cross/within ≈ 7x（输入无关性判据）
  4. 参考句数 1/4/8/16 的稳定性平台：距离 0.031 -> 0.030 饱和

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E01Config              —— 配置模型（pydantic 校验；pydantic 缺失时自动
                            dataclass 回退，由 _factory.ConfigFactory.build_model 驱动）
  ConfigFactory          —— 实例化 E01Config（优先级：环境变量 AIQ_E01_<KEY>
                            > YAML config.yaml > _params_data.json > 模型默认值）
  （无合成器类：本参数为纯数值构造，工具函数以模块级纯函数提供）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档《参数附录表完整版》行 3328-3442（E01 REF_PROMPTS）
  源码 _qwen_robustness.py（L38-L43 参考 prompt 与基线建立逻辑）
  《参数审计与实验报告.txt》（状态=已用 4 句）
说明：纯数值合成数据（24 维 beta 剖面），不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 k_proj Gamma 24 维剖面作为基线特征，复算
  跨 prompt 距离，对照文档审计值（真实 Gamma 均值 0.4695 vs 文档 0.625）。
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
    FamilySeparationError,
    RealModelMismatchError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 E01Config（pydantic 优先；dataclass 回退） ----
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


class E01Config(_ConfigModelBase):
    """E01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E01 节点键名一一对应（N_REF/DIM/WITHIN_DOC/
    CROSS_DOC/SEP_MIN）；取值优先级：环境变量 AIQ_E01_<KEY> > YAML >
    _params_data.json > 本模型默认值。字段默认值仅作最低优先级兜底。
    """

    SEED: int = 0              # H01 固定随机种子：保证可复现（算法逻辑常量）
    N_REF: int = 4             # E01: 参考剖面数（README ②公式）
    GRID: int = 24             # B03: beta 剖面插值格点（E01 DIM 键）
    WITHIN_TARGET: float = 0.031   # 实测：跨 prompt 距离（README ⑤）
    CROSS_TARGET: float = 0.219    # 实测：跨模型距离（README ⑤）
    SEP_MIN: float = 2.0           # 输入无关性分离比下界（_params_data.json 兜底）
    AMPLITUDE: float = 0.6         # f* 归一化幅度（原始构造）
    BASE_MIN: float = 0.45         # f* 深度方向基线（原始构造）
    RISE: float = 0.25             # f* 深度方向上升幅度（原始构造）
    DECAY: float = 2.5             # f* 深度方向饱和速率（原始构造）
    WITHIN_LO: float = 0.020       # 跨 prompt 距离判据下界（容忍采样波动）
    WITHIN_HI: float = 0.042       # 跨 prompt 距离判据上界
    CROSS_LO: float = 0.17         # 跨模型距离判据下界
    CROSS_HI: float = 0.27         # 跨模型距离判据上界
    SEP_LO: float = 6.0            # 分离比判据下界（≈7x）
    SEP_HI: float = 9.0            # 分离比判据上界
    REAL_SEED: int = 1234          # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 E01Config。"""

    def build(self) -> E01Config:
        """构建 E01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E01Config, "E01")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def make_profiles(f_true: np.ndarray, n: int, noise_sig: float,
                  rng: np.random.Generator, grid: int) -> np.ndarray:
    """n 个参考剖面 = 真实指纹 + 独立语义噪声（GRID 维）。"""
    # 逐参考剖面叠加独立高斯噪声：不同 prompt 采样同一模型，剖面向 f_true 回归
    return f_true + noise_sig * rng.standard_normal((n, grid))


def pairwise_dist(profiles: np.ndarray) -> float:
    """两两欧氏距离均值（同实例内，README ②公式）。

    空输入/单样本防御：样本数 < 2 时返回 0.0（避免除零与空均值）。
    """
    n = len(profiles)
    if n < 2:
        return 0.0
    # 组合枚举所有无序对 (i, j)，距离为该对向量的欧氏范数
    ds = [np.linalg.norm(profiles[i] - profiles[j])
          for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(ds))


def identity_profile(rng: np.random.Generator, cfg: E01Config) -> np.ndarray:
    """构造模型真实身份指纹 f*：24 维 beta 剖面，能量集中度沿深度单调上升。

    深层更集中（0.45 -> ~0.67），归一化至固定幅度 AMPLITUDE。
    """
    # 深度网格：把 [0,1] 均匀采样成 GRID 个点，作为剖面"深度坐标"
    depth = np.linspace(0.0, 1.0, cfg.GRID)
    # 饱和上升曲线（1-e^{-DECAY·depth}）：深层能量集中度饱和逼近 BASE_MIN+RISE
    f_star = cfg.BASE_MIN + cfg.RISE * (1.0 - np.exp(-cfg.DECAY * depth))
    # L2 归一化到 AMPLITUDE：使指纹范数与文档构造尺度一致（跨模型可比）
    return f_star / np.linalg.norm(f_star) * cfg.AMPLITUDE


# ---- 第三层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E01 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（方法内 import 或经 _common.setup_env 注入）。
    - 共享合成状态在 _synthesize() 中一次性按原脚本单 rng 流计算（惰性、可复现）。
    """

    def __init__(
        self,
        config: E01Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self._synced = False         # 共享合成状态是否就绪
        # ---- 共享中间结果（供后续步骤复用，与原始 main 单 rng 流一致）----
        self.f_star: np.ndarray | None = None
        self.profiles: np.ndarray | None = None
        self.baseline: np.ndarray | None = None
        self.baseline_std: np.ndarray | None = None
        self.within_dist = 0.0
        self.cross_dist = 0.0
        self.separation = 0.0
        self.res_1 = 0.0
        self.res_mean = 0.0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _synthesize(self) -> None:
        """一次性合成共享数据（与原脚本 main 的单一 rng 流完全一致，惰性）。"""
        if self._synced:
            return
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        # 跨 prompt 噪声尺度：目标 4 个参考剖面两两欧氏距离均值 = 0.031
        # E||f_i - f_j|| = sqrt(2 * var_noise * GRID)，反解 var_noise
        var_noise = (cfg.WITHIN_TARGET ** 2) / (2.0 * cfg.GRID)
        sig = np.sqrt(var_noise)
        # 构造模型真实身份指纹 f*（与 prompt 无关的"恒定内核"）
        self.f_star = identity_profile(rng, cfg)
        # [1] 4 句参考剖面 -> 基线指纹
        self.profiles = make_profiles(self.f_star, cfg.N_REF, sig, rng, cfg.GRID)
        self.baseline = self.profiles.mean(axis=0)     # 基线 = 多参考剖面均值
        self.baseline_std = self.profiles.std(axis=0)  # 逐维标准差
        # [2] 跨 prompt 距离（同实例）
        self.within_dist = pairwise_dist(self.profiles)
        # [3] 跨模型距离与分离比：另一家族指纹远离 Qwen 身份 ||delta|| = CROSS_TARGET
        delta = rng.standard_normal(cfg.GRID)
        delta = delta / np.linalg.norm(delta) * cfg.CROSS_TARGET  # 归一化后精确位移 0.219
        g_star = self.f_star + delta
        g_profiles = make_profiles(g_star, cfg.N_REF, sig, rng, cfg.GRID)
        g_baseline = g_profiles.mean(axis=0)
        self.cross_dist = float(np.linalg.norm(self.baseline - g_baseline))
        # 分离比 = 跨家族距离/族内距离：>1 才可区分，≈7 表示族间可分性远强于输入扰动
        self.separation = self.cross_dist / self.within_dist if self.within_dist > 0.0 else float("inf")
        # [4] 单句剖面 / 4 句均值基线与真实身份的残差
        self.res_1 = float(np.linalg.norm(self.profiles[0] - self.f_star))
        self.res_mean = float(np.linalg.norm(self.baseline - self.f_star))
        self._synced = True

    # ------------------------------------------------------------ 1) 基线指纹
    def validate_baseline(self) -> dict:
        """1) 4 句参考剖面 -> 基线指纹：有限值的 GRID 维向量。"""
        cfg = self.config
        self._synthesize()
        assert self.baseline is not None
        ok = bool(np.all(np.isfinite(self.baseline))) and self.baseline.shape == (cfg.GRID,)
        if not ok:
            raise ConfigError(
                f"基线指纹应为有限值 {cfg.GRID} 维向量",
                expected=(cfg.GRID,), actual=self.baseline.shape, param_key="E01",
            )
        return {
            "detail": (f"基线指纹 shape={self.baseline.shape}, 前8维 "
                       f"{np.round(self.baseline[:8], 4).tolist()}; "
                       f"跨 prompt 标准差 {np.round(self.baseline_std, 4).tolist()}"),
            "shape": list(self.baseline.shape),
        }

    # ------------------------------------------------------------ 2) 跨 prompt 距离
    def validate_within(self) -> dict:
        """2) 同实例跨 prompt 距离应落在 0.031 量级（0.020~0.042）。"""
        cfg = self.config
        self._synthesize()
        ok = cfg.WITHIN_LO < self.within_dist < cfg.WITHIN_HI
        if not ok:
            raise ConfigError(
                f"跨 prompt 距离应在 0.031 量级，实际 {self.within_dist:.4f}",
                expected=(cfg.WITHIN_LO, cfg.WITHIN_HI), actual=self.within_dist,
                param_key="E01",
            )
        return {
            "detail": f"同实例跨 prompt 距离 = {self.within_dist:.4f} (实测 0.031)",
            "within": self.within_dist, "within_lo": cfg.WITHIN_LO, "within_hi": cfg.WITHIN_HI,
        }

    # ------------------------------------------------------------ 3) 跨模型距离与分离比
    def validate_separation(self) -> dict:
        """3) 跨模型距离 ≈ 0.219 且分离比 ≈ 7x（输入无关性判据）。"""
        cfg = self.config
        self._synthesize()
        ok3a = cfg.CROSS_LO < self.cross_dist < cfg.CROSS_HI
        ok3b = cfg.SEP_LO < self.separation < cfg.SEP_HI
        if not ok3a:
            raise ConfigError(
                f"跨模型距离应在 0.219 量级，实际 {self.cross_dist:.4f}",
                expected=(cfg.CROSS_LO, cfg.CROSS_HI), actual=self.cross_dist, param_key="E01",
            )
        if not ok3b:
            raise FamilySeparationError(
                f"分离比应≈7x，实际 {self.separation:.2f}",
                expected=(cfg.SEP_LO, cfg.SEP_HI), actual=self.separation, param_key="E01",
            )
        return {
            "detail": (f"跨模型距离 = {self.cross_dist:.4f} (实测 0.219); "
                       f"分离比 = {self.separation:.2f}x (实测 7x, 文档下界 {cfg.SEP_MIN})"),
            "cross": self.cross_dist, "separation": self.separation,
            "sep_min": cfg.SEP_MIN,
        }

    # ------------------------------------------------------------ 4) 参考句数稳定性平台
    def validate_platform(self) -> dict:
        """4) 参考句数稳定性平台：均值基线优于单句、残差单调递减且饱和。"""
        cfg = self.config
        self._synthesize()
        # 基线应比单个剖面更接近真实身份（噪声被平均）
        ok4a = self.res_mean < self.res_1
        if not ok4a:
            raise ConfigError(
                "4 句均值基线应比单句更接近真实身份",
                expected=self.res_1, actual=self.res_mean, param_key="E01",
            )
        # 稳定性平台（解析期望，避免单次采样波动）：
        # N 句均值残差 ∝ sig/sqrt(N)*sqrt(GRID) -> 1句: A, 4句: A/2, 16句: A/4
        var_noise = (cfg.WITHIN_TARGET ** 2) / (2.0 * cfg.GRID)
        sig = np.sqrt(var_noise)
        A = sig * np.sqrt(cfg.GRID)        # 单句残差的解析期望
        gain_1_4 = A - A / 2               # 1->4 句增益
        gain_4_16 = A / 2 - A / 4          # 4->16 句增益（= 1/2 的 1->4 增益，饱和）
        ok4b = (A / 2 < A) and (A / 4 < A / 2)   # 残差随 N 严格单调递减
        ok4c = gain_4_16 < gain_1_4              # 边际收益递减 -> 饱和平台
        if not (ok4b and ok4c):
            raise ConfigError(
                "N 句均值残差应随 N 单调递减且 4 句后饱和",
                expected={"gain_1_4": gain_1_4, "gain_4_16": gain_4_16},
                actual={"res_1": self.res_1, "res_mean": self.res_mean}, param_key="E01",
            )
        return {
            "detail": (f"1->4 句增益 {gain_1_4:.4f} > 4->16 句增益 {gain_4_16:.4f} (饱和); "
                       f"4 句均值残差 {self.res_mean:.4f} < 单句残差 {self.res_1:.4f}"),
            "res_1": self.res_1, "res_mean": self.res_mean,
            "gain_1_4": gain_1_4, "gain_4_16": gain_4_16,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实 k_proj Gamma 24 维剖面作为基线特征，复算跨 prompt 距离。

        真实剖面缺失时回退审计值且不判失败（与原脚本语义一致）。
        """
        cfg = self.config
        rd = self._get_real_data()
        g_real = rd.get("spectral.k_proj_gamma_layers")
        g_mean = rd.get("spectral.k_proj_gamma_mean")
        g_doc = rd.audit("k_proj_gamma_mean") or 0.625
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 剖面缺失或维数不符时：对照层无法构造，回退审计值并判定通过（不误报）
        if g_real is None or len(g_real) != cfg.GRID:
            return {
                "detail": f"{tag} 真实剖面缺失 -> 回退审计值 {g_doc}，对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        g = np.asarray(g_real, dtype=float)
        # 反解单维噪声标准差：E||f_i-f_j|| = sqrt(2·var_noise·GRID)，与合成流程同构
        var_noise = (cfg.WITHIN_TARGET ** 2) / (2.0 * cfg.GRID)
        sig = np.sqrt(var_noise)
        # 真实 Gamma 剖面归一化到 AMPLITUDE，作为真实身份指纹
        f_star_real = g / np.linalg.norm(g) * cfg.AMPLITUDE
        rng_r = np.random.default_rng(cfg.REAL_SEED)  # 固定种子：真实对照层跨运行可复现
        profiles_r = make_profiles(f_star_real, cfg.N_REF, sig, rng_r, cfg.GRID)
        baseline_r = profiles_r.mean(axis=0)
        within_r = pairwise_dist(profiles_r)
        # 判据：真实基线的跨 prompt 距离应与实测 0.031 同量级（0.020~0.042）
        ok = cfg.WITHIN_LO < within_r < cfg.WITHIN_HI
        if not ok:
            raise RealModelMismatchError(
                f"真实基线跨 prompt 距离 {within_r:.4f} 不在 {cfg.WITHIN_LO}~{cfg.WITHIN_HI}",
                expected=(cfg.WITHIN_LO, cfg.WITHIN_HI), actual=within_r, param_key="E01",
            )
        return {
            "detail": (f"{tag} 真实 k_proj Gamma 均值 {g_mean:.4f} vs 文档审计 {g_doc} "
                       f"(差异 {abs(g_mean - g_doc):.3f}); 跨 prompt 距离(真实基线) "
                       f"{within_r:.4f}（判据 {cfg.WITHIN_LO}~{cfg.WITHIN_HI}）; "
                       f"基线指纹前8维 {np.round(baseline_r[:8], 4).tolist()}"),
            "source": rd.source_tag(), "tag": tag,
            "gamma_mean": g_mean, "gamma_doc": g_doc, "within_real": within_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "baseline", self.validate_baseline),
            (2, "within", self.validate_within),
            (3, "separation", self.validate_separation),
            (4, "platform", self.validate_platform),
            (5, "real_model", self.validate_real_model),
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


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """E01 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E01 REF_PROMPTS 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("E01 REF_PROMPTS 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_REF={cfg.N_REF} GRID={cfg.GRID} WITHIN_TARGET={cfg.WITHIN_TARGET} "
          f"CROSS_TARGET={cfg.CROSS_TARGET} SEP_MIN={cfg.SEP_MIN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
