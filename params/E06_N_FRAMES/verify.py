# -*- coding: utf-8 -*-
"""E06 N_FRAMES DEFF 演示模拟帧数 — 96 帧 DEFF 轨迹三场景状态判定（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 96 帧 DEFF 轨迹模拟：从曲率对 (k1,k2) 计算 DEFF=(|k1|+|k2|)^2/(k1^2+k2^2)
  2. 基准场景：DEFF 锁定 1.58 平台，CV<3%，无告警（|DEFF-1.56|<=0.15）
  3. 公共尺度扰动（ln_sigma=0.8, E07）：DEFF 几乎不变（尺度不变量）
  4. 尖峰场景：每 20 帧结构突变 -> 偏离平台，触发告警

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  E06Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 E06Config（env AIQ_E06_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：工具函数以模块级纯函数提供，共享状态由引擎 _synthesize 惰性计算）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 3888-3995（E06 N_FRAMES）
  源码 _ns_ai_3dphase_plot.py（96 帧 DEFF 演示）
  《参数审计与实验报告.txt》（状态=已用 96 帧）
说明：纯数值合成数据，不加载任何大模型。

真实模型对照：
  经 _real_data 惰性接入真实数据：真实自回归生成 48 token（6 帧 x 8），用真实
  曲率对（RD.phi_pairs()，经 _cfg 相对定位）构建 6 帧 DEFF 帧序列，对照实测存档
  DEFF 平台 1.5920 与 tok/s=7.20，并在公共尺度 lognormal 扰动下验证 DEFF
  尺度不变量。数据来源标注：[真实实测] 或 [审计回退]。
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
    RealModelMismatchError,
    ReproducibilityError,
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

# ---- 第一层：配置模型 E06Config（pydantic 优先；dataclass 回退） ----
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


class E06Config(_ConfigModelBase):
    """E06 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 E06/E07 节点键名对应（N_FRAMES/TOK_PER_FRAME/
    LAM/PLATFORM，LN_SIGMA 联动 E07）；取值优先级：env AIQ_E06_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0                # 固定随机种子：保证可复现（算法逻辑常量）
    N_FRAMES: int = 96           # E06: 模拟帧数（README ①）
    TOK_PER_FRAME: int = 8       # E06: 每帧 token 数（B04，帧对齐）
    LAMBDA: float = 0.3196       # 反解 DEFF=1.58 的曲率比（README ④Step 1）
    PLATFORM: float = 1.56       # T01 平台参考值（README ②公式）
    TOL: float = 0.15            # T01 告警容差（README ②公式，|DEFF-1.56|>0.15 告警）
    LN_SIGMA: float = 0.8        # E07: lognormal 扰动强度（联动，README ⑥）
    NOISE_STD: float = 0.02      # 逐帧激活噪声标准差（原始构造）
    CV_MAX: float = 0.03         # 真实信号判定阈值：CV < 3%（README ③）
    PLATFORM_TARGET: float = 1.58  # DEFF 平台目标值（README ⑤ 实测 1.58±0.01）
    PLATFORM_TOL: float = 0.03     # 两场景均值与平台目标的偏差上限
    SPIKE_PERIOD: int = 20         # 尖峰周期：每 20 帧 k2 放大 2 倍（README ④Step 3）
    SPIKE_FACTOR: float = 2.0      # 尖峰处 k2 放大倍数（λ->0.64, DEFF->1.91）
    MIN_SPIKE_ALERTS: int = 4      # 尖峰场景至少触发的告警数（96/20）
    DRIFT_EPS: float = 1e-9        # 尺度扰动下 DEFF 漂移的判定阈值（解析恒等）
    DEFF_EPS: float = 1e-12        # deff_of 分母下限
    REAL_SEED: int = 55            # 真实对照层固定种子（独立于主验证 SEED）


# ---- 第二层：配置工厂 ConfigFactory（实例化 E06Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """E06 配置工厂：按优先级实例化 E06Config。"""

    def build(self) -> E06Config:
        """构建 E06Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(E06Config, "E06")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def deff_of(k1: np.ndarray, k2: np.ndarray, eps: float) -> np.ndarray:
    """逐点 DEFF = (|k1|+|k2|)^2 / (k1^2 + k2^2) ∈ [1, 2]。

    分母加 DEFF_EPS 防御除零（k1=k2=0 时返回 0）。
    """
    return (np.abs(k1) + np.abs(k2)) ** 2 / (k1 ** 2 + k2 ** 2 + eps)


def stats(traj: np.ndarray, cfg: E06Config) -> tuple:
    """轨迹统计：mean/std/CV/平台告警数。

    返回 (mean, std, cv, n_alert)。空输入防御：空轨迹返回 (0,0,0,0)。
    """
    if len(traj) == 0:                    # 防御空轨迹
        return 0.0, 0.0, 0.0, 0
    m = float(np.mean(traj))
    s = float(np.std(traj))
    cv = s / m if m != 0.0 else 0.0       # 变异系数：相对波动（防御均值 0）
    n_alert = int(np.sum(np.abs(traj - cfg.PLATFORM) > cfg.TOL))  # 越出平台告警带计 1
    return m, s, cv, n_alert


# ---- 第三层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """E06 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 共享三场景轨迹在 _synthesize() 中一次性按原脚本单 rng 流计算。
    """

    def __init__(
        self,
        config: E06Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._synced = False
        # ---- 共享中间结果（与原脚本 main 单 rng 流一致）----
        self.deff_base: np.ndarray | None = None
        self.deff_scale: np.ndarray | None = None
        self.deff_spike: np.ndarray | None = None
        self.rows: dict[str, tuple] = {}
        self.drift_scale = 0.0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _synthesize(self) -> None:
        """一次性生成三场景轨迹（与原脚本 main 的单一 rng 流一致，惰性）。"""
        if self._synced:
            return
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        # ---- 逐帧曲率对生成 ----
        t = np.arange(cfg.N_FRAMES)             # 帧索引 0..95
        k1_clean = np.ones(cfg.N_FRAMES)        # 基准 |k1| = 1
        k2_clean = cfg.LAMBDA * np.ones(cfg.N_FRAMES)      # 基准：DEFF=1.58 平台
        noise1 = rng.normal(0, cfg.NOISE_STD, cfg.N_FRAMES)  # k1 逐帧激活噪声
        noise2 = rng.normal(0, cfg.NOISE_STD, cfg.N_FRAMES)  # k2 逐帧激活噪声
        # 场景 1：基准（小幅激活噪声）
        k1_base, k2_base = k1_clean + noise1, k2_clean + noise2
        self.deff_base = deff_of(k1_base, k2_base, cfg.DEFF_EPS)
        # 场景 2：公共尺度扰动（lognormal 乘性, ln_sigma=0.8, E07）
        s_scale = np.exp(rng.normal(0, cfg.LN_SIGMA, cfg.N_FRAMES))  # 0.45 ~ 2.23
        k1_s, k2_s = k1_base * s_scale, k2_base * s_scale
        self.deff_scale = deff_of(k1_s, k2_s, cfg.DEFF_EPS)
        # 场景 3：尖峰扰动（每 20 帧结构突变：k2 放大 2 倍 -> λ->0.64, DEFF->1.91）
        k1_sp, k2_sp = k1_clean.copy(), k2_clean.copy()
        spike_mask = (t % cfg.SPIKE_PERIOD == 0)   # 尖峰帧位置（0,20,40,60,80）
        k2_sp[spike_mask] *= cfg.SPIKE_FACTOR      # 尖峰处结构突变：k2 放大
        k1_sp, k2_sp = k1_sp + noise1, k2_sp + noise2
        self.deff_spike = deff_of(k1_sp, k2_sp, cfg.DEFF_EPS)
        # 三场景统计表
        self.rows = {
            "基准(无扰动)": stats(self.deff_base, cfg),
            "公共尺度扰动(E07 lnσ=0.8)": stats(self.deff_scale, cfg),
            "尖峰(每20帧结构突变)": stats(self.deff_spike, cfg),
        }
        # 尺度扰动下 DEFF 漂移（应≈0，解析恒等/数值 ~1e-15）
        assert self.deff_scale is not None and self.deff_base is not None
        self.drift_scale = float(np.mean(np.abs(self.deff_scale - self.deff_base)))
        self._synced = True

    # ------------------------------------------------------------ 1) 基准
    def validate_baseline(self) -> dict:
        """1) 基准 96 帧：CV<3%（真实信号）、无告警（平台锁定）。"""
        cfg = self.config
        self._synthesize()
        m_b, s_b, cv_b, na_b = self.rows["基准(无扰动)"]
        ok = (na_b == 0) and (cv_b < cfg.CV_MAX)
        if not ok:
            raise ConfigError(
                f"基准轨迹应平台锁定：告警数 {na_b}，CV={cv_b*100:.2f}%",
                expected={"n_alert": 0, "cv<": cfg.CV_MAX},
                actual={"n_alert": na_b, "cv": cv_b}, param_key="E06",
            )
        return {
            "detail": (f"基准 96 帧: mean={m_b:.4f} std={s_b:.4f} CV={cv_b*100:.2f}%<3%, "
                       f"无告警 -> 真实信号、平台锁定"),
            "mean": m_b, "std": s_b, "cv": cv_b, "n_alert": na_b,
        }

    # ------------------------------------------------------------ 2) 尺度扰动
    def validate_scale(self) -> dict:
        """2) 公共尺度扰动(lnσ=0.8)：DEFF 漂移≈0（尺度不变量）、无告警。"""
        cfg = self.config
        self._synthesize()
        m_sc, s_sc, cv_sc, na_sc = self.rows["公共尺度扰动(E07 lnσ=0.8)"]
        ok = (self.drift_scale < cfg.DRIFT_EPS) and (na_sc == 0)
        if not ok:
            raise ReproducibilityError(
                f"尺度扰动下 DEFF 漂移应≈0 且无告警，漂移 {self.drift_scale:.2e}",
                expected={"drift<": cfg.DRIFT_EPS, "n_alert": 0},
                actual={"drift": self.drift_scale, "n_alert": na_sc}, param_key="E06",
            )
        return {
            "detail": (f"公共尺度扰动(lnσ=0.8): DEFF 漂移 {self.drift_scale:.1e}≈0, "
                       f"无告警（尺度不变量）"),
            "drift": self.drift_scale, "n_alert": na_sc, "mean": m_sc, "cv": cv_sc,
        }

    # ------------------------------------------------------------ 3) 平台均值
    def validate_platform_mean(self) -> dict:
        """3) 基准/尺度场景均值都落在 1.58±0.03 平台附近。"""
        cfg = self.config
        self._synthesize()
        m_b, s_b, cv_b, na_b = self.rows["基准(无扰动)"]
        m_sc, s_sc, cv_sc, na_sc = self.rows["公共尺度扰动(E07 lnσ=0.8)"]
        ok = (abs(m_b - cfg.PLATFORM_TARGET) < cfg.PLATFORM_TOL
              and abs(m_sc - cfg.PLATFORM_TARGET) < cfg.PLATFORM_TOL)
        if not ok:
            raise ConfigError(
                f"DEFF 应锁定在 1.58 平台: 基准 {m_b:.4f}, 尺度 {m_sc:.4f}",
                expected=cfg.PLATFORM_TARGET, actual=(m_b, m_sc), param_key="E06",
            )
        return {
            "detail": f"平台均值对齐: 基准 {m_b:.4f} / 尺度 {m_sc:.4f} ≈ 1.58±0.03",
            "mean_base": m_b, "mean_scale": m_sc,
        }

    # ------------------------------------------------------------ 4) 尖峰
    def validate_spike(self) -> dict:
        """4) 尖峰扰动：触发至少 MIN_SPIKE_ALERTS 次告警并抬升均值。"""
        cfg = self.config
        self._synthesize()
        m_b, s_b, cv_b, na_b = self.rows["基准(无扰动)"]
        m_sp, s_sp, cv_sp, na_sp = self.rows["尖峰(每20帧结构突变)"]
        ok = (na_sp >= cfg.MIN_SPIKE_ALERTS) and (m_sp > m_b)
        if not ok:
            raise ConfigError(
                f"尖峰场景应触发告警并抬升均值: 告警 {na_sp}, mean {m_sp:.4f} vs {m_b:.4f}",
                expected={"n_alert>=": cfg.MIN_SPIKE_ALERTS, "mean>base": True},
                actual={"n_alert": na_sp, "mean": m_sp, "mean_base": m_b}, param_key="E06",
            )
        return {
            "detail": (f"尖峰扰动(每{cfg.SPIKE_PERIOD}帧结构突变): 告警 {na_sp} 次, "
                       f"均值抬升 {m_sp:.4f} > {m_b:.4f}"),
            "n_alert": na_sp, "mean_spike": m_sp, "mean_base": m_b,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实生成 48 token（6 帧 x 8），真实曲率对构建 6 帧 DEFF 帧序列。

        对照实测存档 DEFF 平台 1.5920；公共尺度 lognormal 扰动下验证
        DEFF 尺度不变量（解析恒等）。数据缺失回退且不判失败。
        """
        cfg = self.config
        rd = self._get_real_data()
        ngen = rd.get("engine.ngen_actual")
        tok_s = rd.get("engine.tok_s")
        deff_plat_meas = rd.get("curvature.DEFF_plat")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        pairs = rd.phi_pairs()   # 真实曲率对（经 _cfg.phi_pairs_path() 相对定位）
        # 任一真实数据缺失（含 npy 文件不存在）都回退审计值，不判失败
        if ngen is None or tok_s is None or pairs is None:
            return {
                "detail": (f"{tag} 真实数据缺失（ngen/tok_s/phi_pairs_all.npy）"
                           f" -> 对照层跳过（不判失败）"),
                "skipped": True, "source": tag,
            }
        k1, k2 = pairs[:, 0], pairs[:, 1]
        frames = ngen // cfg.TOK_PER_FRAME          # 帧数 = token/每帧 token 数
        ok_frame = (ngen % cfg.TOK_PER_FRAME == 0) and frames == 6   # 帧对齐校验
        per = len(k1) // frames                     # 每帧曲率点数
        k1f = k1[:per * frames].reshape(frames, per)  # 按帧切分 k1
        k2f = k2[:per * frames].reshape(frames, per)  # 按帧切分 k2
        deff_frames = np.mean(deff_of(k1f, k2f, cfg.DEFF_EPS), axis=1)  # 每帧 DEFF 取均值
        m = float(np.mean(deff_frames))
        cv = float(np.std(deff_frames) / m) if m != 0.0 else 0.0
        na = int(np.sum(np.abs(deff_frames - cfg.PLATFORM) > cfg.TOL))
        ok1 = (m - deff_plat_meas) < 1e-3 and cv < cfg.CV_MAX  # 平台对齐且 CV<3%
        rng_r = np.random.default_rng(cfg.REAL_SEED)   # 固定种子：扰动可复现
        # lognormal 公共尺度因子：整个激活流形被整体缩放
        s = np.exp(rng_r.normal(0, cfg.LN_SIGMA, frames))
        deff_pert = np.mean(deff_of(k1f * s[:, None], k2f * s[:, None], cfg.DEFF_EPS), axis=1)
        drift = float(np.mean(np.abs(deff_pert - deff_frames)))  # 尺度扰动漂移
        ok2 = drift < cfg.DRIFT_EPS              # DEFF 尺度不变量（解析恒等）
        if not (ok_frame and ok1 and ok2):
            raise RealModelMismatchError(
                f"真实 DEFF 帧序列不符: 帧对齐={ok_frame}, 平台均值 {m:.4f} "
                f"(需≈{deff_plat_meas:.4f}), CV={cv*100:.2f}%, 尺度漂移 {drift:.1e}",
                expected={"frame_align": True, "plat": deff_plat_meas,
                          "cv<": cfg.CV_MAX, "drift<": cfg.DRIFT_EPS},
                actual={"m": m, "cv": cv, "drift": drift, "n_alert": na},
                param_key="E06",
            )
        return {
            "detail": (f"{tag} 生成 {ngen} token = {frames} 帧 x 8（帧对齐）; "
                       f"6 帧 DEFF 平台均值 {m:.4f} vs 实测存档 {deff_plat_meas:.4f} "
                       f"(CV={cv*100:.2f}%<3%); 平台告警数 {na}; "
                       f"lognormal 公共尺度扰动(lnσ={cfg.LN_SIGMA}) DEFF 漂移 {drift:.1e}≈0 "
                       f"(尺度不变量); tok/s = {tok_s:.2f}"),
            "source": rd.source_tag(), "tag": tag,
            "ngen": ngen, "tok_s": tok_s, "deff_plat_meas": deff_plat_meas,
            "m": m, "cv": cv, "n_alert": na, "drift": drift,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "baseline", self.validate_baseline),
            (2, "scale", self.validate_scale),
            (3, "platform_mean", self.validate_platform_mean),
            (4, "spike", self.validate_spike),
            (5, "real_model", self.validate_real_model),
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
    """E06 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="E06 N_FRAMES 四层工厂验证")
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

    print("=" * 76)
    print("E06 N_FRAMES 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_FRAMES={cfg.N_FRAMES} TOK_PER_FRAME={cfg.TOK_PER_FRAME} "
          f"LAMBDA={cfg.LAMBDA} PLATFORM={cfg.PLATFORM}±{cfg.TOL} LN_SIGMA={cfg.LN_SIGMA}")
    print("=" * 76)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "e06_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "e06_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "e06_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
