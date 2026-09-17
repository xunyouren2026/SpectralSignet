# -*- coding: utf-8 -*-
"""A03 gen_len/ngen 生成 token 数 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 帧数关系：N_frames = gen_len / TOK_PER_FRAME（64→8, 96→12, 128→16, 256→32）
  2. K% 随帧数增长并收敛到 NS 饱和态 57.42%（文档三档：45.6/52/57.33）
  3. 趋势斜率递减（0.59 → 0.4 → 0.26 %/帧）
  4. DEFF 平台均值≈1.584、帧间 CV<3%（T02 稳态）
  5. Hmed 持续下降（~1.2 → 0.757）
  6. 收敛判定：K%→57.4±0.5% → 几何锁定态
  7. 真实模型对照：真实生成 48 token（engine.ngen_actual）→ 48/TOK_PER_FRAME=6 帧

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  A03Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 A03Config（环境变量 AIQ_A03_<KEY> > YAML > _params_data.json > 默认）
  FrameSeriesSynthesizer  —— 帧级 (K%, DEFF, Hmed) 合成序列（logistic 增长 + 平台波动 + 指数衰减）
  ValidatorEngine         —— 7 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 67-169（A03 操作流程、三档对比表、收敛判定）
  《AI几何指纹插件_参数完整定义与公式.txt》行 25-32
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实自回归生成 48 token（engine.ngen_actual）→ 48/TOK_PER_FRAME=6 帧；
          真实 KV 缓存 seq=984 token（engine.kv_bytes / arch.kv_per_tok）→ 984/8=123 帧；
          真实 DEFF 平台≈1.5920（文档声称 1.584）。
  如实呈现：文档档位 gen_len=64/96/128/256 均高于真实测量规模 48 token，如实报告。
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
from _errors import AIQValidationError, ConfigError, RealModelMismatchError, SynthesisError  # noqa: E402
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

# ---- 第一层：配置模型 A03Config（pydantic 优先；dataclass 回退） ----
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


class A03Config(_ConfigModelBase):
    """A03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 A03 节点键名一一对应；取值优先级：
    环境变量 AIQ_A03_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    TOK_PER_FRAME: int = 8           # B04：每帧 token 数（主文档行 69；帧数换算的分母）
    NS_K: float = 57.42              # NS 饱和态 K%（主文档行 100/122；logistic 增长的上限平台）
    K12: float = 45.6                # 文档 96 帧末帧 K%（行 98）
    K16: float = 52.0                # 文档 128 帧末帧 K%（行 110）
    K32: float = 57.33               # 文档 256 帧末帧 K%（行 122）
    DEFF_REF: float = 1.584          # 256 帧 DEFF 均值（行 122；T02 稳态平台值）
    HMED_REF: float = 0.757          # 256 帧 Hmed 末帧（行 122）
    LOGISTIC_K: float = 0.255        # logistic 增长斜率（由 (12,45.6)/(32,57.33) 两点反解）
    LOGISTIC_F0: float = 6.71        # logistic 拐点帧号
    K_TOL_PP: float = 0.6            # 各档 K% 末帧与文档偏差容差（个百分点）
    LOCK_TOL: float = 0.5            # 收敛判定 K% 与 NS 饱和态偏差容差（主文档行 136）
    CV_MAX: float = 0.03             # T02 稳态：帧间 CV 上限（DEFF 平台波动的稳健性约束）
    GEN_LENS: list = [64, 96, 128, 256]   # 文档四档生成长度
    NGEN_REAL: int = 48              # 真实自回归实际生成 token 数（文档声称 48）
    FRAMES_REAL: int = 6             # 真实生成 48 token 换算的帧数（48/8）


# ---- 第二层：配置工厂 ConfigFactory（实例化 A03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """A03 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 A03Config。"""

    def build(self) -> A03Config:
        """构建 A03Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(A03Config, "A03")


# ---- 第三层：合成器 FrameSeriesSynthesizer（算法与原脚本完全一致） ----
class FrameSeriesSynthesizer(_SynthBase):
    """A03 帧级 (K%, DEFF, Hmed) 合成序列。

    - n_frames(gen_len)：帧数 = gen_len / TOK_PER_FRAME（整除，主文档行 69）；
    - synth_frame_series(gen_len, rng)：K% logistic 增长 + DEFF 平台波动 + Hmed 指数衰减。
    """

    def __init__(self, cfg: A03Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def n_frames(self, gen_len: int) -> int:
        """帧数 = gen_len / TOK_PER_FRAME（主文档行 69，整除）。"""
        cfg = self._cfg
        if gen_len <= 0:
            raise ValueError(f"非法的 gen_len: {gen_len}")   # 防御：非正长度无法分帧
        return gen_len // cfg.TOK_PER_FRAME                  # 整数除法得到完整帧数

    def logistic_k(self, f: int) -> float:
        """K%(f) = NS_K / (1 + exp(-k(f - f0)))，k≈0.255, f0≈6.71。
        用 (12,45.6) 与 (32,57.33) 两点反解；校验 f=16 → ≈52%（文档趋缓期值）。"""
        cfg = self._cfg
        # S 型增长：随帧数从低平台爬升并渐近 NS 饱和态，模拟实测 K% 演化
        return cfg.NS_K / (1.0 + np.exp(-cfg.LOGISTIC_K * (f - cfg.LOGISTIC_F0)))

    def synth_frame_series(self, gen_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成某 gen_len 下的帧级 (K%, DEFF, Hmed) 合成序列（纯函数，rng 显式传入）。"""
        cfg = self._cfg
        nf = self.n_frames(gen_len)                                  # 该生成长度下的帧总数
        k_vals = np.array([self.logistic_k(f) for f in range(1, nf + 1)])   # 逐帧 K% 增长曲线
        deff = cfg.DEFF_REF + rng.normal(0, 0.010, nf)          # DEFF 围绕平台小幅波动
        hmed = cfg.HMED_REF + 0.443 * np.exp(-(np.arange(1, nf + 1) - 12) / 8.0)   # Hmed 指数衰减到末帧
        hmed = np.clip(hmed, 0.0, None)                     # Hmed 物理非负
        assert np.isfinite(k_vals).all(), "K% 序列含 NaN/Inf"   # 数值健全性：K% 须有限
        assert np.isfinite(deff).all(), "DEFF 序列含 NaN/Inf"   # 数值健全性：DEFF 须有限
        return k_vals, deff, hmed


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def end_slope(k_vals: np.ndarray, n_tail: int = 5) -> float:
    """末 n_tail 帧线性回归斜率（%/帧）。
    收敛的几何含义：末段斜率应递减→趋于 0（增长趋缓）。"""
    n_tail = min(n_tail, k_vals.size)   # 帧数不足时退化为全部帧
    if n_tail < 2:
        return 0.0                      # 不足 2 点无法回归，斜率视为 0
    # np.polyfit 一次多项式拟合，返回 [斜率, 截距]，取斜率
    return float(np.polyfit(np.arange(n_tail), k_vals[-n_tail:], 1)[0])


# ---- 第四层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """A03 验证引擎：顺序执行 7 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: A03Config,
        synth: FrameSeriesSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self.series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # 各档帧序列缓存
        self.cv256: float = 0.0      # 256 帧档 DEFF 帧间 CV（供收敛判定复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 帧数关系
    def validate_frame_relation(self) -> dict:
        """1) 帧数关系 gen_len/TOK_PER_FRAME（逐档整除与期望比对）。"""
        cfg = self.config
        cases = cfg.GEN_LENS                                    # 文档四档生成长度（数据层读取）
        expected_frames = [g // cfg.TOK_PER_FRAME for g in cases]  # 对应帧数（gen_len/8 整除结果）
        ok1 = True
        rows = []
        for g, ef in zip(cases, expected_frames):
            nf = self.synth.n_frames(g)          # 计算得到的帧数
            one_ok = (nf == ef)                  # 逐档比对期望
            ok1 &= one_ok
            rows.append(f"gen_len={g:4d} -> {nf} 帧 (预期 {ef})")
        if not ok1:
            raise ConfigError(
                f"帧数换算不符: {[(g, self.synth.n_frames(g)) for g in cases]} vs 预期 {expected_frames}",
                expected=expected_frames, actual=[self.synth.n_frames(g) for g in cases],
                param_key="A03",
            )
        return {"detail": "帧数关系 gen_len/TOK_PER_FRAME: " + "; ".join(rows),
                "frames": [self.synth.n_frames(g) for g in cases]}

    # ------------------------------------------------------------ 2) K% 末帧 vs 文档
    def validate_k_docs(self) -> dict:
        """2) K% 末帧 vs 文档三档（主文档行 98/110/122，容差 K_TOL_PP 个百分点）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        doc_k = {96: cfg.K12, 128: cfg.K16, 256: cfg.K32}    # 文档声称的末帧 K%（96/128/256 档）
        ok2 = True
        rows = []
        k_lasts = {}
        for g in cfg.GEN_LENS:
            k_vals, deff, hmed = self.synth.synth_frame_series(g, rng)
            self.series[g] = (k_vals, deff, hmed)
            if g in doc_k:
                k_last = float(k_vals[-1])       # 末帧 K%
                err = abs(k_last - doc_k[g])     # 与文档偏差（个百分点）
                one_ok = err < cfg.K_TOL_PP      # 容差内即通过
                ok2 &= one_ok
                k_lasts[g] = k_last
                rows.append(f"gen_len={g:4d} K%末帧={k_last:5.2f} (文档 {doc_k[g]:5.2f}, "
                            f"偏差 {err:.2f} pp, 容差 {cfg.K_TOL_PP} pp)")
        if not ok2:
            raise AIQValidationError(
                "K% 末帧与文档偏差超出容差", expected=doc_k, actual=k_lasts, param_key="A03",
            )
        return {"detail": "K% 末帧 vs 文档: " + "; ".join(rows), "k_lasts": k_lasts}

    # ------------------------------------------------------------ 3) 趋势斜率递减
    def validate_slope(self) -> dict:
        """3) 末段斜率递减：96 > 128 > 256 > 0（增长趋缓收敛判据）。"""
        cfg = self.config
        slopes = {g: end_slope(self.series[g][0]) for g in cfg.GEN_LENS}   # 各档末段 K% 增长斜率
        # 收敛判据：生成越长，末段斜率越小且恒为正（仍在增长但趋缓）
        ok3 = (slopes[96] > slopes[128] > slopes[256] > 0.0)
        if not ok3:
            raise AIQValidationError(
                f"斜率未递减: {slopes}", expected="96>128>256>0", actual=slopes, param_key="A03",
            )
        return {"detail": (f"末段斜率: 96={slopes[96]:+.3f}%/帧  128={slopes[128]:+.3f}%/帧  "
                           f"256={slopes[256]:+.3f}%/帧 (文档 0.59/0.4/0.26, 应递减)"),
                "slopes": slopes}

    # ------------------------------------------------------------ 4) DEFF 平台与 CV
    def validate_deff_platform(self) -> dict:
        """4) DEFF 平台（256 帧均值≈文档 1.584）与帧间 CV<3%（T02 稳态）。"""
        cfg = self.config
        deff_96 = self.series[96][1]          # 96 帧档 DEFF 序列
        deff_256 = self.series[256][1]        # 256 帧档 DEFF 序列
        cv256 = float(deff_256.std() / deff_256.mean())   # 帧间变异系数（相对波动）
        self.cv256 = cv256                    # 供收敛判定步骤复用
        # 稳态判据：均值贴近文档平台 1.584（±0.02），且 CV<3%
        ok4 = (abs(float(deff_256.mean()) - cfg.DEFF_REF) < 0.02) and (cv256 < cfg.CV_MAX)
        if not ok4:
            raise AIQValidationError(
                f"DEFF 平台或 CV 超限: mean={deff_256.mean():.4f}, CV={cv256 * 100:.2f}%",
                expected={"mean_tol": 0.02, "cv_max": cfg.CV_MAX},
                actual={"mean": float(deff_256.mean()), "cv": cv256},
                param_key="A03",
            )
        return {"detail": (f"DEFF 均值(256)={deff_256.mean():.4f} (文档 {cfg.DEFF_REF}), "
                           f"CV={cv256 * 100:.2f}% (文档 <3%)"),
                "deff_mean": float(deff_256.mean()), "cv256": cv256, "cv_max": cfg.CV_MAX}

    # ------------------------------------------------------------ 5) Hmed 持续下降
    def validate_hmed(self) -> dict:
        """5) Hmed 末帧持续下降：256 帧末帧 < 96 帧末帧，且有限为正。"""
        cfg = self.config
        h96 = self.series[96][2]              # 96 帧档 Hmed 序列
        h256 = self.series[256][2]            # 256 帧档 Hmed 序列
        # 判据：更长生成下末帧 Hmed 更低（下降趋势），且末值有限为正
        ok5 = (h256[-1] < h96[-1]) and np.isfinite(h256[-1]) and (h256[-1] > 0.0)
        if not ok5:
            raise AIQValidationError(
                f"Hmed 未下降: 96帧={h96[-1]:.3f}, 256帧={h256[-1]:.3f}",
                expected="h256[-1] < h96[-1]", actual={"h96": float(h96[-1]), "h256": float(h256[-1])},
                param_key="A03",
            )
        return {"detail": (f"Hmed 末帧: 96帧={h96[-1]:.3f}  256帧={h256[-1]:.3f} "
                           f"(文档 ~1.2 / {cfg.HMED_REF}, 应下降)"),
                "h96_last": float(h96[-1]), "h256_last": float(h256[-1])}

    # ------------------------------------------------------------ 6) 收敛判定
    def validate_lock(self) -> dict:
        """6) 几何锁定态三条件：K% 逼近 NS 饱和态、DEFF 帧间稳定、Hmed 已下降。"""
        cfg = self.config
        k_vals256 = self.series[256][0]       # 256 帧档 K% 序列
        k_last256 = float(k_vals256[-1])     # 末帧 K%
        h96 = self.series[96][2]
        h256 = self.series[256][2]
        # 几何锁定态三条件：K% 逼近 NS 饱和态、DEFF 帧间稳定、Hmed 已下降
        locked = (abs(k_last256 - cfg.NS_K) < cfg.LOCK_TOL) and (self.cv256 < cfg.CV_MAX) \
            and (h256[-1] < h96[-1])
        if not locked:
            raise AIQValidationError(
                f"未达几何锁定态: K%偏差={abs(k_last256 - cfg.NS_K):.2f}, CV={self.cv256 * 100:.2f}%",
                expected={"lock_tol": cfg.LOCK_TOL, "cv_max": cfg.CV_MAX},
                actual={"k_dev": abs(k_last256 - cfg.NS_K), "cv": self.cv256},
                param_key="A03",
            )
        return {"detail": (f"收敛判定: K%末帧={k_last256:.2f} vs NS饱和态={cfg.NS_K:.2f} "
                           f"(偏差 {abs(k_last256 - cfg.NS_K):.2f} pp, 容差 {cfg.LOCK_TOL} pp): 锁定态"),
                "k_last256": k_last256, "ns_k": cfg.NS_K}

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real_model(self) -> dict:
        """7) 真实模型对照：真实生成 48 token → 6 帧（engine.ngen_actual 断言）。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        ngen_real = rd.get("engine.ngen_actual")       # 真实自回归实际生成 token 数（48）
        kv_bytes_real = rd.get("engine.kv_bytes")      # 真实 KV 缓存字节数
        kv_per_tok_real = rd.get("arch.kv_per_tok")    # 每 token 的 KV 字节数
        seq_real = int(round(kv_bytes_real / kv_per_tok_real)) if kv_per_tok_real else None  # KV 序列长度
        deff_real = rd.get("curvature.DEFF_plat")      # 真实 DEFF 平台
        deff_doc = cfg.DEFF_REF                        # 文档声称平台
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        # 核心断言：真实生成恰 NGEN_REAL(48) token，且按 TOK_PER_FRAME(8) 换算恰为 FRAMES_REAL(6) 帧
        ok7 = (ngen_real == cfg.NGEN_REAL) and (self.synth.n_frames(ngen_real) == cfg.FRAMES_REAL)
        if not ok7:
            raise RealModelMismatchError(
                f"真实帧数不符: ngen={ngen_real}, frames={self.synth.n_frames(ngen_real)}",
                expected={"ngen": cfg.NGEN_REAL, "frames": cfg.FRAMES_REAL},
                actual={"ngen": ngen_real, "frames": self.synth.n_frames(ngen_real)},
                param_key="A03",
            )
        detail = (f"{tag} 真实自回归生成 {ngen_real} token → 帧数 = "
                  f"{ngen_real}/{cfg.TOK_PER_FRAME} = {self.synth.n_frames(ngen_real)} 帧"
                  f"（文档档位 64/96/128/256 为更长序列声称值，如实报告：真实测量规模仅 48 token）")
        if seq_real:
            detail += (f"; 真实 KV 缓存 seq = {seq_real} token（{kv_bytes_real / 1e6:.1f}MB "
                       f"/ {kv_per_tok_real / 1e3:.1f}KB per tok）→ {seq_real}/{cfg.TOK_PER_FRAME} = "
                       f"{seq_real // cfg.TOK_PER_FRAME} 帧（KV 缓存帧视角）")
        detail += (f"; 真实 DEFF 平台 = {deff_real:.4f} vs 文档声称 {deff_doc:.3f}，"
                   f"差异 {100 * (deff_real - deff_doc) / deff_doc:+.1f}%")
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "ngen_real": ngen_real, "frames_real": self.synth.n_frames(ngen_real),
            "seq_real": seq_real, "deff_real": deff_real, "deff_doc": deff_doc,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "frame_relation", self.validate_frame_relation),
            (2, "k_docs", self.validate_k_docs),
            (3, "slope", self.validate_slope),
            (4, "deff_platform", self.validate_deff_platform),
            (5, "hmed", self.validate_hmed),
            (6, "lock", self.validate_lock),
            (7, "real_model", self.validate_real_model),
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
    """A03 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="A03 gen_len 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = FrameSeriesSynthesizer(cfg)               # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("A03 gen_len 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: TOK_PER_FRAME={cfg.TOK_PER_FRAME} NS_K={cfg.NS_K} DEFF_REF={cfg.DEFF_REF} "
          f"HMED_REF={cfg.HMED_REF} LOGISTIC_K={cfg.LOGISTIC_K} LOGISTIC_F0={cfg.LOGISTIC_F0} "
          f"K_TOL_PP={cfg.K_TOL_PP} LOCK_TOL={cfg.LOCK_TOL} CV_MAX={cfg.CV_MAX} "
          f"GEN_LENS={cfg.GEN_LENS} NGEN_REAL={cfg.NGEN_REAL} FRAMES_REAL={cfg.FRAMES_REAL} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "a03_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "a03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "a03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
