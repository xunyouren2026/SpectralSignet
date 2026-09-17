# -*- coding: utf-8 -*-
"""A01 model 被测模型路径 / HuggingFace ID — 四层工厂架构验证
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. model_id 字符串解析（家族/变体）+ 加载参数传递（dtype, seed, n_layers）
  2. 同 ID 固定种子可复现（H01 seed=0 语义，两次指纹逐元素相等）
  3. base vs instruct 指纹相关 ≈ 1（SFT 不变性，实测 corr=0.9999）+ 中层锚定
  4. Qwen 系 vs GPT-2 家族分离（跨家族距离 >> 实例内厚度，分离比 >2）
  5. 家族溯源判定正确率（21 样本，实测 100%）
  6. 真实模型实测对照（Qwen2.5-0.5B-Instruct，_real_metrics.json，惰性导入）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  A01Config              —— 配置模型（pydantic 校验；pydantic 缺失时自动
                            dataclass 回退，由 _factory.ConfigFactory.build_model 驱动）
  ConfigFactory          —— 实例化 A01Config（优先级：环境变量 AIQ_A01_<KEY>
                            > YAML config.yaml > _params_data.json > 模型默认值）
  FingerprintSynthesizer —— 逐层 β 剖面合成：synth_beta_profile / sft_perturb /
                            batch_synth（向量化 float32）/ interp_to_grid（LazyInterp）；
                            lru_cache 按 (family, variant, noise, seed) 缓存
                            （复用 _perf.cached_synth）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 11-38（A01 10 阶段流水线）
  _aiq_verdict.py（base/instruct spl_gamma 对比、corr）
  《AI几何指纹插件_参数审计与实验报告.txt》行 34
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（由 _real_model_harness.py 实测，数据存于
            _real_metrics.json；经共享库 _real_data.py 惰性读取）。
  接入点：真实架构 24L / 2KV / 64hd / hidden=896；真实 decode tok/s≈7.20
          （文档声称 8.39）；真实 k_proj Gamma≈0.4695（文档声称 0.625）。
  数据来源标注：真实值输出前缀 [真实实测]；真实数据缺失时前缀 [审计回退]。
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
    ReproducibilityError,
    SFTInvarianceError,
    SynthesisError,
    TraceabilityError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import CachedSynth, LazyInterp, cached_synth, profile_run, synth_batch  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 A01Config（pydantic 优先；dataclass 回退） ----
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

        子类 A01Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class A01Config(_ConfigModelBase):
    """A01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 A01 节点键名一一对应；取值优先级：
    环境变量 AIQ_A01_<KEY> > YAML > _params_data.json > 本模型默认值。
    字段默认值仅作最低优先级兜底（数据文件缺失时），正常由配置层覆盖。
    """

    M: int = 3              # B01: C-子空间主方向数（主文档行 20）
    DTYPE: str = "float32"  # H05 加载精度（from_pretrained 的 dtype 参数）
    N_LAYERS_QWEN: int = 24    # Qwen 层数（层名/深度归一化共用）
    N_LAYERS_GPT2: int = 12    # GPT-2 层数（对比家族，层数不同是家族可分性来源之一）
    SFT_SCALE: float = 1.005   # base→instruct 仅 ~0.5% 尺度微扰（主文档行 470）
    SFT_NOISE: float = 0.0005  # SFT 微扰加噪量级（模拟微调引入的极小随机性）
    CORR_REF: float = 0.9999   # base vs instruct 实测相关（_aiq_verdict.py）
    SEP_RATIO_MIN: float = 2.0     # 家族分离比下界（判据：cross >> within）
    N_TOTAL: int = 21          # 溯源测试集规模（主文档行 555）
    GRID: int = 24             # B03 GRID：统一深度格点（跨层数对齐基准）
    CORR_MIN: float = 0.99     # SFT 不变性相关下界（低于此值判定 SFT 改变指纹）
    SEED: int = 0              # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 A01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """A01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 A01Config。"""

    def build(self) -> A01Config:
        """构建 A01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(A01Config, "A01")


# ---- 第三层：指纹合成器 FingerprintSynthesizer（算法与原脚本完全一致） ----
class FingerprintSynthesizer(_SynthBase):
    """A01 逐层 β 剖面合成器：三层模板 + SFT 微扰 + 批量/惰性插值。

    - synth_beta_profile(family, variant, noise, rng)：三层模板算法（保真）；
    - sft_perturb：base → instruct 尺度微扰（SFT 不变性模拟）；
    - batch_synth：向量化批量合成（_perf.synth_batch，float32 堆叠）；
    - interp_to_grid：经 LazyInterp 惰性插值到统一深度格点。
    """

    def __init__(self, cfg: A01Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)
        # lru_cache 包装：键 = (family, variant, noise, seed)，复用 _perf.cached_synth
        self._cached: CachedSynth = cached_synth(self._synth_by_seed)

    # ---- 种子版入口：缓存/批量的统一哈希单元（rng 由固定种子派生，保证可复现）----
    def _synth_by_seed(self, family: str, variant: str, noise: float, seed: int) -> np.ndarray:
        """由固定种子派生 rng 后调用 synth_beta_profile（缓存键的取值单元）。"""
        return self.synth_beta_profile(family, variant, noise, np.random.default_rng(seed))

    @property
    def cached(self) -> CachedSynth:
        """缓存包装入口：synth.cached(family, variant, noise, seed) 命中 lru_cache。

        注意命名：ValidatorEngine 基类把合成器实例存为 self.synth，
        故缓存包装不叫 synth，避免与合成器实例遮蔽。
        """
        return self._cached

    def synth_beta_profile(
        self,
        family: str,
        variant: str = "base",
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """合成逐层 β 剖面（三层结构，主文档行 245-249；算法与原脚本一致）。

          浅层 0-7   : β 0.2-0.3 （输入编码重塑区）
          中层 8-15  : β 0.5-0.8 （高保真传输区）
          深层 16-23 : β 0.3-0.6 （输出前重组区）
        返回每层 spl_gamma（β 为能量占比，数值裁剪到 [0,1]）。
        """
        if rng is None:
            rng = np.random.default_rng(self._seed)  # 未显式传 rng 时用固定种子，可复现
        cfg = self._cfg
        # 按家族取层数：gpt2 用其自有层数，其余（qwen/unknown）按 Qwen
        n = cfg.N_LAYERS_GPT2 if family == "gpt2" else cfg.N_LAYERS_QWEN
        if n <= 0:
            raise SynthesisError(f"非法的层数: {n}", actual=n)  # 防御：层数非正则无法构造剖面
        prof = np.zeros(n)  # 逐层 β 容器
        if family == "qwen":
            # Qwen 三层模板：中层集中度显著高于浅/深层 → 中层为指纹核心区（B10 anchor 依据）
            prof[0:8] = rng.uniform(0.20, 0.30, 8)
            prof[8:16] = rng.uniform(0.55, 0.75, 8)
            prof[16:24] = rng.uniform(0.30, 0.55, 8)
        else:  # gpt2（或未知家族回退模板）：12 层模板，各段均低于 Qwen 同层段 → 家族可分
            prof[0:4] = rng.uniform(0.10, 0.20, 4)
            prof[4:8] = rng.uniform(0.30, 0.45, 4)
            prof[8:12] = rng.uniform(0.20, 0.35, 4)
        if variant == "instruct":
            prof = prof * cfg.SFT_SCALE  # SFT 微调：仅 ~0.5% 尺度微扰（同架构不改变指纹）
        if noise > 0.0:
            prof = prof + rng.normal(0, noise, n)  # 可选高斯噪声模拟实测涨落
        return np.clip(prof, 0.0, 1.0)  # β 为能量占比，物理上须落在 [0,1]

    def sft_perturb(
        self,
        base_prof: np.ndarray,
        rng: np.random.Generator,
        scale: float | None = None,
        noise: float | None = None,
    ) -> np.ndarray:
        """模拟 SFT（base → Instruct）：共享同一条几何剖面，仅 ~0.5% 尺度微扰
        加极小噪声（实测 corr=0.9999，主文档行 470：同架构 SFT 不改变指纹）。"""
        cfg = self._cfg
        if scale is None:
            scale = cfg.SFT_SCALE
        if noise is None:
            noise = cfg.SFT_NOISE
        if base_prof.size == 0:
            raise SynthesisError("空剖面输入", actual=base_prof.shape)  # 防御：空数组无法微扰
        # base 剖面整体缩放到 scale 倍并叠加高斯噪声，随后裁剪回 [0,1]
        return np.clip(np.asarray(base_prof) * scale + rng.normal(0, noise, base_prof.size), 0.0, 1.0)

    def batch_synth(
        self, params_list: list[tuple[str, str, float, int]], dtype: Any = np.float32
    ) -> np.ndarray:
        """向量化批量合成：复用缓存路径，经 _perf.synth_batch 堆叠为 float32 (N, ...) 数组。"""
        return synth_batch(self._cached, params_list, dtype)

    def interp_to_grid(self, prof: np.ndarray, n_grid: int | None = None) -> np.ndarray:
        """B03 GRID：把不同层数的剖面（如 GPT-2 的 12 层）线性插值到统一深度格点。

        经 LazyInterp 惰性化：构造零成本，resolve() 时才执行 np.interp 并缓存结果。
        """
        cfg = self._cfg
        if n_grid is None:
            n_grid = cfg.GRID
        prof = np.asarray(prof, dtype=float)
        if prof.size == 0:
            raise SynthesisError("空剖面输入", actual=prof.shape)  # 防御：空剖面无法定义插值域
        if n_grid <= 0:
            raise SynthesisError(f"非法的格点数: {n_grid}", actual=n_grid)  # 防御：格点须为正
        # 源采样点：层索引归一化到 [0,1]；目标格点：统一 GRID 个等距深度
        x_src = np.linspace(0, 1, prof.size)
        x_dst = np.linspace(0, 1, n_grid)
        return LazyInterp(x_src, prof, x_dst).resolve()


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def l2_dist(a: Any, b: Any) -> float:
    """欧氏距离 ||a - b||（48 维联合指纹的距离度量子程序）。"""
    # 统一转 float 数组后取二范数，返回 Python 标量供比较
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def parse_model_id(model_id: str, cfg: A01Config) -> dict:
    """模拟 A01 字符串 → 加载参数/家族标签的解析逻辑（与原脚本一致）。

    返回 dict 含 dtype/local_files_only/seed/family/variant/n_layers，
    与真实 from_pretrained 的参数传递一一对应。
    """
    s = model_id.lower()  # 统一小写，前缀匹配大小写无关
    # 家族判定：按前缀区分 Qwen 系 / GPT-2 系 / 未知
    if s.startswith("qwen"):
        family = "qwen"
    elif s.startswith("gpt2") or s.startswith("gpt-2"):
        family = "gpt2"
    else:
        family = "unknown"
    # 变体判定：instruct（SFT 产物）优先，其次 base（预训练基座），其余 plain
    variant = "instruct" if "instruct" in s else ("base" if "base" in s else "plain")
    # 模拟 from_pretrained 的关键参数传递（主文档行 15）
    return {
        "dtype": cfg.DTYPE,               # H05 加载精度
        "local_files_only": True,         # 仅本地加载（审计场景不得联网）
        "seed": cfg.SEED,                 # H01（本脚本用 np.random 模拟）
        "family": family,                 # 家族标签（溯源判定基准）
        "variant": variant,               # 变体标签（SFT 不变性验证基准）
        "n_layers": cfg.N_LAYERS_GPT2 if family == "gpt2" else cfg.N_LAYERS_QWEN,
    }


def verdict(unknown: Any, ref_qwen: Any, ref_gpt2: Any) -> tuple[str, float, float]:
    """家族溯源判定：到 Qwen 参考簇 vs 到 GPT-2 参考簇的距离（主文档行 539-547）。
    返回 (判定结果, d_q, d_g)，距离更近的家族获胜。"""
    d_q = l2_dist(unknown, ref_qwen)  # 未知样本到 Qwen 参考簇中心的距离
    d_g = l2_dist(unknown, ref_gpt2)  # 未知样本到 GPT-2 参考剖面的距离
    return ("Qwen家族" if d_q < d_g else "GPT-2家族", d_q, d_g)  # 近邻判定


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """A01 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（方法内 import 或经 _common.setup_env 注入）。
    """

    def __init__(
        self,
        config: A01Config,
        synth: FingerprintSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self.g_base: np.ndarray | None = None   # 共享中间结果（供后续步骤复用）
        self.g_instruct: np.ndarray | None = None
        self.g_gpt2: np.ndarray | None = None
        self.ref_q: np.ndarray | None = None
        self.ref_g: np.ndarray | None = None

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要，避免拖慢其他路径）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 解析
    def validate_parse(self) -> dict:
        """1) model_id 解析 + 加载参数传递（家族/变体/dtype/seed/层数逐字段断言）。"""
        cfg = self.config
        cfg_q = parse_model_id("Qwen2.5-0.5B-Instruct", cfg)  # 主测 ID：Qwen 系 instruct 变体
        cfg_g = parse_model_id("gpt2-124m", cfg)              # 对比 ID：GPT-2 家族
        ok = (cfg_q["family"] == "qwen" and cfg_q["variant"] == "instruct"
              and cfg_q["dtype"] == cfg.DTYPE and cfg_q["seed"] == cfg.SEED
              and cfg_q["n_layers"] == cfg.N_LAYERS_QWEN
              and cfg_g["family"] == "gpt2" and cfg_g["n_layers"] == cfg.N_LAYERS_GPT2)
        if not ok:
            raise ConfigError(
                "model_id 解析/参数传递不符",
                expected={"family": "qwen", "variant": "instruct", "dtype": cfg.DTYPE,
                          "seed": cfg.SEED, "n_layers_qwen": cfg.N_LAYERS_QWEN,
                          "n_layers_gpt2": cfg.N_LAYERS_GPT2},
                actual={"qwen": cfg_q, "gpt2": cfg_g},
                param_key="A01",
            )
        return {
            "detail": (f"Qwen2.5-0.5B-Instruct -> {cfg_q['family']}/{cfg_q['variant']} "
                       f"dtype={cfg_q['dtype']} seed={cfg_q['seed']} "
                       f"n_layers={cfg_q['n_layers']}; gpt2-124m -> {cfg_g['family']} "
                       f"n_layers={cfg_g['n_layers']}"),
            "qwen": {k: cfg_q[k] for k in ("family", "variant", "dtype", "seed", "n_layers")},
            "gpt2": {k: cfg_g[k] for k in ("family", "variant", "dtype", "seed", "n_layers")},
        }

    # ------------------------------------------------------------ 2) 可复现
    def validate_reproducibility(self) -> dict:
        """2) 可复现性：同 ID 同种子 → 同指纹（H01 seed=0 语义，逐元素相等）。"""
        cfg = self.config
        # 路径一：经 lru_cache（键 = family/variant/noise/seed）
        p1 = self.synth.cached("qwen", "base", 0.0, cfg.SEED)
        # 路径二：独立直算（不经缓存、全新 rng）——真正验证"同种子→同指纹"
        p2 = self.synth.synth_beta_profile("qwen", "base", 0.0, np.random.default_rng(cfg.SEED))
        ok = bool(np.array_equal(p1, p2))  # H01 语义：同种子必须逐元素相等
        if not ok:
            raise ReproducibilityError(
                f"同种子({cfg.SEED})两次合成指纹不一致",
                expected=0.0,
                actual=float(np.max(np.abs(np.asarray(p1, dtype=float)
                                           - np.asarray(p2, dtype=float)))),
                param_key="A01",
            )
        return {"detail": f"seed={cfg.SEED} 两次指纹逐元素相等 = {ok}", "seed": cfg.SEED}

    # ------------------------------------------------------------ 3) SFT 不变性
    def validate_sft_invariance(self) -> dict:
        """3) SFT 不变性（base vs instruct，实测 corr=0.9999）+ 中层锚定。"""
        cfg = self.config
        # 与原脚本一致：共享同一 rng 流（先合成 base，再在其上做 SFT 微扰）
        rng = np.random.default_rng(cfg.SEED)
        g_base = self.synth.synth_beta_profile("qwen", "base", 0.0003, rng)  # base 基线（噪声量级为算法常量）
        g_instruct = self.synth.sft_perturb(g_base, rng)                     # SFT 微扰后的 instruct 剖面
        corr_bi = float(np.corrcoef(g_base, g_instruct)[0, 1])  # 两条剖面 Pearson 相关
        mid_beta = float(g_base[8:16].mean())  # 中层 β 均值（高保真传输区）
        sh_beta = float(g_base[0:8].mean())    # 浅层 β 均值（输入编码重塑区）
        dp_beta = float(g_base[16:24].mean())  # 深层 β 均值（输出前重组区）
        # 共享中间结果：供家族分离/溯源步骤复用（即便本步 FAIL 也先落盘）
        self.g_base, self.g_instruct = g_base, g_instruct
        # SFT 不变性判据：相关须高于 CORR_MIN；中层锚定判据：中层 β 须高于浅/深层
        ok = (corr_bi > cfg.CORR_MIN) and (mid_beta > sh_beta) and (mid_beta > dp_beta)
        if not ok:
            raise SFTInvarianceError(
                f"SFT 不变性不满足: corr={corr_bi:.4f} 需>{cfg.CORR_MIN}, "
                f"中层β={mid_beta:.3f} 须高于浅/深层",
                expected={"corr_min": cfg.CORR_MIN, "mid_gt_sh": True, "mid_gt_dp": True},
                actual={"corr": corr_bi, "mid_beta": mid_beta,
                        "sh_beta": sh_beta, "dp_beta": dp_beta},
                param_key="A01",
            )
        return {
            "detail": (f"corr = {corr_bi:.4f} (实测 {cfg.CORR_REF}, 要求 >{cfg.CORR_MIN}); "
                       f"中层β = {mid_beta:.3f} > 浅层 {sh_beta:.3f} / 深层 {dp_beta:.3f}"),
            "corr": corr_bi, "corr_min": cfg.CORR_MIN, "corr_ref": cfg.CORR_REF,
            "mid_beta": mid_beta, "sh_beta": sh_beta, "dp_beta": dp_beta,
        }

    # ------------------------------------------------------------ 4) 家族分离
    def validate_family_separation(self) -> dict:
        """4) 家族分离（Qwen 系 vs GPT-2，B03 GRID=24 对齐，分离比 > SEP_RATIO_MIN）。"""
        cfg = self.config
        assert self.g_base is not None and self.g_instruct is not None  # 前置步骤已计算
        # 用独立种子 7 生成 GPT-2 剖面，避免与 Qwen 序列相关（原脚本语义）
        g_gpt2 = self.synth.interp_to_grid(self.synth.cached("gpt2", "base", 0.0, 7), cfg.GRID)
        within = l2_dist(self.g_base, self.g_instruct)  # 实例内厚度（同家族两变体距离）
        cross = l2_dist(self.g_base, g_gpt2)            # 跨家族距离（Qwen↔GPT-2）
        ratio = cross / within if within > 0.0 else float("inf")  # 分离比，防御除零
        self.g_gpt2 = g_gpt2
        self.ref_q = (self.g_base + self.g_instruct) / 2.0  # Qwen 参考簇中心（两变体取均值）
        self.ref_g = g_gpt2                                 # GPT-2 参考剖面
        ok = ratio > cfg.SEP_RATIO_MIN  # 家族可分性：跨家族距离须显著大于实例内厚度
        if not ok:
            raise FamilySeparationError(
                f"家族分离比不足: within={within:.4f}, cross={cross:.4f}, "
                f"ratio={ratio:.2f} 需>{cfg.SEP_RATIO_MIN}",
                expected=cfg.SEP_RATIO_MIN, actual=ratio, param_key="A01",
            )
        return {
            "detail": (f"within(base↔instruct)={within:.4f}, cross(Qwen↔GPT-2)={cross:.4f}, "
                       f"分离比={ratio:.2f} (实测 7×, 要求 >{cfg.SEP_RATIO_MIN})"),
            "within": within, "cross": cross, "ratio": ratio,
            "sep_min": cfg.SEP_RATIO_MIN,
        }

    # ------------------------------------------------------------ 5) 溯源
    def validate_traceability(self) -> dict:
        """5) 家族溯源判定：N_TOTAL 样本（70% Qwen / 30% GPT-2），acc=100%。"""
        cfg = self.config
        assert self.ref_q is not None and self.ref_g is not None  # 前置步骤已计算
        # 构造批量参数：每样本独立种子 100+k，噪声 0.002（原脚本语义，测试集可复现）
        qwen_params: list[tuple[str, str, float, int]] = []
        gpt2_params: list[tuple[str, str, float, int]] = []
        for k in range(cfg.N_TOTAL):
            seed = 100 + k
            if k % 10 < 7:  # 70% 来自 Qwen 系（instruct/base 交替）
                qwen_params.append(("qwen", "instruct" if k % 2 else "base", 0.002, seed))
            else:           # 30% 来自 GPT-2 系
                gpt2_params.append(("gpt2", "base", 0.002, seed))
        # 标签顺序与堆叠顺序一致（先 Qwen 批、后 GPT-2 批）
        true_labels = ["Qwen家族"] * len(qwen_params) + ["GPT-2家族"] * len(gpt2_params)
        # 向量化批量合成（float32 堆叠）：Qwen 直接 24 层；GPT-2 插值到 GRID 对齐
        qwen_batch = self.synth.batch_synth(qwen_params)                       # (n_q, 24)
        gpt2_batch = self.synth.batch_synth(gpt2_params)                       # (n_g, 12)
        gpt2_grid = np.stack([self.synth.interp_to_grid(p) for p in gpt2_batch])  # (n_g, 24)
        samples = np.concatenate([qwen_batch, gpt2_grid], axis=0)              # (N, 24)
        # 近邻判定（向量化）：到 Qwen 参考簇 vs 到 GPT-2 参考的距离，更近者获胜
        d_q = np.linalg.norm(np.asarray(samples, dtype=float) - self.ref_q, axis=1)
        d_g = np.linalg.norm(np.asarray(samples, dtype=float) - self.ref_g, axis=1)
        preds = np.where(d_q < d_g, "Qwen家族", "GPT-2家族")
        n_ok = int(np.sum(preds == np.asarray(true_labels)))  # 正确计数
        acc = 100.0 * n_ok / cfg.N_TOTAL                       # 正确率百分比
        ok = n_ok == cfg.N_TOTAL                               # 文档实测 100%，须全部命中
        if not ok:
            raise TraceabilityError(
                f"溯源准确率未达 100%: {n_ok}/{cfg.N_TOTAL}",
                expected=cfg.N_TOTAL, actual=n_ok, param_key="A01",
            )
        return {
            "detail": f"acc = {acc:.1f}% ({n_ok}/{cfg.N_TOTAL}) (实测 100%)",
            "acc_pct": acc, "n_ok": n_ok, "n_total": cfg.N_TOTAL,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型实测对照（Qwen2.5-0.5B-Instruct，_real_metrics.json，惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()  # 惰性导入 / 注入的 _real_data
        arch = rd.get("arch", {}) or {}    # 真实架构字段（层数/KV头/头维/hidden）
        eng = rd.get("engine", {}) or {}   # 真实引擎字段（decode tok/s 等）
        n_layers_real = arch.get("n_layers")  # 真实层数（24）
        n_kv_real = arch.get("n_kv")          # 真实 KV 头数（2）
        hd_real = arch.get("hd")              # 真实每头维度（64）
        hidden_real = arch.get("hidden")      # 真实隐藏维度（896）
        tok_s_real = eng.get("tok_s")         # 真实 decode 速度（tok/s）
        tok_s_doc = rd.audit("tok_s") or float("nan")      # 文档声称值（_real_data 审计表）
        gamma_real = rd.get("spectral.k_proj_gamma_mean")  # 真实 k_proj Gamma 全局均值
        gamma_doc = rd.audit("k_proj_gamma_mean") or float("nan")  # 文档声称值
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        cfg_q = parse_model_id("Qwen2.5-0.5B-Instruct", cfg)
        # 核心一致性断言：真实层数须等于常量 24 且等于解析 cfg_q 的层数；
        # 真实 tok/s 须存在（真实测量已接入）
        ok = (n_layers_real == cfg.N_LAYERS_QWEN == cfg_q["n_layers"]) and (tok_s_real is not None)
        if not ok:
            raise RealModelMismatchError(
                f"真实架构与解析不一致: real n_layers={n_layers_real}, 解析={cfg_q['n_layers']}",
                expected=cfg.N_LAYERS_QWEN, actual=n_layers_real, param_key="A01",
            )
        return {
            "detail": (f"{tag} 真实架构 {n_layers_real}L / {n_kv_real}KV / {hd_real}hd / "
                       f"hidden={hidden_real}（与解析 cfg n_layers={cfg_q['n_layers']} 一致）; "
                       f"decode tok/s={tok_s_real:.2f} vs 文档 {tok_s_doc:.2f}; "
                       f"k_proj Gamma={gamma_real:.4f} vs 文档 {gamma_doc:.3f}"),
            "source": rd.source_tag(), "tag": tag,
            "n_layers_real": n_layers_real, "n_layers_cfg": cfg.N_LAYERS_QWEN,
            "n_kv": n_kv_real, "hd": hd_real, "hidden": hidden_real,
            "tok_s_real": tok_s_real, "tok_s_doc": tok_s_doc,
            "gamma_real": gamma_real, "gamma_doc": gamma_doc,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "parse", self.validate_parse),
            (2, "reproducibility", self.validate_reproducibility),
            (3, "sft_invariance", self.validate_sft_invariance),
            (4, "family_separation", self.validate_family_separation),
            (5, "traceability", self.validate_traceability),
            (6, "real_model", self.validate_real_model),
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
    """A01 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="A01 model 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = FingerprintSynthesizer(cfg)               # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("A01 model 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: M={cfg.M} DTYPE={cfg.DTYPE} N_LAYERS_QWEN={cfg.N_LAYERS_QWEN} "
          f"N_LAYERS_GPT2={cfg.N_LAYERS_GPT2} SFT_SCALE={cfg.SFT_SCALE} "
          f"SFT_NOISE={cfg.SFT_NOISE} CORR_REF={cfg.CORR_REF} "
          f"SEP_RATIO_MIN={cfg.SEP_RATIO_MIN} N_TOTAL={cfg.N_TOTAL} "
          f"GRID={cfg.GRID} CORR_MIN={cfg.CORR_MIN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "a01_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "a01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "a01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
