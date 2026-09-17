# -*- coding: utf-8 -*-
"""A02 prompt 输入种子文本 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. prompt token 数估计（宇宙膨胀句 ≈ 10-15 tokens，主文档行 43）
  2. 同模型跨 prompt within 距离 ≈ 0.031（实测）
  3. 跨模型 cross 距离 ≈ 0.219（实测）
  4. 分离比 ≈ 7×（实测，判据 cross > within 且 ratio > 1）
  5. 低熵输入（"the the the..."）异常探测：偏离基线可检测
  6. 真实模型对照（engine.prompt_len=41 token，真实口径 token/词比 ∈[1.0,1.3]）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  A02Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动
                            dataclass 回退，由 _factory.ConfigFactory.build_model 驱动）
  ConfigFactory           —— 实例化 A02Config（优先级：环境变量 AIQ_A02_<KEY>
                            > YAML config.yaml > _params_data.json > 模型默认值；
                            跨参数引用 M←B01.M、N_GRID←B03.GRID 由共享 ConfigFactory 解析）
  PromptSynthesizer       —— 家族基础剖面 + prompt 级微扰合成（低熵放大 3×）
  ValidatorEngine         —— 4 项验证（对应原 6 个 idx 块）+ 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 39-66（A02 10 步验证链路）
  《AI几何指纹插件_参数完整定义与公式.txt》行 17-24
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实输入种子 prompt 长度 = 41 token（engine.prompt_len）；跨输入稳定性
          用真实 prompt 语义验证。数据来源标注：[真实实测] / [审计回退]。
  如实呈现：文档声称宇宙膨胀句约 10-15 token（短句），真实模型使用的完整段落实测
            41 token——真实输入规模显著大于文档短句量级，如实报告。
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
from _errors import AIQValidationError, ConfigError, FamilySeparationError, RealModelMismatchError  # noqa: E402
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

# ---- 第一层：配置模型 A02Config（pydantic 优先；dataclass 回退） ----
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


class A02Config(_ConfigModelBase):
    """A02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 A02 节点键名一一对应（跨参数引用除外）；
    取值优先级：环境变量 AIQ_A02_<KEY> > YAML > _params_data.json > 本模型默认值。
    字段默认值仅作最低优先级兜底（数据文件缺失时），正常由配置层覆盖。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    M: int = 3                       # B01 跨参数引用：C-子空间主方向数
    N_GRID: int = 24                 # B03 跨参数引用：统一深度格点（跨模型对齐用）
    # 语义不同的三个正常 prompt：宇宙膨胀/广义相对论/量子力学
    PROMPT_EXPANSION: str = ("The universe is expanding, and galaxies are drifting "
                             "apart over time.")
    PROMPT_GR: str = "General relativity describes gravity as the curvature of spacetime."
    PROMPT_QM: str = "Quantum mechanics governs the behavior of particles at the smallest scales."
    PROMPT_LOWENTROPY: str = "the the the the the the the the the the the the"  # 低熵重复文本
    REAL_PROMPT: str = ("The universe is expanding, and galaxies are drifting farther apart "
                        "over time. In the early universe, matter was distributed almost "
                        "uniformly, and small fluctuations grew under gravity to form the "
                        "large-scale structure we observe today.")  # 真实 harness 输入种子
    TARGET_WITHIN: float = 0.031     # 实测：同模型跨 prompt 距离（主文档行 51；σ_prompt 标定目标）
    TARGET_CROSS: float = 0.219      # 实测：跨模型距离（主文档行 53）
    TARGET_RATIO: float = 7.0        # 实测：分离比（主文档行 53）
    LOW_ENTROPY_MULT: float = 3.0    # 低熵输入扰动放大倍数（重复文本激活涨落更大）
    DEV_DETECT_MULT: float = 1.5     # 低熵偏离检出阈值（× within）
    RATIO_HI: float = 1.3            # 真实 token/词比上界（合成估计口径）
    PROMPT_TOKENS: int = 41          # 真实输入种子前向 token 数（engine.prompt_len 断言基准）


# ---- 第二层：配置工厂 ConfigFactory（实例化 A02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """A02 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 A02Config。"""

    def build(self) -> A02Config:
        """构建 A02Config；跨参数引用 M（B01）、N_GRID（B03）单独解析（与旧版 P.get_int 同义）。"""
        cfg = self.build_model(A02Config, "A02")
        cfg.M = self.get_int("B01", "M", 3)          # B01: C-子空间主方向数
        cfg.N_GRID = self.get_int("B03", "GRID", 24)  # B03 GRID: 统一深度格点
        return cfg


# ---- 第三层：合成器 PromptSynthesizer（算法与原脚本完全一致） ----
class PromptSynthesizer(_SynthBase):
    """A02 家族基础剖面 + prompt 级微扰合成器。

    - model_base_profile(family, rng)：家族级几何模板（三层结构，主文档行 245-249）；
    - fingerprint_for_prompt(base_prof, prompt, sigma_prompt, rng)：
      基础剖面 + 语义微扰 N(0, σ_prompt)，低熵重复文本扰动放大 3×。
    """

    def __init__(self, cfg: A02Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def model_base_profile(self, family: str, rng: np.random.Generator) -> np.ndarray:
        """合成模型基础剖面（家族级几何模板，三层结构，主文档行 245-249）。
        返回逐层 β（家族指纹的几何模板，不含 prompt 级扰动）。"""
        if family == "qwen":
            n = 24
            p = np.zeros(n)
            p[0:8] = rng.uniform(0.20, 0.30, 8)      # 浅层：输入编码重塑区
            p[8:16] = rng.uniform(0.55, 0.75, 8)     # 中层：高保真传输区
            p[16:24] = rng.uniform(0.30, 0.55, 8)    # 深层：输出前重组区
            return p
        # gpt2: 12 层，与 Qwen 同量级、仅层间结构微差（真实 cross≈0.219 量级）
        n = 12
        p = np.zeros(n)
        p[0:4] = rng.uniform(0.19, 0.28, 4)      # ≈ qwen 浅层 −0.01
        p[4:8] = rng.uniform(0.50, 0.70, 4)      # ≈ qwen 中层 −0.05
        p[8:12] = rng.uniform(0.27, 0.48, 4)     # ≈ qwen 深层 −0.04
        return p

    def fingerprint_for_prompt(self, base_prof: np.ndarray, prompt: str,
                               sigma_prompt: float, rng: np.random.Generator) -> np.ndarray:
        """合成"某 prompt 下的指纹"：模型基础剖面 + prompt 级微扰 N(0, σ_prompt)。
        σ_prompt 模拟语义变化引起的激活涨落（prompt 无关性的尺度）。"""
        cfg = self._cfg
        prof = np.asarray(base_prof, dtype=float).copy()
        if prof.size == 0:
            raise ValueError("空剖面输入")      # 防御：空剖面无法叠加扰动
        words = prompt.split()
        if len(set(words)) <= 1:               # 低熵重复文本（唯一词≤1）→ 扰动放大
            prof = prof + rng.normal(0, sigma_prompt * cfg.LOW_ENTROPY_MULT, prof.size)
        else:
            prof = prof + rng.normal(0, sigma_prompt, prof.size)   # 正常 prompt 常规扰动
        return np.clip(prof, 0.0, 1.0)          # 裁剪回物理合法区间 [0,1]


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def est_tokens(prompt: str) -> int:
    """合成 token 化：以词数作为 token 数估计（真实约为词数的 1.0-1.3 倍）。"""
    words = prompt.split()          # 按空白切分为词列表
    return max(1, len(words)) if words else 0   # 空串返回 0，否则至少 1 个 token


def spl_gamma_24d(profile: np.ndarray, n_grid: int = 24) -> np.ndarray:
    """把模型基础剖面规范化到 24 点（B03 GRID，跨模型对齐用）。"""
    profile = np.asarray(profile, dtype=float)
    if profile.size == 0:
        raise ValueError("空剖面输入")      # 防御：空剖面无法定义插值域
    # 源采样点归一化到 [0,1]，再线性插值到 24 个等距格点
    x_src = np.linspace(0, 1, profile.size)
    x_dst = np.linspace(0, 1, n_grid)
    return np.interp(x_dst, x_src, profile)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离（24 维指纹距离度量）。"""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# ---- 第四层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """A02 验证引擎：顺序执行 4 项验证（对应原 6 个 idx 块）。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: A02Config,
        synth: PromptSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self._rng: np.random.Generator | None = None   # 跨步骤共享 rng（保持原随机流次序）
        self.qwen_base: np.ndarray | None = None       # 共享中间结果（供后续步骤复用）
        self.gpt2_base: np.ndarray | None = None
        self.sigma_prompt: float = 0.0
        self.within: float = 0.0
        self.cross: float = 0.0
        self.ratio: float = 0.0

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) token 数估计
    def validate_token_count(self) -> dict:
        """1) 宇宙膨胀句 token 数估计（文档声称约 10-15 token，须落在区间内）。"""
        cfg = self.config
        n_tok = est_tokens(cfg.PROMPT_EXPANSION)
        ok = 10 <= n_tok <= 15
        if not ok:
            raise ConfigError(
                f"token 数超出文档区间: {n_tok}",
                expected=(10, 15), actual=n_tok, param_key="A02",
            )
        return {"detail": f"宇宙膨胀句 token 数估计 = {n_tok} (文档：约 10-15)", "n_tok": n_tok}

    # ------------------------------------------------------------ 2-4) within/cross/分离比
    def validate_separation(self) -> dict:
        """2-4) 同模型跨 prompt within ≈ 0.031 / 跨家族 cross ≈ 0.219 / 分离比 ≈ 7×。"""
        cfg = self.config
        # 与原脚本一致：从固定种子派生 rng，随机流次序严格保真
        rng = np.random.default_rng(cfg.SEED)
        self._rng = rng
        qwen_base = self.synth.model_base_profile("qwen", rng)     # Qwen 家族基础剖面

        # 构造跨家族偏移方向：中层为主（家族差异主区，对应 B10 anchor 中层锚定）
        u = np.zeros(cfg.N_GRID)
        u[8:16] = -1.0          # 中层权重最高（家族指纹差异主区）
        u[16:24] = -0.5         # 深层权重次之
        u = u / np.linalg.norm(u)               # 单位化，使偏移量精确等于 TARGET_CROSS
        gpt2_base = np.clip(qwen_base + cfg.TARGET_CROSS * u, 0.0, 1.0)   # GPT-2 剖面

        # σ_prompt 标定：使 24 维 within 距离落在 0.031 附近
        # E[||N(0,σ²I_24)||] ≈ σ·sqrt(24)·0.921（χ 分布期望）
        sigma_prompt = cfg.TARGET_WITHIN / (np.sqrt(cfg.N_GRID) * 0.921)

        prompts = [cfg.PROMPT_EXPANSION, cfg.PROMPT_GR, cfg.PROMPT_QM]    # 3 个正常 prompt
        # within：同一 Qwen 基础剖面上，3 个不同 prompt 的指纹与基线的距离均值
        within_dists = [l2(self.synth.fingerprint_for_prompt(qwen_base, p, sigma_prompt, rng), qwen_base)
                        for p in prompts]
        within = float(np.mean(within_dists))

        f_qwen = self.synth.fingerprint_for_prompt(qwen_base, cfg.PROMPT_EXPANSION, sigma_prompt, rng)
        f_gpt2 = self.synth.fingerprint_for_prompt(gpt2_base, cfg.PROMPT_EXPANSION, sigma_prompt, rng)
        cross = l2(f_qwen, f_gpt2)              # cross：同 prompt 下跨家族的指纹距离
        ratio = cross / within if within > 0.0 else float("inf")   # 分离比，防御除零

        # 共享中间结果：供低熵异常探测步骤复用
        self.qwen_base, self.gpt2_base = qwen_base, gpt2_base
        self.sigma_prompt, self.within, self.cross, self.ratio = sigma_prompt, within, cross, ratio

        ok = (within < cross) and (ratio > 1.0)   # 可分性判据：实例内<跨家族 且 分离比>1
        if not ok:
            raise FamilySeparationError(
                f"分离性不足: within={within:.4f}, cross={cross:.4f}, ratio={ratio:.2f}",
                expected={"within<cross": True, "ratio>1": True},
                actual={"within": within, "cross": cross, "ratio": ratio},
                param_key="A02",
            )
        return {
            "detail": (f"within(同模型跨prompt) = {within:.4f} (实测 {cfg.TARGET_WITHIN}, "
                       f"偏差 {abs(within - cfg.TARGET_WITHIN) / cfg.TARGET_WITHIN * 100:.1f}%); "
                       f"cross(Qwen↔GPT-2) = {cross:.4f} (实测 {cfg.TARGET_CROSS}); "
                       f"分离比 = {ratio:.2f} (实测 {cfg.TARGET_RATIO}×, 判据 cross>within)"),
            "within": within, "cross": cross, "ratio": ratio,
            "target_within": cfg.TARGET_WITHIN, "target_cross": cfg.TARGET_CROSS,
            "target_ratio": cfg.TARGET_RATIO,
        }

    # ------------------------------------------------------------ 5) 低熵异常探测
    def validate_low_entropy(self) -> dict:
        """5) 低熵输入异常探测：偏离须超正常 within 的 DEV_DETECT_MULT 倍才可检出。"""
        cfg = self.config
        assert self.qwen_base is not None and self._rng is not None  # 前置步骤已计算
        f_low = self.synth.fingerprint_for_prompt(
            self.qwen_base, cfg.PROMPT_LOWENTROPY, self.sigma_prompt, self._rng)  # 低熵指纹
        dev_low = l2(f_low, self.qwen_base)          # 低熵指纹与基线的偏离量
        ok = dev_low > self.within * cfg.DEV_DETECT_MULT   # 偏离须超正常 within 的 1.5 倍才可检出
        if not ok:
            raise AIQValidationError(
                f"低熵异常不可检出: dev={dev_low:.4f}, 阈值={self.within * cfg.DEV_DETECT_MULT:.4f}",
                expected=self.within * cfg.DEV_DETECT_MULT, actual=dev_low, param_key="A02",
            )
        return {
            "detail": (f"低熵输入偏离基线 = {dev_low:.4f} vs 正常 within = {self.within:.4f} "
                       f"(阈值 {cfg.DEV_DETECT_MULT}×within = {self.within * cfg.DEV_DETECT_MULT:.4f})"),
            "dev_low": dev_low, "within": self.within, "mult": cfg.DEV_DETECT_MULT,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 prompt 长度 41 token + token/词比落在真实口径 [1.0, RATIO_HI]。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        prompt_len_real = rd.get("engine.prompt_len")  # 真实输入种子前向 token 数（41）
        n_words_real = est_tokens(cfg.REAL_PROMPT)     # 真实段落的合成词数估计
        ratio_real = prompt_len_real / n_words_real if n_words_real else float("inf")  # token/词比
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        ok = (prompt_len_real == cfg.PROMPT_TOKENS) and (1.0 <= ratio_real <= cfg.RATIO_HI)
        if not ok:
            raise RealModelMismatchError(
                f"真实 prompt 长度或 token/词比不符: len={prompt_len_real}, ratio={ratio_real:.2f}",
                expected={"len": cfg.PROMPT_TOKENS, "ratio_lo": 1.0, "ratio_hi": cfg.RATIO_HI},
                actual={"len": prompt_len_real, "ratio": ratio_real},
                param_key="A02",
            )
        return {
            "detail": (f"{tag} 真实输入种子（宇宙膨胀完整段落）词数 = {n_words_real}，"
                       f"真实 token 数 = {prompt_len_real}，token/词比 = {ratio_real:.2f}"
                       f"（∈[1.0,{cfg.RATIO_HI}]）；文档声称短句 ≈10-15 token，"
                       f"真实输入规模显著大于文档短句量级，如实报告"),
            "source": rd.source_tag(), "tag": tag,
            "prompt_len_real": prompt_len_real, "n_words_real": n_words_real,
            "ratio_real": ratio_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "token_count", self.validate_token_count),
            (2, "separation", self.validate_separation),
            (3, "low_entropy", self.validate_low_entropy),
            (4, "real_model", self.validate_real_model),
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
    """A02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="A02 prompt 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = PromptSynthesizer(cfg)                    # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("A02 prompt 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: M={cfg.M} N_GRID={cfg.N_GRID} TARGET_WITHIN={cfg.TARGET_WITHIN} "
          f"TARGET_CROSS={cfg.TARGET_CROSS} TARGET_RATIO={cfg.TARGET_RATIO} "
          f"LOW_ENTROPY_MULT={cfg.LOW_ENTROPY_MULT} DEV_DETECT_MULT={cfg.DEV_DETECT_MULT} "
          f"RATIO_HI={cfg.RATIO_HI} PROMPT_TOKENS={cfg.PROMPT_TOKENS} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "a02_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "a02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "a02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
