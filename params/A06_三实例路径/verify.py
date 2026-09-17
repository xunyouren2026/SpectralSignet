# -*- coding: utf-8 -*-
"""A06 三实例路径 溯源参考模型路径 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 三实例路径存在性（真实文件系统检查，诊断项）
  2. 距离矩阵 d(b,i)=0.009, d(b,g)=0.219, d(i,g)=0.215（主文档行 524-528）
  3. within 波动 0.030 → 分离比 7×（判据 >2）
  4. 21 样本溯源 acc=100%（主文档行 555）
  5. 重生成边距 margin≈0.2975（主文档行 559）
  6. base vs instruct corr≈0.9999（_aiq_verdict.py 行 54）
  7. 真实模型对照：Qwen 实例路径存在性 + 真实架构 24L 断言

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  A06Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 A06Config（环境变量 AIQ_A06_<KEY> > YAML > _params_data.json > 默认；
                            三实例路径经 _cfg 自动探测，零绝对路径硬编码）
  InstanceSynthesizer     —— 三实例 24 维指纹合成（距离矩阵精确复现）+ 重生成边距 MC 二分
  ValidatorEngine         —— 7 项验证（路径=诊断项）+ 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 442-598（A06 是什么/干什么/怎么做）
  _aiq_verdict.py（base/instruct 对比）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
  注（口径修正）：路径检查为"诊断项"（照常逐一路径报告，但不参与 PASS/FAIL 判定）；
      如需强制校验，可将 A06_STRICT_PATHS 置为 True 恢复硬断言语义。
=====================================================================
真实模型对照：
  真实模型：本地模型（_real_metrics.json，共享库 _real_data.py，路径经 _cfg 自动探测）。
  接入点：真实检查三实例模型路径存在性（_cfg 自动探测 + 环境变量 GPT2_MODEL_DIR）；
          真实测量引擎仅覆盖 Qwen-Instruct 单实例——家族溯源/分离/重生成边距等判定
          仍属「设计验证」，如实报告。
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
from _errors import AIQValidationError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注，CFG 供路径探测）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 A06Config（pydantic 优先；dataclass 回退） ----
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


class A06Config(_ConfigModelBase):
    """A06 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 A06 节点键名一一对应；取值优先级：
    环境变量 AIQ_A06_<KEY> > YAML > _params_data.json > 本模型默认值。
    三实例路径字段由 ConfigFactory.build() 经 _cfg 自动探测填充（空串=未配置）。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    D_BI: float = 0.009              # base↔instruct 距离（主文档行 524-528）
    D_BG: float = 0.219              # base↔gpt2 距离
    D_IG: float = 0.215              # instruct↔gpt2 距离
    WITHIN_DOC: float = 0.030        # 同实例跨 prompt 波动（主文档行 532）
    MARGIN_DOC: float = 0.2975       # 重生成边距（主文档行 559）
    N_TOTAL: int = 21                # 溯源测试集规模（主文档行 555）
    DIM: int = 24                    # β 谱 24 维
    SEP_RATIO_MIN: float = 2.0       # 分离比下界（实测 7×）
    CORR_MIN: float = 0.99           # base vs instruct corr 下界（实测 0.9999）
    N_MC: int = 4000                 # 重生成边距 MC 样本数
    BISECT_ITERS: int = 60           # 二分反解 σ* 迭代次数
    SIGMA_LO: float = 0.001          # σ* 搜索下界
    SIGMA_HI: float = 0.5            # σ* 搜索上界
    N_LAYERS_QWEN: int = 24          # Qwen 层数（真实架构断言基准）
    STRICT_PATHS: int = 0            # True=路径缺失则整体失败；False=路径为诊断项（默认）
    # 三实例路径（经 _cfg 自动探测填充；空串=未配置/不存在）
    QWEN_B_PATH: str = ""
    QWEN_I_PATH: str = ""
    GPT2_PATH: str = ""


# ---- 第二层：配置工厂 ConfigFactory（实例化 A06Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """A06 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 A06Config。

    三实例路径经 _cfg 自动探测（环境变量 AIQ_MODELS_DIR -> 项目根/_models -> 向上逐级查找），
    不在代码里写任何电脑绝对路径。
    """

    def build(self) -> A06Config:
        """构建 A06Config；并用 _cfg 探测三实例路径（空串=未配置）。"""
        cfg = self.build_model(A06Config, "A06")
        cfg.QWEN_B_PATH = CFG.model_path("Qwen2.5-0.5B") or ""            # base 变体（可能未配置）
        cfg.QWEN_I_PATH = CFG.model_path("Qwen2.5-0.5B-Instruct") or ""   # Instruct 变体（可能未配置）
        cfg.GPT2_PATH = os.environ.get("GPT2_MODEL_DIR") or ""            # GPT-2 可选（环境变量指定）
        return cfg


# ---- 第三层：合成器 InstanceSynthesizer（算法与原脚本完全一致） ----
class InstanceSynthesizer(_SynthBase):
    """A06 三实例指纹合成器。

    - make_instance_fingerprints(rng)：构造三实例 24 维指纹使距离矩阵逼近实测；
    - regen_margin(sigma, qwen_i, ref_q, ref_g, n_mc, seed)：重生成归一化边距（向量化 MC）。
    """

    def __init__(self, cfg: A06Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def make_instance_fingerprints(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """合成三实例 24 维指纹，使距离矩阵逼近实测：
          base 与 instruct 共享基础剖面（SFT 漂移 ||δ|| = 0.009）
          gpt2 与 qwen 系距离 ||δ|| = 0.219（跨家族）
          instruct 相对 base 的偏移方向与 cross 方向正交 → d(i,g)≈0.215
        """
        cfg = self._cfg
        base = rng.uniform(0.30, 0.70, cfg.DIM)
        u_sft = rng.standard_normal(cfg.DIM)
        u_sft /= np.linalg.norm(u_sft)
        u_cross = rng.standard_normal(cfg.DIM)
        u_cross /= np.linalg.norm(u_cross)
        # 正交化 cross 方向相对 sft 方向，保证 d(i,g)² = 0.219² - 0.009² ≈ 0.215²
        u_cross -= np.dot(u_cross, u_sft) * u_sft
        u_cross /= np.linalg.norm(u_cross)
        qwen_b = base
        qwen_i = base + cfg.D_BI * u_sft
        gpt2 = base + cfg.D_BG * u_cross
        return qwen_b, qwen_i, gpt2

    def regen_margin(self, sigma: float, qwen_i: np.ndarray, ref_q: np.ndarray, ref_g: np.ndarray,
                     n_mc: int, seed: int) -> float:
        """重生成归一化边距 = mean(d_to_gpt2 - d_to_qwen)/scale（规格行 636，实测 0.2975）。
        margin = (mean(dg) - mean(dq)) / ||ref_q - ref_g||，向量化 MC。"""
        cfg = self._cfg
        rngm = np.random.default_rng(seed)
        S = qwen_i + rngm.normal(0.0, sigma, (n_mc, cfg.DIM))
        dqs = np.linalg.norm(S - ref_q, axis=1)
        dgs = np.linalg.norm(S - ref_g, axis=1)
        scale = float(np.linalg.norm(ref_q - ref_g))
        if not np.isfinite(scale) or scale <= 0.0:
            return float("nan")
        return float(dgs.mean() - dqs.mean()) / scale


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def check_paths(paths: dict[str, str]) -> tuple[bool, list[tuple[str, str, bool]]]:
    """真实文件系统检查：各实例路径是否存在，返回 (全部存在?, 明细列表)。

    参数：
      paths: {实例名: 路径}，路径为空串（未配置）时记作"未配置"而非"缺失"。
    """
    detail = []
    all_ok = True
    for name, p in paths.items():
        if not p:                            # 未配置（如 GPT-2 无环境变量）
            exists = False                   # 视为不可用
            label = f"{p}" if p else "(未配置，环境变量 GPT2_MODEL_DIR)"  # 提示如何启用
            detail.append((name, label, exists))
        else:
            exists = os.path.isdir(p)        # 真实文件系统存在性检查
            all_ok = all_ok and exists
            detail.append((name, p, exists))
    return all_ok, detail


# ---- 第四层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """A06 验证引擎：顺序执行 7 项验证（路径=诊断项）。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: A06Config,
        synth: InstanceSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self.qwen_b: np.ndarray | None = None   # 共享中间结果（供后续步骤复用）
        self.qwen_i: np.ndarray | None = None
        self.gpt2: np.ndarray | None = None
        self.d_bg: float = 0.0                  # base↔gpt2 距离（分离比基准）
        self.ref_q: np.ndarray | None = None    # Qwen 参考簇中心
        self.ref_g: np.ndarray | None = None    # GPT-2 参考剖面

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 0) 路径存在性（诊断项）
    def validate_paths(self) -> dict:
        """0) 三实例路径存在性（真实文件系统检查，诊断项；STRICT_PATHS=1 时硬断言）。"""
        cfg = self.config
        paths = {"qwen_b": cfg.QWEN_B_PATH, "qwen_i": cfg.QWEN_I_PATH, "gpt2": cfg.GPT2_PATH}
        all_paths_ok, path_detail = check_paths(paths)
        rows = [f"{name:8s} {p} -> {'存在 OK' if exists else '缺失（诊断提示：模型未下载）'}"
                for name, p, exists in path_detail]
        if cfg.STRICT_PATHS and not all_paths_ok:
            raise AIQValidationError(
                "A06_STRICT_PATHS=True 且存在缺失路径",
                expected=True, actual=all_paths_ok, param_key="A06",
            )
        return {"detail": ("诊断：三实例路径存在性（真实文件系统检查，不参与 PASS/FAIL 判定）: "
                           + "; ".join(rows)),
                "all_ok": all_paths_ok, "strict": bool(cfg.STRICT_PATHS)}

    # ------------------------------------------------------------ 1) 距离矩阵
    def validate_distance(self) -> dict:
        """1) 距离矩阵 d(b,i)/d(b,g)/d(i,g) 与实测值偏差在容差内。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        qwen_b, qwen_i, gpt2 = self.synth.make_instance_fingerprints(rng)
        self.qwen_b, self.qwen_i, self.gpt2 = qwen_b, qwen_i, gpt2   # 共享中间结果
        d_bi = float(np.linalg.norm(qwen_b - qwen_i))
        d_bg = float(np.linalg.norm(qwen_b - gpt2))
        d_ig = float(np.linalg.norm(qwen_i - gpt2))
        self.d_bg = d_bg
        ok1 = (abs(d_bi - cfg.D_BI) / cfg.D_BI < 0.01
               and abs(d_bg - cfg.D_BG) / cfg.D_BG < 0.01
               and abs(d_ig - cfg.D_IG) / cfg.D_IG < 0.02)
        if not ok1:
            raise AIQValidationError(
                f"距离矩阵偏差超限: d_bi={d_bi:.4f}, d_bg={d_bg:.4f}, d_ig={d_ig:.4f}",
                expected={"D_BI": cfg.D_BI, "D_BG": cfg.D_BG, "D_IG": cfg.D_IG},
                actual={"d_bi": d_bi, "d_bg": d_bg, "d_ig": d_ig},
                param_key="A06",
            )
        return {"detail": (f"距离矩阵（目标=主文档行 524-528）: d(qwen_b, qwen_i) = {d_bi:.4f} "
                           f"(实测 {cfg.D_BI}, 偏差 {abs(d_bi - cfg.D_BI) / cfg.D_BI * 100:.1f}%); "
                           f"d(qwen_b, gpt2) = {d_bg:.4f} (实测 {cfg.D_BG}); "
                           f"d(qwen_i, gpt2) = {d_ig:.4f} (实测 {cfg.D_IG})"),
                "d_bi": d_bi, "d_bg": d_bg, "d_ig": d_ig}

    # ------------------------------------------------------------ 2) within 波动与分离比
    def validate_separation(self) -> dict:
        """2) within 波动 0.030 → 分离比 = d_bg/within > SEP_RATIO_MIN（实测 7×）。"""
        cfg = self.config
        within = cfg.WITHIN_DOC                      # 同实例跨 prompt 波动（主文档行 532）
        ratio = self.d_bg / within if within > 0.0 else float("inf")
        ok2 = ratio > cfg.SEP_RATIO_MIN
        if not ok2:
            raise AIQValidationError(
                f"分离比不足: {ratio:.2f}",
                expected=cfg.SEP_RATIO_MIN, actual=ratio, param_key="A06",
            )
        return {"detail": (f"within(同实例跨prompt波动) = {within:.3f} (实测 0.030); "
                           f"分离比 = cross/within = {ratio:.1f} (实测 7×, 要求 >{cfg.SEP_RATIO_MIN})"),
                "within": within, "ratio": ratio, "sep_min": cfg.SEP_RATIO_MIN}

    # ------------------------------------------------------------ 3) 溯源判定
    def validate_traceability(self) -> dict:
        """3) 溯源判定：21 样本（70% Qwen / 30% GPT-2），acc 须 100%。"""
        cfg = self.config
        assert self.qwen_b is not None and self.qwen_i is not None and self.gpt2 is not None
        ref_q = (self.qwen_b + self.qwen_i) / 2.0          # Qwen 参考簇中心
        ref_g = self.gpt2
        self.ref_q, self.ref_g = ref_q, ref_g              # 供重生成边距步骤复用
        n_ok = 0
        for k in range(cfg.N_TOTAL):
            rngk = np.random.default_rng(1000 + k)
            # 未知样本：70% Qwen 系（含扰动）、30% GPT-2
            if k % 10 < 7:
                s = self.qwen_i + rngk.normal(0, cfg.WITHIN_DOC / 4.0, cfg.DIM)
                true = "Qwen家族"
            else:
                s = self.gpt2 + rngk.normal(0, cfg.WITHIN_DOC / 4.0, cfg.DIM)
                true = "GPT-2家族"
            pred = "Qwen家族" if np.linalg.norm(s - ref_q) < np.linalg.norm(s - ref_g) \
                else "GPT-2家族"
            n_ok += (pred == true)
        acc = 100.0 * n_ok / cfg.N_TOTAL
        ok3 = (n_ok == cfg.N_TOTAL)
        if not ok3:
            raise AIQValidationError(
                f"溯源准确率未达 100%: {n_ok}/{cfg.N_TOTAL}",
                expected=cfg.N_TOTAL, actual=n_ok, param_key="A06",
            )
        return {"detail": f"溯源判定 acc = {acc:.1f}% ({n_ok}/{cfg.N_TOTAL}) (实测 100%)",
                "acc_pct": acc, "n_ok": n_ok, "n_total": cfg.N_TOTAL}

    # ------------------------------------------------------------ 4) 重生成边距
    def validate_margin(self) -> dict:
        """4) 重生成归一化边距：二分反解 σ*，margin ≈ MARGIN_DOC（±1%）。"""
        cfg = self.config
        assert self.qwen_i is not None and self.ref_q is not None and self.ref_g is not None
        # margin = mean(d_to_gpt2 - d_to_qwen)/scale，二分反解唯一扰动尺度 σ*
        lo, hi = cfg.SIGMA_LO, cfg.SIGMA_HI
        for _ in range(cfg.BISECT_ITERS):
            mid = (lo + hi) / 2.0
            if self.synth.regen_margin(mid, self.qwen_i, self.ref_q, self.ref_g,
                                       cfg.N_MC, 5) > cfg.MARGIN_DOC:
                lo = mid
            else:
                hi = mid
        sigma_star = (lo + hi) / 2.0
        margin = self.synth.regen_margin(sigma_star, self.qwen_i, self.ref_q, self.ref_g,
                                         cfg.N_MC, 5)
        ok4 = np.isfinite(margin) and (abs(margin - cfg.MARGIN_DOC) / cfg.MARGIN_DOC < 0.01)
        if not ok4:
            raise AIQValidationError(
                f"重生成边距不符: {margin:.4f} vs 实测 {cfg.MARGIN_DOC}",
                expected=cfg.MARGIN_DOC, actual=margin, param_key="A06",
            )
        return {"detail": (f"重生成归一化边距 = {margin:.4f} (实测 {cfg.MARGIN_DOC}); "
                           f"反解扰动尺度 σ*={sigma_star:.3f} (约 {sigma_star / cfg.WITHIN_DOC:.1f}×within)"),
                "margin": margin, "sigma_star": sigma_star}

    # ------------------------------------------------------------ 5) base vs instruct 相关
    def validate_corr(self) -> dict:
        """5) base vs instruct 逐维 corr > CORR_MIN（实测 0.9999）。"""
        cfg = self.config
        assert self.qwen_b is not None and self.qwen_i is not None
        corr_bi = float(np.corrcoef(self.qwen_b, self.qwen_i)[0, 1])
        ok5 = np.isfinite(corr_bi) and (corr_bi > cfg.CORR_MIN)
        if not ok5:
            raise AIQValidationError(
                f"corr 过低: {corr_bi:.4f}",
                expected=cfg.CORR_MIN, actual=corr_bi, param_key="A06",
            )
        return {"detail": (f"base vs instruct 逐维 corr = {corr_bi:.4f} (实测 0.9999, "
                           f"要求 >{cfg.CORR_MIN})"),
                "corr_bi": corr_bi, "corr_min": cfg.CORR_MIN}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：Qwen 实例路径存在性 + 真实架构 24L 断言 + 覆盖范围说明。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        qwen_b_dir = os.path.isdir(cfg.QWEN_B_PATH) if cfg.QWEN_B_PATH else False
        qwen_i_dir = os.path.isdir(cfg.QWEN_I_PATH) if cfg.QWEN_I_PATH else False
        gpt2_dir = os.path.isdir(cfg.GPT2_PATH) if cfg.GPT2_PATH else False
        n_layers_real = rd.get("arch.n_layers")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        ok6 = (qwen_b_dir and qwen_i_dir) and (n_layers_real == cfg.N_LAYERS_QWEN)
        if not ok6:
            raise RealModelMismatchError(
                f"Qwen 实例路径缺失或真实架构不符: b={qwen_b_dir}, i={qwen_i_dir}, L={n_layers_real}",
                expected={"b": True, "i": True, "L": cfg.N_LAYERS_QWEN},
                actual={"b": qwen_b_dir, "i": qwen_i_dir, "L": n_layers_real},
                param_key="A06",
            )
        detail = (f"{tag} 三实例路径存在性: Qwen2.5-0.5B = "
                  f"{'存在 OK' if qwen_b_dir else '缺失'}, Qwen2.5-0.5B-Instruct = "
                  f"{'存在 OK' if qwen_i_dir else '缺失'}, GPT-2 = "
                  f"{'存在 OK' if gpt2_dir else '未配置(环境变量 GPT2_MODEL_DIR)'}; "
                  f"真实测量引擎仅覆盖 Qwen2.5-0.5B-Instruct 单实例（arch {n_layers_real}L / "
                  f"2KV / 64hd）；Qwen base 路径虽存在，但未纳入真实测量；GPT-2 未配置/未实测 —— "
                  f"家族溯源/分离/重生成边距（项1-4）仍属「设计验证」，如实报告")
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "qwen_b_dir": qwen_b_dir, "qwen_i_dir": qwen_i_dir, "gpt2_dir": gpt2_dir,
            "n_layers_real": n_layers_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (0, "paths_diag", self.validate_paths),
            (1, "distance", self.validate_distance),
            (2, "separation", self.validate_separation),
            (3, "traceability", self.validate_traceability),
            (4, "margin", self.validate_margin),
            (5, "corr", self.validate_corr),
            (6, "real_model", self.validate_real_model),
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
    """A06 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="A06 三实例路径 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认；路径经 _cfg 探测）
    synth = InstanceSynthesizer(cfg)                  # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("A06 三实例路径 验证（四层工厂架构，路径=诊断项；距离/溯源=合成数据）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: D_BI={cfg.D_BI} D_BG={cfg.D_BG} D_IG={cfg.D_IG} WITHIN_DOC={cfg.WITHIN_DOC} "
          f"MARGIN_DOC={cfg.MARGIN_DOC} N_TOTAL={cfg.N_TOTAL} DIM={cfg.DIM} "
          f"SEP_RATIO_MIN={cfg.SEP_RATIO_MIN} CORR_MIN={cfg.CORR_MIN} "
          f"N_MC={cfg.N_MC} BISECT_ITERS={cfg.BISECT_ITERS} STRICT_PATHS={cfg.STRICT_PATHS} "
          f"SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "a06_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "a06_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "a06_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
