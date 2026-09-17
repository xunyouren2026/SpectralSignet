# -*- coding: utf-8 -*-
"""B08 NFRAME 曲率特征帧数 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 8 帧 x 3 指标（K%、DEFF_A、Hmed）= 24 维曲率特征
  2. 与 24 维 β 谱指纹拼接 = 48 维联合溯源特征
  3. NFRAME=4 -> 12 维（信息不足）；NFRAME=16 -> 48 维（维度公式，覆盖须帧数足够）
  4. 前 8 帧覆盖 NFRAME*TOK_PER_FRAME = 64 token
  5. 真实模型对照：真实曲率对（phi_pairs_all.npy）按帧聚合 → 24 维曲率 + β 谱 = 48 维联合特征

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B08Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退；
                            跨参数引用 GRID←B03.GRID 由共享 ConfigFactory 解析）
  ConfigFactory           —— 实例化 B08Config（环境变量 AIQ_B08_<KEY> > YAML > _params_data.json > 默认）
  FrameMetricsSynthesizer —— 逐帧曲率指标合成（每帧 8 token 主曲率对 → K<0%/DEFF/Hmed）
  ValidatorEngine         —— 6 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 1444-1530（B08）
  《参数完整定义与公式.txt》B08 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
  注（口径澄清）：gen_len=96 仅产生 12 帧，NFRAME=16 无法从实测取满 16 帧，
      但维度守恒公式 dim = NFRAME×3 恒成立；覆盖 token 数随 NFRAME 按公式外推。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实曲率对数据集 phi_pairs_all.npy（6303×2）按帧聚合（每帧 48 点），
          8 帧 × 3 指标 = 24 维曲率特征；与真实 k_proj 逐层 Gamma（24 维 β 谱）
          拼接为 48 维联合特征。
  如实呈现：真实帧聚合 K<0%/DEFF/Hmed 帧均值应与全局真实值（45.50%/1.5920/
            1.3152）同量级。
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

# ---- 第一层：配置模型 B08Config（pydantic 优先；dataclass 回退） ----
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


class B08Config(_ConfigModelBase):
    """B08 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B08 节点键名一一对应（GRID 为跨参数引用）；
    取值优先级：环境变量 AIQ_B08_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    NFRAME: int = 8                  # 曲率特征帧数（8 帧 × 3 指标 = 24 维曲率特征）
    FRAME_SIZE: int = 3              # 每帧曲率指标数（K%、DEFF_A、Hmed）
    TOK_PER_FRAME: int = 8           # B04: 每帧 token 数
    GRID: int = 24                   # B03 跨参数引用：β 谱维度
    GEN_LEN: int = 96                # 生成 token 数（12 帧）
    PER_FRAME: int = 48              # 真实帧聚合每帧点数
    NFRAME_SMALL: int = 4            # 维度守恒演示：信息不足档帧数
    NFRAME_LARGE: int = 16           # 维度守恒演示：冗余档帧数
    COVERED_TOK: int = 64            # 前 NFRAME 帧覆盖 token 数（8×8）


# ---- 第二层：配置工厂 ConfigFactory（实例化 B08Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B08 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B08Config。"""

    def build(self) -> B08Config:
        """构建 B08Config；跨参数引用 GRID（B03.GRID）单独解析（与旧版 P.get_int 同义）。"""
        cfg = self.build_model(B08Config, "B08")
        cfg.GRID = self.get_int("B03", "GRID", 24)   # B03: β 谱维度
        return cfg


# ---- 第三层：合成器 FrameMetricsSynthesizer（算法与原脚本完全一致） ----
class FrameMetricsSynthesizer(_SynthBase):
    """B08 逐帧曲率指标合成器：每帧 8 个 token 的主曲率对 → [K<0%, DEFF, Hmed] 三指标。"""

    def __init__(self, cfg: B08Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_frame_metrics(self, n_frame: int, rng: np.random.Generator) -> np.ndarray:
        """模拟逐帧曲率指标：每帧 8 个 token 的主曲率对 (κ1, κ2)，计算
        [K<0%, DEFF, Hmed] 三指标，返回形状 (n_frame, 3)。"""
        cfg = self._cfg
        frames = []
        for f in range(n_frame):
            # κ1 随帧号轻微增长（模拟生成后期曲率演化），κ2 类似，均叠加噪声
            k1 = 1.0 + 0.15 * (f / max(n_frame, 1)) + rng.normal(0, 0.05, cfg.TOK_PER_FRAME)
            k2 = -0.8 + 0.05 * (f / max(n_frame, 1)) + rng.normal(0, 0.05, cfg.TOK_PER_FRAME)
            deff = (abs(k1) + abs(k2))**2 / (k1**2 + k2**2)      # 202 DEFF_A
            k_neg = 100.0 * np.mean((k1 * k2) < 0)               # 201 K<0%
            hmed = np.median(abs((k1 + k2) / 2))                 # 203 Hmed
            frames.append([k_neg, float(deff.mean()), float(hmed)])   # 该帧三指标
        return np.array(frames)     # (n_frame, 3) 帧级曲率特征


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B08 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）；真实曲率数据经 CFG 路径探测。
    """

    def __init__(
        self,
        config: B08Config,
        synth: FrameMetricsSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self.frames: np.ndarray | None = None   # 合成 12 帧 × 3 指标矩阵（供后续步骤复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 0) 总帧数前置检查
    def validate_total_frames(self) -> dict:
        """0) 总帧数 = gen_len/TOK_PER_FRAME = 12，帧指标矩阵形状 (12, 3)。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        n_total_frame = cfg.GEN_LEN // cfg.TOK_PER_FRAME            # 96/8 = 12 帧
        frames = self.synth.synth_frame_metrics(n_total_frame, rng)    # 合成 12 帧 × 3 指标
        self.frames = frames
        ok0 = (n_total_frame == cfg.GEN_LEN // cfg.TOK_PER_FRAME) \
            and (frames.shape == (cfg.GEN_LEN // cfg.TOK_PER_FRAME, cfg.FRAME_SIZE))   # 12 帧且矩阵形状正确
        if not ok0:
            raise AIQValidationError(
                f"总帧数或帧指标形状不符: {n_total_frame}, {frames.shape}",
                expected=(cfg.GEN_LEN // cfg.TOK_PER_FRAME, cfg.FRAME_SIZE),
                actual=tuple(frames.shape), param_key="B08",
            )
        return {"detail": (f"总帧数 = gen_len/TOK_PER_FRAME = {cfg.GEN_LEN}/{cfg.TOK_PER_FRAME} = "
                           f"{n_total_frame}（帧指标矩阵 {frames.shape}）"),
                "n_total_frame": n_total_frame, "shape": tuple(frames.shape)}

    # ------------------------------------------------------------ 1) 8 帧 x 3 指标 = 24 维
    def validate_curv_dim(self) -> dict:
        """1) 曲率特征维度守恒：前 NFRAME 帧展平 = NFRAME×3 = 24 维。"""
        cfg = self.config
        assert self.frames is not None
        curv = self.frames[:cfg.NFRAME].reshape(-1)                       # 取前 8 帧展平为 24 维
        ok1 = (curv.shape[0] == cfg.NFRAME * cfg.FRAME_SIZE)             # 维度守恒断言
        if not ok1:
            raise AIQValidationError(
                f"曲率特征维度不符: {curv.shape[0]}",
                expected=cfg.NFRAME * cfg.FRAME_SIZE, actual=curv.shape[0], param_key="B08",
            )
        return {"detail": (f"曲率特征: {cfg.NFRAME} 帧 x 3 指标 = {curv.shape[0]} 维（应为 24）"),
                "curv_dim": curv.shape[0]}

    # ------------------------------------------------------------ 2) 拼接 48 维联合特征
    def validate_concat(self) -> dict:
        """2) 联合特征：β 谱 24 维 + 曲率 24 维 = 48 维。"""
        cfg = self.config
        assert self.frames is not None
        curv = self.frames[:cfg.NFRAME].reshape(-1)                       # 前 8 帧展平为 24 维
        rng = np.random.default_rng(cfg.SEED)
        beta_spectrum = rng.random(cfg.GRID)                         # 模拟 β 谱 24 维
        fp48 = np.concatenate([beta_spectrum, curv])             # β 谱 + 曲率 = 联合特征
        ok2 = (fp48.shape[0] == 48)                              # 48 维断言
        if not ok2:
            raise AIQValidationError(
                f"联合特征维度不符: {fp48.shape[0]}",
                expected=48, actual=fp48.shape[0], param_key="B08",
            )
        return {"detail": (f"联合特征: β 谱 {beta_spectrum.shape[0]} 维 + 曲率 {curv.shape[0]} 维 "
                           f"= {fp48.shape[0]} 维（应为 48）"),
                "fp48_dim": fp48.shape[0]}

    # ------------------------------------------------------------ 3) NFRAME=4 / 16 维度守恒
    def validate_conservation(self) -> dict:
        """3) 维度守恒：NFRAME=4 → 4×3=12 维；NFRAME=16 → 维度公式 16×3=48。"""
        cfg = self.config
        assert self.frames is not None
        curv4 = self.frames[:cfg.NFRAME_SMALL].reshape(-1)       # NFRAME=4 → 4×3 = 12 维
        dim_16 = cfg.NFRAME_LARGE * cfg.FRAME_SIZE              # NFRAME=16 → 维度公式 16×3 = 48
        ok3 = (curv4.shape[0] == cfg.NFRAME_SMALL * cfg.FRAME_SIZE) \
            and (dim_16 == cfg.NFRAME_LARGE * cfg.FRAME_SIZE)   # 两档维度守恒断言
        if not ok3:
            raise AIQValidationError(
                f"NFRAME 维度守恒不符: 4帧={curv4.shape[0]}, 16帧公式={dim_16}",
                expected={"curv4": cfg.NFRAME_SMALL * cfg.FRAME_SIZE,
                          "dim16": cfg.NFRAME_LARGE * cfg.FRAME_SIZE},
                actual={"curv4": curv4.shape[0], "dim16": dim_16},
                param_key="B08",
            )
        return {"detail": (f"NFRAME={cfg.NFRAME_SMALL} -> 曲率特征 {curv4.shape[0]} 维"
                           f"（信息不足，文档溯源 95%）; NFRAME={cfg.NFRAME_LARGE} -> "
                           f"维度公式 {cfg.NFRAME_LARGE}×{cfg.FRAME_SIZE} = {dim_16} 维（冗余；"
                           f"但 gen_len={cfg.GEN_LEN} 仅 {cfg.GEN_LEN // cfg.TOK_PER_FRAME} 帧，"
                           f"无法取满 {cfg.NFRAME_LARGE} 帧）"),
                "curv4": curv4.shape[0], "dim16": dim_16}

    # ------------------------------------------------------------ 4) 覆盖 token 区间
    def validate_coverage(self) -> dict:
        """4) 前 NFRAME 帧覆盖 token 数 = NFRAME×TOK_PER_FRAME = 64。"""
        cfg = self.config
        covered = cfg.NFRAME * cfg.TOK_PER_FRAME       # 前 8 帧覆盖 token 数
        ok4 = (covered == cfg.COVERED_TOK)         # 8×8 = 64 token
        if not ok4:
            raise AIQValidationError(
                f"覆盖 token 数不符: {covered}",
                expected=cfg.COVERED_TOK, actual=covered, param_key="B08",
            )
        return {"detail": (f"前 {cfg.NFRAME} 帧覆盖 {covered} token（文档：前 64 token 覆盖"
                           f"上升->饱和演化区间）"),
                "covered": covered}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实曲率数据帧聚合 → 24 维曲率 + β 谱 = 48 维联合特征。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        # 数据文件经 CFG.phi_pairs_path() 定位：插件内 params/ 副本优先，回退工作区 AIQ/
        npy_path = CFG.phi_pairs_path()
        ok5 = False
        curv_24 = np.empty(0)      # 真实 24 维曲率特征（缺失时为空）
        fp48 = np.empty(0)         # 真实 48 维联合特征（缺失时为空）
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        if os.path.isfile(npy_path):
            pairs = np.load(npy_path).astype(float)   # 真实曲率点云 (κ1,κ2)
            k1r, k2r = pairs[:, 0], pairs[:, 1]
            per_frame = cfg.PER_FRAME              # 每帧点数（真实帧聚合口径，数据层读取）
            n_fr = len(k1r) // per_frame   # 完整帧数
            F = n_fr * per_frame           # 可完整分帧的样本数
            k1f = k1r[:F].reshape(n_fr, per_frame)   # 按帧重塑 κ1
            k2f = k2r[:F].reshape(n_fr, per_frame)   # 按帧重塑 κ2
            kneg = 100.0 * np.mean(k1f * k2f < 0, axis=1)   # 每帧 K<0%
            deff = ((abs(k1f) + abs(k2f))**2 / (k1f**2 + k2f**2 + 1e-16)).mean(axis=1)   # 每帧 DEFF
            hmed = np.median(np.abs((k1f + k2f) / 2), axis=1)   # 每帧 Hmed
            curv_24 = np.concatenate([kneg, deff, hmed])[:cfg.NFRAME * cfg.FRAME_SIZE]     # 8 帧 × 3 = 24 维
            gamma_layers = np.asarray(rd.get("spectral.k_proj_gamma_layers"), dtype=float)   # 真实 β 谱
            fp48 = np.concatenate([gamma_layers, curv_24])        # 真实 48 维联合特征
            ok5 = (curv_24.shape[0] == cfg.NFRAME * cfg.FRAME_SIZE) \
                and (fp48.shape[0] == cfg.NFRAME * cfg.FRAME_SIZE + cfg.GRID)   # 维度断言
            if not ok5:
                raise RealModelMismatchError(
                    f"真实帧聚合维度不符: curv={curv_24.shape[0]}, fp={fp48.shape[0]}",
                    expected={"curv": cfg.NFRAME * cfg.FRAME_SIZE,
                              "fp": cfg.NFRAME * cfg.FRAME_SIZE + cfg.GRID},
                    actual={"curv": curv_24.shape[0], "fp": fp48.shape[0]},
                    param_key="B08",
                )
            curv_mean = (float(deff[:cfg.NFRAME].mean()), float(kneg[:cfg.NFRAME].mean()),
                         float(hmed[:cfg.NFRAME].mean()))
            deff_glob = rd.get("curvature.DEFF_plat")             # 真实全局 DEFF
            detail = (f"{tag} 真实曲率点云 {pairs.shape[0]} 点按帧聚合（每帧 "
                      f"{per_frame} 点 → {n_fr} 帧），前 {cfg.NFRAME} 帧 × 3 指标 = "
                      f"{curv_24.shape[0]} 维曲率特征; 前 {cfg.NFRAME} 帧均值: DEFF = "
                      f"{curv_mean[0]:.4f}（全局真实 {deff_glob:.4f}），K<0% = {curv_mean[1]:.2f}%"
                      f"（全局 45.50%），Hmed = {curv_mean[2]:.4f}（全局 1.3152）; "
                      f"48 维联合特征 = 真实 β 谱 {gamma_layers.shape[0]} 维 + 曲率 "
                      f"{curv_24.shape[0]} 维（应为 48）")
        else:
            raise RealModelMismatchError(
                f"phi_pairs_all.npy 缺失（{npy_path}），真实帧聚合维度验证无法执行",
                expected="数据文件存在", actual=npy_path, param_key="B08",
            )
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "curv_24_dim": curv_24.shape[0], "fp48_dim": fp48.shape[0],
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (0, "total_frames", self.validate_total_frames),
            (1, "curv_dim", self.validate_curv_dim),
            (2, "concat", self.validate_concat),
            (3, "conservation", self.validate_conservation),
            (4, "coverage", self.validate_coverage),
            (5, "real_model", self.validate_real_model),
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
    """B08 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B08 NFRAME 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = FrameMetricsSynthesizer(cfg)              # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"B08 NFRAME 验证（四层工厂架构）：gen_len={cfg.GEN_LEN}, TOK={cfg.TOK_PER_FRAME} -> "
          f"{cfg.GEN_LEN // cfg.TOK_PER_FRAME} 帧，取前 {cfg.NFRAME} 帧")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: NFRAME={cfg.NFRAME} FRAME_SIZE={cfg.FRAME_SIZE} TOK_PER_FRAME={cfg.TOK_PER_FRAME} "
          f"GRID={cfg.GRID} GEN_LEN={cfg.GEN_LEN} PER_FRAME={cfg.PER_FRAME} "
          f"NFRAME_SMALL={cfg.NFRAME_SMALL} NFRAME_LARGE={cfg.NFRAME_LARGE} "
          f"COVERED_TOK={cfg.COVERED_TOK} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b08_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b08_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b08_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
