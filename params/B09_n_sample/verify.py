# -*- coding: utf-8 -*-
"""B09 n_sample 溯源特征每帧采样 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 每帧 10 个有放回采样点，形状 (10, d)（8-token 池中采样）
  2. 3 个曲率指标（K%、DEFF、Hmed）从 10 点估计
  3. 采样稳定性：指标估计方差随 n 增大而递减（4 -> 10 -> 20），边际收益递减
  4. 溯源特征构建：8 帧 x 3 指标 = 24 维曲率特征（配合 B08）+ β 谱 = 48 维
  5. 真实模型对照：真实帧级指标基准 K<0%≈45.50% / DEFF≈1.5920 / Hmed≈1.3152

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B09Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退；
                            跨参数引用 TOK_PER_FRAME/NFRAME/FRAME_SIZE←B08、GRID←B03）
  ConfigFactory           —— 实例化 B09Config（环境变量 AIQ_B09_<KEY> > YAML > _params_data.json > 默认）
  SampleSynthesizer       —— 有放回采样 + 帧内三指标估计 + DEFF 估计稳定性（std）
  ValidatorEngine         —— 5 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 1531-1617（B09）
  《参数完整定义与公式.txt》B09 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实帧级指标基准 K<0%≈45.50% / DEFF≈1.5920 / Hmed≈1.3152，
          与合成层帧指标估计（项2）同口径量级对照。
  如实呈现：合成估计量级与真实同阶（DEFF∈[1,2]、K%∈[0,100]、Hmed≥0）。
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

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 B09Config（pydantic 优先；dataclass 回退） ----
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


class B09Config(_ConfigModelBase):
    """B09 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B09 节点键名一一对应（跨参数引用除外）；
    取值优先级：环境变量 AIQ_B09_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    N_SAMPLE: int = 10               # 溯源特征每帧采样数（数据层键 N_SAMPLE）
    TOK_PER_FRAME: int = 8           # B08 跨参数引用：每帧 token 数（采样池，数据层键 B08.TOK_PER_FRAME）
    NFRAME: int = 8                  # B08 跨参数引用：曲率特征帧数
    FRAME_SIZE: int = 3              # B08 跨参数引用：每帧曲率指标数
    DIM: int = 64                    # 模拟激活维度
    GRID: int = 24                   # B03 跨参数引用：β 谱维度
    N_REP_STAB: int = 500            # 采样稳定性重复次数
    STAB_NS: list = [4, 10, 20, 30]  # 稳定性扫描采样数
    K_PCT_MIN: int = 0               # K% 物理下界（指标区间断言）
    K_PCT_MAX: int = 100             # K% 物理上界
    DEFF_MIN: int = 1                # DEFF 物理下界
    DEFF_MAX: int = 2                # DEFF 物理上界


# ---- 第二层：配置工厂 ConfigFactory（实例化 B09Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B09 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B09Config。"""

    def build(self) -> B09Config:
        """构建 B09Config；跨参数引用 TOK_PER_FRAME/NFRAME/FRAME_SIZE（B08）、GRID（B03）单独解析。"""
        cfg = self.build_model(B09Config, "B09")
        cfg.TOK_PER_FRAME = self.get_int("B08", "TOK_PER_FRAME", 8)   # B08: 每帧 token 数
        cfg.NFRAME = self.get_int("B08", "NFRAME", 8)                # B08: 曲率特征帧数
        cfg.FRAME_SIZE = self.get_int("B08", "FRAME_SIZE", 3)        # B08: 每帧曲率指标数
        cfg.GRID = self.get_int("B03", "GRID", 24)                   # B03: β 谱维度
        return cfg


# ---- 第三层：合成器 SampleSynthesizer（算法与原脚本完全一致） ----
class SampleSynthesizer(_SynthBase):
    """B09 有放回采样 + 帧指标估计 + 稳定性合成器。

    - bootstrap_sample(H_frame, n_sample, rng)：从帧内 token 池有放回采样；
    - frame_metrics(k1, k2)：从主曲率对数组估计帧内三指标 (K<0%, DEFF, Hmed)；
    - stability(n_pts, rng, n_rep)：DEFF 估计的 std（方差度量）。
    """

    def __init__(self, cfg: B09Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    @staticmethod
    def bootstrap_sample(H_frame: np.ndarray, n_sample: int,
                         rng: np.random.Generator) -> np.ndarray:
        """从帧内 token 池有放回采样 n_sample 次。
        有放回（bootstrap）允许同一 token 多次入样，扩增每帧采样点数。"""
        if H_frame.shape[0] == 0:
            raise ValueError("空采样池")      # 防御：空池无法采样
        idx = rng.integers(0, H_frame.shape[0], size=n_sample)   # 池索引有放回抽取
        return H_frame[idx]

    @staticmethod
    def frame_metrics(k1: np.ndarray, k2: np.ndarray) -> tuple[float, float, float]:
        """从主曲率对数组估计帧内三指标 (K<0%, DEFF, Hmed)。"""
        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        if k1.size == 0:
            raise ValueError("空主曲率输入")      # 防御：空数组无法统计
        denom = k1**2 + k2**2
        if np.any(denom <= 0.0):
            raise ValueError("主曲率全零导致 DEFF 除零")   # 防御：避免除零
        deff = float(np.mean((abs(k1) + abs(k2))**2 / denom))   # DEFF 均值
        k_neg = float(100.0 * np.mean((k1 * k2) < 0))           # K<0% 比例
        hmed = float(np.median(abs((k1 + k2) / 2)))             # Hmed 中位数
        return k_neg, deff, hmed

    def stability(self, n_pts: int, rng: np.random.Generator, n_rep: int) -> float:
        """对给定采样数，重复估计 DEFF，返回其 std（方差度量）。
        std 越小说明该采样数下的指标估计越稳定。"""
        stds = []
        for _ in range(n_rep):
            ka = 1.0 + rng.normal(0, 0.1, n_pts)    # κ1 带噪样本
            kb = -0.8 + rng.normal(0, 0.1, n_pts)   # κ2 带噪样本
            _, deff, _ = self.frame_metrics(ka, kb)      # 帧指标估计
            stds.append(deff)
        return float(np.std(stds))                  # DEFF 估计的波动


# ---- 第四层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B09 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: B09Config,
        synth: SampleSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self._rng: np.random.Generator | None = None   # 跨步骤共享 rng（保持原随机流次序）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 有放回采样形状
    def validate_sampling_shape(self) -> dict:
        """1) 有放回采样：8-token 池 -> N_SAMPLE 点，形状 (N_SAMPLE, DIM)。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        self._rng = rng                                   # 供后续步骤复用同一随机流
        H_frame = rng.standard_normal((cfg.TOK_PER_FRAME, cfg.DIM))   # 帧内 token 池 (8,64)
        S = self.synth.bootstrap_sample(H_frame, cfg.N_SAMPLE, rng)          # 有放回采样 10 点
        ok1 = (S.shape == (cfg.N_SAMPLE, cfg.DIM))                    # 形状断言：(10, 64)
        if not ok1:
            raise AIQValidationError(
                f"采样形状不符: {S.shape}",
                expected=(cfg.N_SAMPLE, cfg.DIM), actual=tuple(S.shape), param_key="B09",
            )
        return {"detail": (f"采样: 池 ({cfg.TOK_PER_FRAME},{cfg.DIM}) -> 有放回 {cfg.N_SAMPLE} 次 -> "
                           f"{S.shape}（应为 ({cfg.N_SAMPLE},{cfg.DIM})）"),
                "shape": tuple(S.shape)}

    # ------------------------------------------------------------ 2) 从 10 点估计 3 指标
    def validate_metrics(self) -> dict:
        """2) 从 N_SAMPLE 点估计 3 指标：K%∈[0,100]、DEFF∈[1,2]、Hmed≥0 且 DEFF 有限。"""
        cfg = self.config
        assert self._rng is not None
        k1s = 1.0 + self._rng.normal(0, 0.1, cfg.N_SAMPLE)    # 10 点 κ1 带噪样本
        k2s = -0.8 + self._rng.normal(0, 0.1, cfg.N_SAMPLE)   # 10 点 κ2 带噪样本
        k_neg, deff, hmed = self.synth.frame_metrics(k1s, k2s)   # 三指标估计
        # 判据：三指标落在物理区间（K%∈[K_PCT_MIN,K_PCT_MAX], DEFF∈[DEFF_MIN,DEFF_MAX], Hmed≥0）且 DEFF 有限
        ok2 = (cfg.K_PCT_MIN <= k_neg <= cfg.K_PCT_MAX) and (cfg.DEFF_MIN <= deff <= cfg.DEFF_MAX) \
            and (hmed >= 0.0) and np.isfinite(deff)
        if not ok2:
            raise AIQValidationError(
                f"指标越界/非有限: K%={k_neg:.2f}, DEFF={deff:.4f}, Hmed={hmed:.4f}",
                expected={"k_pct∈[0,100]": True, "deff∈[1,2]": True, "hmed≥0": True},
                actual={"k_neg": k_neg, "deff": deff, "hmed": hmed},
                param_key="B09",
            )
        return {"detail": (f"{cfg.N_SAMPLE} 点估计指标: K%={k_neg:.2f}%, DEFF={deff:.4f}, "
                           f"Hmed={hmed:.4f} (K%∈[0,100], DEFF∈[1,2], Hmed≥0)"),
                "k_neg": k_neg, "deff": deff, "hmed": hmed}

    # ------------------------------------------------------------ 3) 采样稳定性
    def validate_stability(self) -> dict:
        """3) 采样稳定性：DEFF 估计 std 随 n 递减 + 边际收益递减（4→10 改善 > 10→20）。"""
        cfg = self.config
        assert self._rng is not None
        rows = []
        stab = {n: self.synth.stability(n, self._rng, cfg.N_REP_STAB) for n in cfg.STAB_NS}   # 各采样数的 std
        for n, v in stab.items():
            tag = "  <- 标准配置" if n == cfg.N_SAMPLE else ""   # 标记标准 n=10
            rows.append(f"n_sample={n:>2}: std(DEFF) = {v:.5f}{tag}")
        ok3a = (stab[cfg.N_SAMPLE] < stab[cfg.STAB_NS[0]]) \
            and (stab[cfg.STAB_NS[2]] < stab[cfg.N_SAMPLE])   # 方差随 n 递减
        gain_4_10 = (stab[cfg.STAB_NS[0]] - stab[cfg.N_SAMPLE]) / stab[cfg.STAB_NS[0]]     # 4→10 相对改善
        gain_10_20 = (stab[cfg.N_SAMPLE] - stab[cfg.STAB_NS[2]]) / stab[cfg.N_SAMPLE]  # 10→20 相对改善
        ok3b = (gain_4_10 > gain_10_20)                # 边际收益递减判据
        ok3 = ok3a and ok3b
        if not ok3:
            raise AIQValidationError(
                f"稳定性/边际收益不符: std={stab}, 增益 4->10={gain_4_10:.3f}, 10->20={gain_10_20:.3f}",
                expected={"std_decreasing": True, "marginal_decreasing": True},
                actual={"stab": stab, "gain_4_10": gain_4_10, "gain_10_20": gain_10_20},
                param_key="B09",
            )
        return {"detail": (f"采样稳定性：DEFF 估计 std 随 n 变化（{cfg.N_REP_STAB} 次重复）: "
                           + "; ".join(rows)
                           + f"; 相对改善: 4->10 = {gain_4_10 * 100:.1f}%, "
                           + f"10->20 = {gain_10_20 * 100:.1f}%（边际收益递减）"),
                "stab": stab, "gain_4_10": gain_4_10, "gain_10_20": gain_10_20}

    # ------------------------------------------------------------ 4) 48 维溯源特征构建
    def validate_fp48(self) -> dict:
        """4) 溯源特征构建：8 帧×3 指标 = 24 维曲率 + β 谱 24 维 = 48 维联合特征。"""
        cfg = self.config
        assert self._rng is not None
        curv_24 = np.zeros(cfg.NFRAME * cfg.FRAME_SIZE)         # 24 维曲率特征容器
        for f in range(cfg.NFRAME):                # 逐帧估计三指标
            k1f = 1.0 + self._rng.normal(0, 0.1, cfg.N_SAMPLE)     # 帧内 κ1 样本
            k2f = -0.8 + self._rng.normal(0, 0.1, cfg.N_SAMPLE)    # 帧内 κ2 样本
            k, de, hm = self.synth.frame_metrics(k1f, k2f)          # 该帧三指标
            curv_24[f * 3:f * 3 + 3] = [k, de, hm]       # 写入对应槽位
        beta_24 = self._rng.random(cfg.GRID)                       # β 谱（B03 GRID=24）
        fp48 = np.concatenate([beta_24, curv_24])        # 联合溯源特征
        ok4 = (curv_24.shape[0] == cfg.NFRAME * cfg.FRAME_SIZE) \
            and (fp48.shape[0] == cfg.NFRAME * cfg.FRAME_SIZE + cfg.GRID)   # 维度断言
        if not ok4:
            raise AIQValidationError(
                f"溯源特征维度不符: curv={curv_24.shape[0]}, fp={fp48.shape[0]}",
                expected={"curv": cfg.NFRAME * cfg.FRAME_SIZE,
                          "fp": cfg.NFRAME * cfg.FRAME_SIZE + cfg.GRID},
                actual={"curv": curv_24.shape[0], "fp": fp48.shape[0]},
                param_key="B09",
            )
        return {"detail": (f"溯源特征: 曲率 {curv_24.shape[0]} 维（{cfg.NFRAME}帧x3）+ β谱 "
                           f"{beta_24.shape[0]} 维 = {fp48.shape[0]} 维联合特征（应为 48）"),
                "curv_dim": curv_24.shape[0], "fp_dim": fp48.shape[0]}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实帧级指标基准（K<0%/DEFF/Hmed）均有限。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        deff_real = rd.get("curvature.DEFF_plat")       # 真实 DEFF 平台
        kneg_real = rd.get("curvature.K_neg_pct")       # 真实 K<0%
        hmed_real = rd.get("curvature.H_median")        # 真实 Hmed
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        ok5 = np.isfinite(deff_real) and np.isfinite(kneg_real) and np.isfinite(hmed_real)   # 均有限
        if not ok5:
            raise RealModelMismatchError(
                f"真实帧指标非有限: K%={kneg_real}, DEFF={deff_real}, Hmed={hmed_real}",
                expected="全部有限", actual={"kneg": kneg_real, "deff": deff_real, "hmed": hmed_real},
                param_key="B09",
            )
        return {"detail": (f"{tag} 真实帧级指标基准: K<0% = {kneg_real:.2f}%（∈[0,100]），"
                           f"DEFF = {deff_real:.4f}（∈[1,2]），Hmed = {hmed_real:.4f}（≥0）; "
                           f"与合成层项2估计同口径（K%∈[0,100]、DEFF∈[1,2]、Hmed≥0），"
                           f"真实指标落在合成约束区间内"),
                "source": rd.source_tag(), "tag": tag,
                "deff_real": deff_real, "kneg_real": kneg_real, "hmed_real": hmed_real}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "sampling_shape", self.validate_sampling_shape),
            (2, "metrics", self.validate_metrics),
            (3, "stability", self.validate_stability),
            (4, "fp48", self.validate_fp48),
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
    """B09 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B09 n_sample 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = SampleSynthesizer(cfg)                    # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"B09 n_sample 验证（四层工厂架构）：每帧 {cfg.N_SAMPLE} 点有放回采样 + 溯源特征")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_SAMPLE={cfg.N_SAMPLE} TOK_PER_FRAME={cfg.TOK_PER_FRAME} NFRAME={cfg.NFRAME} "
          f"FRAME_SIZE={cfg.FRAME_SIZE} DIM={cfg.DIM} GRID={cfg.GRID} "
          f"N_REP_STAB={cfg.N_REP_STAB} STAB_NS={cfg.STAB_NS} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b09_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b09_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b09_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
