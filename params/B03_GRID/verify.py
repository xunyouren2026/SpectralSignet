# -*- coding: utf-8 -*-
"""B03 GRID 深度剖面插值格点 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 深度归一化 depth = layer_index/(num_layers-1)，端点 0 与 1
  2. GRID=24 点均匀格点 [0, 1/23, ..., 1]
  3. 12 层模型线性插值到 24 格点，复现文档 GPT-2 示例
  4. 层数 == GRID 的模型（Qwen 24 层）无需插值 -> 恒等映射
  5. 端点值保持 + 留一法插值误差量级（报告项，无断言）
  6. 真实模型对照：真实 24 层 k_proj Gamma → 24 格点插值（恒等映射 + 端点保持）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B03Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 B03Config（环境变量 AIQ_B03_<KEY> > YAML > _params_data.json > 默认）
  GridSynthesizer         —— 深度归一化 + 均匀格点 + 线性插值（np.interp）+ 留一法误差
  ValidatorEngine         —— 6 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 862-994（B03）
  《参数完整定义与公式.txt》B03 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实 k_proj 逐层 Gamma（24 维数组，spectral.k_proj_gamma_layers）
          → GRID=24 格点插值（层数==GRID 恒等映射 + 端点保持）。
  如实呈现：真实 24 层剖面直接落在 24 格点上，无需插值（恒等）；文档 GPT-2
            12→24 示例为另一家族剖面，仅作插值算法对照。
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

# ---- 第一层：配置模型 B03Config（pydantic 优先；dataclass 回退） ----
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


class B03Config(_ConfigModelBase):
    """B03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B03 节点键名一一对应；取值优先级：
    环境变量 AIQ_B03_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    GRID: int = 24                   # 深度剖面插值格点数（统一跨模型对齐的深度采样数）
    EPS: float = 1e-12               # 深度端点/格点均匀性误差容差——浮点容差，保留
    DOC_TOL: float = 0.03            # 与文档插值结果最大绝对误差容差（文档为手输/四舍五入值）
    LO_DEPTH: float = 0.0            # 深度格点下界
    HI_DEPTH: float = 1.0            # 深度格点上界
    # 文档给出的 GPT-2 插值结果（四舍五入到 2 位小数，数据层读取）
    DOC_INTERP: list = [
        0.32, 0.35, 0.38, 0.41, 0.45, 0.48, 0.52, 0.55,
        0.58, 0.60, 0.61, 0.60, 0.59, 0.57, 0.55, 0.52,
        0.48, 0.45, 0.42, 0.40, 0.38, 0.37, 0.36, 0.35,
    ]
    RAW_GPT2: list = [0.32, 0.38, 0.45, 0.52, 0.58, 0.61,
                      0.59, 0.55, 0.48, 0.42, 0.38, 0.35]   # 12 层原始 β（README ⑤）
    L_PROBES: list = [12, 24, 32]    # 深度归一化检查的层数


# ---- 第二层：配置工厂 ConfigFactory（实例化 B03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B03 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B03Config。"""

    def build(self) -> B03Config:
        """构建 B03Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(B03Config, "B03")


# ---- 第三层：合成器 GridSynthesizer（算法与原脚本完全一致） ----
class GridSynthesizer(_SynthBase):
    """B03 深度插值合成器：深度归一化 + 均匀格点 + 线性插值（np.interp）+ 留一法误差。"""

    def __init__(self, cfg: B03Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def depth_normalized(self, n_layers: int) -> np.ndarray:
        """深度归一化：depth = layer_index/(num_layers-1)，端点 0 与 1。"""
        if n_layers < 2:
            raise ValueError(f"层数须 >= 2 才有归一化深度: {n_layers}")   # 防御：<2 层无深度区间
        return np.arange(n_layers) / (n_layers - 1.0)   # 首层 0、末层 1，等距归一化

    def make_grid(self, n_grid: int) -> np.ndarray:
        """目标格点：GRID 个等间距采样点 [LO_DEPTH, ..., HI_DEPTH]。"""
        cfg = self._cfg
        if n_grid < 2:
            raise ValueError(f"格点数须 >= 2: {n_grid}")   # 防御：<2 点不成区间
        return np.linspace(cfg.LO_DEPTH, cfg.HI_DEPTH, n_grid)               # 等距格点（含端点）

    def interp_profile(self, prof: np.ndarray, n_grid: int) -> np.ndarray:
        """线性插值到统一深度格点（源码用 np.interp）。"""
        prof = np.asarray(prof, dtype=float)
        if prof.size == 0:
            raise ValueError("空剖面输入")      # 防御：空剖面无法定义插值域
        if n_grid < 2:
            raise ValueError(f"格点数须 >= 2: {n_grid}")   # 防御：格点须成区间
        pos = self.depth_normalized(prof.size)       # 源层深度位置（[0,1]）
        return np.interp(self.make_grid(n_grid), pos, prof)      # 目标格点处线性插值

    def leave_one_out_err(self, prof: np.ndarray) -> tuple[float, float]:
        """留一法插值误差：删去一层，用其余层插值该深度，返回 (max, mean)。"""
        prof = np.asarray(prof, dtype=float)
        n = prof.size
        if n < 3:
            raise ValueError(f"留一法须 n >= 3: {n}")   # 防御：<3 层无法留一
        errs = []
        for leave in range(n):
            keep_idx = np.delete(np.arange(n), leave)         # 保留层索引（剔除 leave）
            keep_pos = keep_idx / (n - 1.0)                   # 保留层深度位置
            # 用保留层插值被删层深度处的值，与该层真实值比较得误差
            est = np.interp(np.array([leave / (n - 1.0)]), keep_pos, prof[keep_idx])[0]
            errs.append(abs(est - prof[leave]))
        return float(max(errs)), float(np.mean(errs))         # (最大误差, 平均误差)


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B03 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: B03Config,
        synth: GridSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 深度归一化端点
    def validate_depth_endpoints(self) -> dict:
        """1) 深度归一化端点：多种层数（12/24/32）下首层 0、末层 1。"""
        cfg = self.config
        ok1 = True
        rows = []
        # 在多种层数（12/24/32）下检查归一化深度端点
        for L in cfg.L_PROBES:
            depth = self.synth.depth_normalized(L)
            one_ok = (abs(depth[0] - 0.0) < cfg.EPS and abs(depth[-1] - 1.0) < cfg.EPS)   # 端点 0 与 1
            ok1 &= one_ok
            rows.append(f"层数 L={L:>2}: 深度端点 = [{depth[0]:.3f}, {depth[-1]:.3f}]")
        if not ok1:
            raise AIQValidationError(
                "深度归一化端点不符（应为 0 与 1）",
                expected=(0.0, 1.0), actual=rows, param_key="B03",
            )
        return {"detail": "深度归一化端点: " + "; ".join(rows)}

    # ------------------------------------------------------------ 2) 24 点均匀格点
    def validate_grid_uniform(self) -> dict:
        """2) 24 点均匀格点：max|grid - j/23| < EPS。"""
        cfg = self.config
        grid = self.synth.make_grid(cfg.GRID)                      # 目标格点
        expected_grid = np.arange(cfg.GRID) / (cfg.GRID - 1.0)   # 理论格点 j/23
        err_grid = float(np.max(np.abs(grid - expected_grid)))   # 均匀性偏差
        ok2 = err_grid < cfg.EPS                        # 须与理论格点逐点一致
        if not ok2:
            raise AIQValidationError(
                f"格点不均匀: max 偏差 {err_grid:.2e}",
                expected=cfg.EPS, actual=err_grid, param_key="B03",
            )
        return {"detail": (f"格点均匀性: max|grid - j/23| = {err_grid:.2e}; "
                           f"格点 = {np.round(grid, 4)[:6]} ... {np.round(grid, 4)[-3:]}（{cfg.GRID} 点）"),
                "err_grid": err_grid}

    # ------------------------------------------------------------ 3) GPT-2 12 层插值到 24 格点
    def validate_gpt2_interp(self) -> dict:
        """3) GPT-2 12 层 → 24 格点插值：与文档示例偏差 ≤ DOC_TOL 且形状 (GRID,)。"""
        cfg = self.config
        doc_interp = np.asarray(cfg.DOC_INTERP, dtype=float)
        raw_gpt2 = np.asarray(cfg.RAW_GPT2, dtype=float)
        interp = self.synth.interp_profile(raw_gpt2, cfg.GRID)     # 12 层 → 24 格点插值
        err_doc = float(np.max(np.abs(interp - doc_interp)))   # 与文档值的最大偏差
        err_end = max(abs(interp[0] - raw_gpt2[0]), abs(interp[-1] - raw_gpt2[-1]))   # 端点保持
        ok3 = (err_doc <= cfg.DOC_TOL) and (interp.shape == (cfg.GRID,))   # 偏差容差 + 形状断言
        if not ok3:
            raise AIQValidationError(
                f"与文档插值结果偏差过大: {err_doc:.4f}",
                expected=cfg.DOC_TOL, actual=err_doc, param_key="B03",
            )
        bad_idx = np.where(np.abs(interp - doc_interp) > 0.01)[0]   # 偏差>0.01 的格点索引
        note = ""
        if len(bad_idx):
            note = (f"; 注：文档值与精确插值偏差>0.01 的格点 idx = {bad_idx.tolist()}，"
                    f"max 偏差 {err_doc:.4f}（文档手输/四舍五入所致，精确 np.interp 为基准）")
        return {"detail": (f"GPT-2 12 层 -> 24 格点插值（文档示例对照）: 与文档最大绝对误差 = "
                           f"{err_doc:.4f}（容差 {cfg.DOC_TOL}）；端点保持误差 = {err_end:.2e}{note}"),
                "err_doc": err_doc, "err_end": err_end, "shape": tuple(interp.shape)}

    # ------------------------------------------------------------ 4) 层数 == GRID 恒等映射
    def validate_identity(self) -> dict:
        """4) 层数 == GRID（Qwen 24 层）时插值是恒等映射（偏差 < EPS）。"""
        cfg = self.config
        # 构造 24 层带微小振荡的剖面（模拟 Qwen 24 层剖面）
        qwen_raw = np.linspace(0.15, 0.60, 24) + 0.02 * np.sin(np.arange(24))
        qwen_interp = self.synth.interp_profile(qwen_raw, cfg.GRID)      # 24 层 → 24 格点
        err_identity = float(np.max(np.abs(qwen_interp - qwen_raw)))   # 应 ≈ 0
        ok4 = err_identity < cfg.EPS        # 层数==GRID 时插值是恒等映射
        if not ok4:
            raise AIQValidationError(
                f"恒等映射失败: max 偏差 {err_identity:.2e}",
                expected=cfg.EPS, actual=err_identity, param_key="B03",
            )
        return {"detail": (f"Qwen 24 层（层数==GRID）: 插值后与原值最大偏差 = {err_identity:.2e}"
                           f"（应≈0，无需插值）"),
                "err_identity": err_identity}

    # ------------------------------------------------------------ 5) 留一法插值误差（报告项）
    def validate_loo(self) -> dict:
        """5) 留一法插值误差量级（报告项，仅数值健全性检查）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        # 合成正弦型剖面（带轻微噪声，模拟真实 β 剖面的平滑结构）
        prof = 0.30 + 0.30 * np.sin(np.linspace(0, np.pi, 12)) + 0.005 * rng.standard_normal(12)
        loo_max, loo_mean = self.synth.leave_one_out_err(prof)      # 留一法插值误差
        ok5 = np.isfinite(loo_max)     # 仅作数值健全性检查
        if not ok5:
            raise AIQValidationError(
                f"留一法插值出现非有限值: max={loo_max}",
                expected="finite", actual=loo_max, param_key="B03",
            )
        return {"detail": (f"留一法插值误差: max = {loo_max:.4f}, mean = {loo_mean:.4f} "
                           f"（文档报 ~0.005：真实 β 剖面更平滑；本合成正弦剖面 LOO 误差略大，"
                           f"属剖面依赖，不设严格断言）"),
                "loo_max": loo_max, "loo_mean": loo_mean}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 24 层 k_proj Gamma → 24 格点插值（恒等映射 + 端点保持）。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        gamma_layers = np.asarray(rd.get("spectral.k_proj_gamma_layers"), dtype=float)  # 真实逐层 Gamma
        interp_real = self.synth.interp_profile(gamma_layers, cfg.GRID)   # 24 层 → 24 格点
        err_identity_real = float(np.max(np.abs(interp_real - gamma_layers)))   # 恒等偏差
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        ok6 = (gamma_layers.size == cfg.GRID) and (err_identity_real < cfg.EPS)   # 恒等映射断言
        if not ok6:
            raise RealModelMismatchError(
                f"真实剖面插值非恒等: size={gamma_layers.size}, err={err_identity_real:.2e}",
                expected={"size": cfg.GRID, "err": cfg.EPS},
                actual={"size": gamma_layers.size, "err": err_identity_real},
                param_key="B03",
            )
        return {"detail": (f"{tag} 真实 k_proj 逐层 Gamma（{gamma_layers.size} 层）→ "
                           f"{cfg.GRID} 格点插值（层数==GRID，恒等映射）: 最大偏差 = {err_identity_real:.2e}; "
                           f"端点保持: 层0 gamma={gamma_layers[0]:.4f} ↔ 格点0={interp_real[0]:.4f}; "
                           f"层23 gamma={gamma_layers[-1]:.4f} ↔ 格点23={interp_real[-1]:.4f}; "
                           f"真实 gamma 剖面范围 [{gamma_layers.min():.3f}, {gamma_layers.max():.3f}]"
                           f"（与文档 GPT-2 示例为不同家族，仅演示真实插值）"),
                "source": rd.source_tag(), "tag": tag,
                "err_identity_real": err_identity_real}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "depth_endpoints", self.validate_depth_endpoints),
            (2, "grid_uniform", self.validate_grid_uniform),
            (3, "gpt2_interp", self.validate_gpt2_interp),
            (4, "identity", self.validate_identity),
            (5, "loo", self.validate_loo),
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
    """B03 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B03 GRID 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = GridSynthesizer(cfg)                      # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"B03 GRID 验证（四层工厂架构）：GRID={cfg.GRID} 点深度插值")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: GRID={cfg.GRID} EPS={cfg.EPS:.0e} DOC_TOL={cfg.DOC_TOL} "
          f"DEPTH=[{cfg.LO_DEPTH},{cfg.HI_DEPTH}] L_PROBES={cfg.L_PROBES} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b03_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
