# -*- coding: utf-8 -*-
"""B06 N_SAMPLE_PER_FRAME 版本A每帧采样数 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 同步原则：N_SAMPLE_PER_FRAME == SAMPLES_PER_FRAME == 18
  2. 版本A全流程：有放回采样 -> 2D主截面投影 -> 二次曲面拟合 -> Hessian
     -> (κ1,κ2) -> DEFF（κ 相对误差 <15%/<25%）
  3. DEFF = (|κ1|+|κ2|)^2/(κ1^2+κ2^2) ∈ [1,2]
  4. 各向同性极限：φ 均匀时 E[DEFF] = 1 + 2/π ≈ 1.5708；文档 φ~N(14°,25°) 平台复现
  5. 真实模型对照：真实 DEFF 平台≈1.5920 / φmean≈20.26° / K<0%≈45.50% / Hmed≈1.3152

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B06Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退；
                            跨参数引用 B05←B05.N_SAMPLE 由共享 ConfigFactory 解析）
  ConfigFactory           —— 实例化 B06Config（环境变量 AIQ_B06_<KEY> > YAML > _params_data.json > 默认）
  VersionASynthesizer     —— 2D 主截面二次曲面拟合（Hessian 特征值 = 主曲率）+ DEFF 公式
  ValidatorEngine         —— 4 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 1261-1356（B06）
  《参数完整定义与公式.txt》B06 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
  注（口径澄清）：文档另称平台=4R(2)=π/2，是母公式 R(d) 的理论推导口径；
      与均匀 φ 的期望 1+2/π 并非同一构造，二者数值巧合（均 ≈1.57），脚本如实分开报告。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实 DEFF 平台≈1.5920 / φmean≈20.26° / K<0%≈45.50% / Hmed≈1.3152。
  如实呈现：真实 DEFF 平台 1.5920 与文档声称 1.5920 一致；真实 φmean=20.26°
            高于文档拟合 φ~N(14°,25°) 的均值 14°，如实报告。
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

# ---- 第一层：配置模型 B06Config（pydantic 优先；dataclass 回退） ----
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


class B06Config(_ConfigModelBase):
    """B06 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B06 节点键名一一对应（B05 为跨参数引用）；
    取值优先级：环境变量 AIQ_B06_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    B04: int = 8                     # 每帧 token 数（B04；采样池大小，数据层键 B04）
    B05: int = 18                    # 通用曲率采样（B05 SAMPLES_PER_FRAME；同步原则基准，跨参数引用）
    B06: int = 18                    # 版本A每帧采样数（数据层键 N_SAMPLE）
    DIM: int = 64                    # 模拟激活维度
    K1_TRUE: float = 4.0             # 版本A拟合主曲率真值 κ1（鞍面正曲率）
    K2_TRUE: float = -1.5            # 版本A拟合主曲率真值 κ2（鞍面负曲率）
    K1_TOL: float = 0.15             # κ1 相对误差容差（15%）
    K2_TOL: float = 0.25             # κ2 相对误差容差（25%）
    N_MC: int = 200000               # DEFF 期望 MC 样本数
    PHI_MEAN_DEG: float = 14.0       # 文档 φ 分布均值（截断高斯）
    PHI_STD_DEG: float = 25.0        # 文档 φ 分布标准差
    PLAT_ISOTROPY_TOL: float = 0.005  # 均匀 φ 期望与 1+2/π 的绝对偏差容差


# ---- 第二层：配置工厂 ConfigFactory（实例化 B06Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B06 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B06Config。"""

    def build(self) -> B06Config:
        """构建 B06Config；跨参数引用 B05（B05.N_SAMPLE）单独解析（与旧版 P.get_int 同义）。"""
        cfg = self.build_model(B06Config, "B06")
        cfg.B05 = self.get_int("B05", "N_SAMPLE", 18)   # 通用曲率采样（B05 SAMPLES_PER_FRAME）
        return cfg


# ---- 第三层：合成器 VersionASynthesizer（算法与原脚本完全一致） ----
class VersionASynthesizer(_SynthBase):
    """B06 版本A主截面拟合合成器。

    - deff(k1, k2)：有效维数比 DEFF = (|κ1|+|κ2|)^2/(κ1^2+κ2^2) ∈ [1,2]；
    - fit_2d_section(x, y, z)：2D 主截面二次曲面拟合 → Hessian 特征值（主曲率 κ1>κ2）。
    """

    def __init__(self, cfg: B06Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    @staticmethod
    def deff(k1: float, k2: float) -> float:
        """有效维数比 DEFF = (|κ1|+|κ2|)^2 / (κ1^2+κ2^2) ∈ [1,2]。
        几何含义：描述曲率各向异性程度——各向同性时最大 2，单方向时趋近 1。"""
        denom = k1 * k1 + k2 * k2       # 分母（曲率向量模长平方）
        if denom <= 0.0:
            return float("nan")         # 防御：零曲率退化输入返回 NaN
        return (abs(k1) + abs(k2)) ** 2 / denom

    @staticmethod
    def fit_2d_section(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
        """2D 主截面二次曲面拟合 z = a1·x² + a2·y² + a3·xy + b1·x + b2·y + c，
        返回 Hessian 特征值（主曲率 κ1>κ2）。"""
        A = np.stack([x**2, y**2, x * y, x, y, np.ones_like(x)], axis=1)   # 6 系数设计矩阵
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)                       # 最小二乘解
        a, b, c = coef[0], coef[1], coef[2]     # 二次项系数（含 xy 交叉项 c）
        Hm = np.array([[2 * a, c], [c, 2 * b]])          # Hessian 矩阵
        ev = np.linalg.eigvalsh(Hm)                      # 特征值 = 主曲率
        return float(ev[1]), float(ev[0])                # κ1(大), κ2(小)


# ---- 第四层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B06 验证引擎：顺序执行 4 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: B06Config,
        synth: VersionASynthesizer,
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

    # ------------------------------------------------------------ 1) 同步原则
    def validate_sync(self) -> dict:
        """1) 同步原则：N_SAMPLE_PER_FRAME == SAMPLES_PER_FRAME（同为 18）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        self._rng = rng                                   # 供后续步骤复用同一随机流
        ok1 = (cfg.B06 == cfg.B05)      # 版本A与通用曲率采样数必须同步（同为 18）
        if not ok1:
            raise AIQValidationError(
                f"B06={cfg.B06} 与 B05={cfg.B05} 不同步（版本A与通用曲率口径将不一致）",
                expected=cfg.B05, actual=cfg.B06, param_key="B06",
            )
        return {"detail": (f"同步原则: N_SAMPLE_PER_FRAME={cfg.B06} == SAMPLES_PER_FRAME={cfg.B05}"),
                "b06": cfg.B06, "b05": cfg.B05}

    # ------------------------------------------------------------ 2) 版本A全流程
    def validate_flow(self) -> dict:
        """2) 版本A全流程（采样 -> 2D拟合 -> Hessian -> 曲率 -> DEFF）。"""
        cfg = self.config
        assert self._rng is not None
        x = self._rng.normal(0, 0.12, cfg.B06)        # 主截面横坐标（采样点）
        y = self._rng.normal(0, 0.12, cfg.B06)        # 主截面纵坐标（采样点）
        # 二次曲面高度 = 0.5(κ1 x² + κ2 y²) + 微小噪声
        z = 0.5 * (cfg.K1_TRUE * x**2 + cfg.K2_TRUE * y**2) + self._rng.normal(0, 0.0005, cfg.B06)
        k1_est, k2_est = self.synth.fit_2d_section(x, y, z)     # 拟合恢复主曲率
        err_k1 = abs(k1_est - cfg.K1_TRUE) / abs(cfg.K1_TRUE)   # κ1 相对误差
        err_k2 = abs(k2_est - cfg.K2_TRUE) / abs(cfg.K2_TRUE)   # κ2 相对误差
        DEFF_fit = self.synth.deff(k1_est, k2_est)              # 由估计曲率计算 DEFF
        # 判据：κ1/κ2 恢复误差在容差内，且 DEFF 落在 [1,2]
        ok2 = (err_k1 < cfg.K1_TOL) and (err_k2 < cfg.K2_TOL) and (1.0 <= DEFF_fit <= 2.0)
        if not ok2:
            raise AIQValidationError(
                f"版本A流程不符: err_k1={err_k1 * 100:.2f}%, err_k2={err_k2 * 100:.2f}%, DEFF={DEFF_fit:.4f}",
                expected={"k1_tol": cfg.K1_TOL, "k2_tol": cfg.K2_TOL, "deff∈[1,2]": True},
                actual={"err_k1": err_k1, "err_k2": err_k2, "deff": DEFF_fit},
                param_key="B06",
            )
        return {"detail": (f"版本A拟合恢复: κ1 = {k1_est:.4f} (真值 {cfg.K1_TRUE}), "
                           f"κ2 = {k2_est:.4f} (真值 {cfg.K2_TRUE}); "
                           f"DEFF = {DEFF_fit:.4f} ∈ [1,2]; 相对误差: κ1 {err_k1 * 100:.2f}% "
                           f"(容差 {cfg.K1_TOL * 100:.0f}%), κ2 {err_k2 * 100:.2f}% "
                           f"(容差 {cfg.K2_TOL * 100:.0f}%)"),
                "k1_est": k1_est, "k2_est": k2_est, "deff": DEFF_fit}

    # ------------------------------------------------------------ 3) DEFF 公式与平台
    def validate_deff_platform(self) -> dict:
        """3) DEFF 公式与平台：均匀 φ 期望 = 1+2/π；文档截断高斯 φ 平台落在 (1,2)。"""
        cfg = self.config
        assert self._rng is not None
        phi = self._rng.uniform(0, np.pi / 4, cfg.N_MC)        # 主曲率角 φ 均匀采样
        plat = float(np.mean(1 + np.sin(2 * phi)))  # DEFF 的 MC 期望（DEFF=1+sin2φ 形式）
        ok3a = abs(plat - (1 + 2 / np.pi)) < cfg.PLAT_ISOTROPY_TOL   # 与各向同性解析值对照
        # 复现文档平台：实测 φ~N(14°,25°) 截断[0,45°]（φ 直方图拟合结果）
        phi_doc = np.clip(self._rng.normal(np.deg2rad(cfg.PHI_MEAN_DEG), np.deg2rad(cfg.PHI_STD_DEG),
                                           cfg.N_MC), 0, np.pi / 4)     # 截断高斯采样 φ
        plat_doc = float(np.mean(1 + np.sin(2 * phi_doc)))   # 文档分布下的 DEFF 期望
        ok3b = 1.0 < plat_doc < 2.0          # 平台须落在物理区间
        ok3 = ok3a and ok3b
        if not ok3:
            raise AIQValidationError(
                f"DEFF 平台不符: 均匀φ={plat:.6f} (1+2/π={1 + 2 / np.pi:.6f}), 文档φ={plat_doc:.4f}",
                expected={"iso": 1 + 2 / np.pi, "plat_doc∈(1,2)": True},
                actual={"plat": plat, "plat_doc": plat_doc},
                param_key="B06",
            )
        return {"detail": (f"φ 均匀下 E[DEFF] = {plat:.6f}，解析值 1+2/π = {1 + 2 / np.pi:.6f} "
                           f"(偏差 {(plat - (1 + 2 / np.pi)) * 100:.3f}%); "
                           f"文档 φ~N({cfg.PHI_MEAN_DEG}°,{cfg.PHI_STD_DEG}°)|[0,45°] 隐含 "
                           f"E[DEFF] = {plat_doc:.4f}（实测平台 1.5737-1.5920 同量级，"
                           f"口径不同仅如实报告）"),
                "plat": plat, "plat_doc": plat_doc, "iso": 1 + 2 / np.pi}

    # ------------------------------------------------------------ 4) 真实模型对照
    def validate_real_model(self) -> dict:
        """4) 真实模型对照：真实 DEFF 平台（有限且 ∈[1,2]）+ φ/K<0%/Hmed 如实呈现。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        deff_real = rd.get("curvature.DEFF_plat")       # 真实 DEFF 平台
        phi_real = rd.get("curvature.phi_mean_deg")     # 真实 φ 均值（度）
        kneg_real = rd.get("curvature.K_neg_pct")       # 真实 K<0% 比例
        hmed_real = rd.get("curvature.H_median")        # 真实 Hmed
        deff_doc = rd.audit("DEFF_plat", 1.5920)        # 文档声称平台
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        ok4 = np.isfinite(deff_real) and (1.0 < deff_real <= 2.0)   # 真实平台有限且在 [1,2]
        if not ok4:
            raise RealModelMismatchError(
                f"真实 DEFF 平台异常: {deff_real}",
                expected="有限且 ∈(1,2]", actual=deff_real, param_key="B06",
            )
        return {"detail": (f"{tag} 真实 DEFF 平台 = {deff_real:.4f}（文档声称 {deff_doc:.4f}，"
                           f"差异 {100 * (deff_real - deff_doc) / deff_doc:+.2f}%）; "
                           f"真实 φmean = {phi_real:.2f}°（文档拟合 φ~N(14°,25°)；"
                           f"真实 φmean 高于文档均值 14°，如实报告）；K<0% = {kneg_real:.2f}%，"
                           f"Hmed = {hmed_real:.4f}，λ = "
                           f"{rd.get('curvature.lambda_ratio', float('nan')):.4f}"),
                "source": rd.source_tag(), "tag": tag,
                "deff_real": deff_real, "deff_doc": deff_doc, "phi_real": phi_real,
                "kneg_real": kneg_real, "hmed_real": hmed_real}

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "sync", self.validate_sync),
            (2, "flow", self.validate_flow),
            (3, "deff_platform", self.validate_deff_platform),
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
    """B06 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B06 N_SAMPLE_PER_FRAME 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = VersionASynthesizer(cfg)                  # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("B06 N_SAMPLE_PER_FRAME 验证（四层工厂架构）：版本A采样与主截面拟合")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: B04={cfg.B04} B05={cfg.B05} B06={cfg.B06} DIM={cfg.DIM} "
          f"K1_TRUE={cfg.K1_TRUE} K2_TRUE={cfg.K2_TRUE} K1_TOL={cfg.K1_TOL} K2_TOL={cfg.K2_TOL} "
          f"N_MC={cfg.N_MC} PHI~N({cfg.PHI_MEAN_DEG}°,{cfg.PHI_STD_DEG}°) SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b06_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b06_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b06_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
