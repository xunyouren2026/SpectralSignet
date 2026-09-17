# -*- coding: utf-8 -*-
"""D10 target_deff — 平台反解目标：反解与未闭环判定验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 目标值完整性：NS=1.5711, AI=1.5920（两者差 1.3%）
  2. 基于 D09 映射（MC 400000）反解：
       1.5711 -> λ*_NS（审计报告：λ*≈0.41-0.52）
       1.5920 -> λ*_AI（E(λ) 单调增，应 λ*_AI > λ*_NS）
  3. 与实测 λ=0.8516 对比：偏差 >15% -> 未闭环（与审计报告 T06 一致）
  4. 目标单调性：target 增大 -> λ* 增大（含极端值 1.50）
  5. 真实模型对照（真实 DEFF=1.5920 反解 λ*_AI≈0.51，与真实 λ=0.8516
     偏差≈40% > 15% -> 未闭环，与审计 T06 一致）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D10Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D10Config（环境变量 AIQ_D10_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            GRID_LAM 单源在 D09，跨节点在 build() 解析）
  LookupSynthesizer      —— λ→E[DEFF] MC 映射合成（各向异性高斯）
  ValidatorEngine        —— 5 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 3074-3207（D10 六步流程）
  《参数审计与实验报告.txt》行 8、15、61、109、140、154（状态=已用）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
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

# ---- 第一层：配置模型 D10Config（pydantic 优先；dataclass 回退） ----
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

        子类 D10Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D10Config(_ConfigModelBase):
    """D10 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（MC_N/CRIT/
    GRID_LAM）在 ConfigFactory.build() 中按来源节点显式解析（D08/D11/D09）。
    """

    MC_N: int = 400000                  # D08 固定采样数（D08.MC_N 同源）
    LAM_MEAS: float = 0.8516            # 实测 λ（审计报告行 15）
    TARGET_NS: float = 1.5711           # NS 平台目标（主文档行 3077）
    TARGET_AI: float = 1.5920           # AI 实测 DEFF（审计报告行 8）
    CRIT: float = 0.15                  # D11 闭环判据阈值（D11.CLOSURE_THR 同源）
    GRID_LAM: tuple = (0.05, 1.0, 40)   # λ 网格（单源在 D09.GRID_LAM）
    LAM_NS_LO: float = 0.30             # λ*_NS 审计区间下界
    LAM_NS_HI: float = 0.60             # λ*_NS 审计区间上界
    LAM_AI_LO: float = 0.35             # λ*_AI 审计区间下界
    LAM_AI_HI: float = 0.70             # λ*_AI 审计区间上界
    EXTREME_LO: float = 1.50            # 单调性测试极端目标下值
    EXTREME_HI: float = 1.62            # 单调性测试极端目标上值
    SEED: int = 7                       # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D10Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D10 配置工厂：实例化 D10Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D10Config:
        cfg = self.build_model(D10Config, "D10")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D08/D11/D09）
        cfg.MC_N = self.get_int("D08", "MC_N", 400000)
        cfg.CRIT = self.get_float("D11", "CLOSURE_THR", 0.15)
        cfg.GRID_LAM = self.get_grid("D09", "GRID_LAM", (0.05, 1.0, 40))
        return cfg


# ---- 第三层：合成器 LookupSynthesizer（算法与原脚本完全一致） ----
class LookupSynthesizer(_SynthBase):
    """D10 合成器：λ→E[DEFF] MC 映射（各向异性高斯 (κ1,κ2) 对）。"""

    def __init__(self, cfg: D10Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def build_lookup(self, seed: int | None = None) -> tuple:
        """D09 映射：λ∈linspace(0.05,1,40) -> E[DEFF]（各向异性高斯 MC）。

        边界防御：MC 结果须全部有限。
        返回 (lam_grid, deff_lookup)。
        """
        cfg = self._cfg
        lam_grid = np.linspace(*cfg.GRID_LAM)   # (start, stop, count) 展开
        rng = np.random.default_rng(self._seed if seed is None else seed)
        deff_lookup = []
        for lam in lam_grid:
            k1 = rng.normal(0.0, 1.0, cfg.MC_N)
            k2 = rng.normal(0.0, lam, cfg.MC_N)
            deff = (np.abs(k1) + np.abs(k2)) ** 2 / (k1 ** 2 + k2 ** 2)
            deff_lookup.append(float(deff.mean()))
        deff_arr = np.array(deff_lookup)
        assert np.all(np.isfinite(deff_arr)), "build_lookup: MC 映射含 NaN/Inf"
        return lam_grid, deff_arr


# ---- 第四层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D10 验证引擎：顺序执行 5 项验证（4 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D10Config,
        synth: LookupSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real 方法内 import）
        self.lam_grid: np.ndarray | None = None   # 共享中间结果（供后续步骤复用）
        self.deff_lookup: np.ndarray | None = None

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _lookup(self, seed: int | None = None) -> tuple:
        """惰性构建 λ→E[DEFF] 映射（首次构建后缓存复用）。"""
        if self.deff_lookup is None:
            self.lam_grid, self.deff_lookup = self.synth.build_lookup(seed)
        return self.lam_grid, self.deff_lookup

    # ------------------------------------------------------------ 1) 目标值完整性
    def validate_targets(self) -> dict:
        """1) 目标值完整性（README ①）。"""
        cfg = self.config
        ok = (np.isclose(cfg.TARGET_NS, 1.5711, rtol=0, atol=1e-12)
              and np.isclose(cfg.TARGET_AI, 1.5920, rtol=0, atol=1e-12))
        if not ok:
            raise AIQValidationError(
                f"目标值不符: NS={cfg.TARGET_NS}, AI={cfg.TARGET_AI}",
                expected={"NS": 1.5711, "AI": 1.5920},
                actual={"NS": cfg.TARGET_NS, "AI": cfg.TARGET_AI}, param_key="D10",
            )
        return {
            "detail": f"NS={cfg.TARGET_NS}, AI={cfg.TARGET_AI}, "
                      f"Δ={abs(cfg.TARGET_AI - cfg.TARGET_NS) / cfg.TARGET_NS * 100:.2f}%",
            "target_ns": cfg.TARGET_NS, "target_ai": cfg.TARGET_AI,
        }

    # ------------------------------------------------------------ 2) 反解
    def validate_inversion(self) -> dict:
        """2) 反解 λ*_NS / λ*_AI（README ⑤表第 2-3 行）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup()
        lam_star_ns = float(np.interp(cfg.TARGET_NS, deff_lookup, lam_grid))
        lam_star_ai = float(np.interp(cfg.TARGET_AI, deff_lookup, lam_grid))
        ok = (cfg.LAM_NS_LO < lam_star_ns < cfg.LAM_NS_HI
              and cfg.LAM_AI_LO < lam_star_ai < cfg.LAM_AI_HI
              and lam_star_ai > lam_star_ns)
        if not ok:
            raise AIQValidationError(
                f"反解区间/单调不符: λ*_NS={lam_star_ns:.3f}, λ*_AI={lam_star_ai:.3f}",
                expected={"ns in": [cfg.LAM_NS_LO, cfg.LAM_NS_HI],
                          "ai in": [cfg.LAM_AI_LO, cfg.LAM_AI_HI],
                          "ai > ns": True},
                actual={"lam_ns": lam_star_ns, "lam_ai": lam_star_ai},
                param_key="D10",
            )
        return {
            "detail": f"DEFF={cfg.TARGET_NS}→λ*_NS={lam_star_ns:.3f}; "
                      f"DEFF={cfg.TARGET_AI}→λ*_AI={lam_star_ai:.3f} "
                      f"(审计区间 0.41-0.52; 主文档叙述 0.853/0.910 为示意值)",
            "lam_ns": lam_star_ns, "lam_ai": lam_star_ai,
        }

    # ------------------------------------------------------------ 3) 未闭环
    def validate_closure(self) -> dict:
        """3) 与实测 λ 对比：偏差>15% 未闭环（README ③用途 / 审计 T06）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup()
        lam_star_ns = float(np.interp(cfg.TARGET_NS, deff_lookup, lam_grid))
        lam_star_ai = float(np.interp(cfg.TARGET_AI, deff_lookup, lam_grid))
        dev_ns = abs(lam_star_ns - cfg.LAM_MEAS) / cfg.LAM_MEAS
        dev_ai = abs(lam_star_ai - cfg.LAM_MEAS) / cfg.LAM_MEAS
        ok = (dev_ns > cfg.CRIT) and (dev_ai > cfg.CRIT)
        if not ok:
            raise AIQValidationError(
                f"未闭环偏差不足: NS 偏差={dev_ns * 100:.1f}%, AI 偏差={dev_ai * 100:.1f}%",
                expected={"dev >": cfg.CRIT},
                actual={"dev_ns": dev_ns, "dev_ai": dev_ai}, param_key="D10",
            )
        return {
            "detail": f"NS: 偏差={dev_ns * 100:.1f}%; AI: 偏差={dev_ai * 100:.1f}% "
                      f"(T06 审计: 各向异性高斯模型被否)",
            "dev_ns": dev_ns, "dev_ai": dev_ai,
        }

    # ------------------------------------------------------------ 4) 单调性
    def validate_monotonicity(self) -> dict:
        """4) target 单调性 + 极端值 1.50 未闭环（README ⑤表第 5 行）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup()
        t_test = np.array([cfg.EXTREME_LO, cfg.TARGET_NS, cfg.TARGET_AI, cfg.EXTREME_HI])
        l_test = np.interp(t_test, deff_lookup, lam_grid)
        mono = bool(np.all(np.diff(l_test) > 0))
        lam_150 = float(np.interp(cfg.EXTREME_LO, deff_lookup, lam_grid))
        dev_150 = abs(lam_150 - cfg.LAM_MEAS) / cfg.LAM_MEAS
        ok = mono and (dev_150 > cfg.CRIT)
        if not ok:
            raise AIQValidationError(
                f"单调性/极端值判据不满足: mono={mono}, dev(1.50)={dev_150 * 100:.1f}%",
                expected={"mono": True, "dev >": cfg.CRIT},
                actual={"mono": mono, "dev_150": dev_150}, param_key="D10",
            )
        t_str = ", ".join(f"DEFF={t:.3f}→λ*={l:.3f}" for t, l in zip(t_test, l_test))
        return {
            "detail": f"{t_str}; DEFF={cfg.EXTREME_LO}→λ*={lam_150:.3f} "
                      f"偏差={dev_150 * 100:.1f}%（主文档叙述 0.65/23.7% 示意）",
            "mono": mono, "dev_150": dev_150,
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real(self) -> dict:
        """5) 真实模型对照（真实 DEFF=1.5920、λ=0.8516 代入反解）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        lam_grid, deff_lookup = self._lookup()
        if not rd.has_real():
            # 审计回退：真实数据缺失时回退审计 λ=0.8516
            lam_star_ns = float(np.interp(cfg.TARGET_NS, deff_lookup, lam_grid))
            lam_star_ai = float(np.interp(cfg.TARGET_AI, deff_lookup, lam_grid))
            dev_ns = abs(lam_star_ns - cfg.LAM_MEAS) / cfg.LAM_MEAS
            dev_ai = abs(lam_star_ai - cfg.LAM_MEAS) / cfg.LAM_MEAS
            ok = (dev_ns > cfg.CRIT) and (dev_ai > cfg.CRIT)
            if not ok:
                raise AIQValidationError(
                    f"回退未闭环偏差不足: {dev_ns * 100:.1f}%/{dev_ai * 100:.1f}%",
                    expected=cfg.CRIT,
                    actual={"dev_ns": dev_ns, "dev_ai": dev_ai}, param_key="D10",
                )
            return {"detail": f"NS/AI 偏差={dev_ns * 100:.1f}%/{dev_ai * 100:.1f}%",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        lam_real = rd.get("curvature.lambda_ratio", cfg.LAM_MEAS)
        deff_real = rd.get("curvature.DEFF_plat", cfg.TARGET_AI)
        lam_star_real = float(np.interp(deff_real, deff_lookup, lam_grid))
        dev_real = abs(lam_star_real - lam_real) / lam_real
        ok = dev_real > cfg.CRIT
        if not ok:
            raise RealModelMismatchError(
                f"真实反解未闭环偏差不足: {dev_real * 100:.1f}%",
                expected=cfg.CRIT, actual=dev_real, param_key="D10",
            )
        return {
            "detail": (f"{tag} 真实 DEFF={deff_real:.4f} 反解 λ*_AI={lam_star_real:.3f} "
                       f"vs 实测 λ={lam_real:.4f}, 偏差={dev_real * 100:.1f}% > "
                       f"{cfg.CRIT * 100:.0f}% -> 未闭环（真实 λ/DEFF 同源实测，T06 一致）"),
            "source": rd.source_tag(), "tag": tag,
            "lam_real": lam_real, "deff_real": deff_real,
            "lam_star_real": lam_star_real, "dev_real": dev_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "targets", self.validate_targets),
            (2, "inversion", self.validate_inversion),
            (3, "closure", self.validate_closure),
            (4, "monotonicity", self.validate_monotonicity),
            (5, "real_model", self.validate_real),
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
    """D10 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D10 target_deff 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = LookupSynthesizer(cfg)             # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D10 target_deff 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MC_N={cfg.MC_N} LAM_MEAS={cfg.LAM_MEAS} TARGET_NS={cfg.TARGET_NS} "
          f"TARGET_AI={cfg.TARGET_AI} CRIT={cfg.CRIT} GRID_LAM={cfg.GRID_LAM} "
          f"SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d10_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d10_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d10_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
