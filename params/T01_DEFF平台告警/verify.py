# -*- coding: utf-8 -*-
"""T01 DEFF平台告警 DEFF 平台偏离告警 — 结构型几何稳定性监控
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 判据 |DEFF - 1.56| > 0.15 触发告警，DEFF 在 [1.41, 1.71] 内无告警
  2. 本次逐点 DEFF=1.5920 与白盒 8 帧均值 1.5737 均判「平台内」
  3. 白盒 8 帧 DEFF 序列全部落在 [1.41, 1.71] 内
  4. 构造越界值 1.75 / 1.30 触发告警；边界值 1.71 / 1.41 仍平台内（含边界）
  5. 文档偏差 |1.5920 - 1.56| = 0.0320 复算一致
  6. 工程化防御：NaN/Inf 输入视为越界告警
  7. 真实模型实测对照：真实 DEFF=1.5920 代入判据 -> 平台内
  8. 真实 vs 文档审计值差异（舍入级 < 1e-4，真实实测为准）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T01Config / ConfigFactory / T01Validator(ValidatorEngine) /
  ReportGenerator / main —— 同 A01（env AIQ_T01_<KEY> 覆盖由共享工厂处理）。
  本参数无数据合成环节，故省略合成器层（engine 传入 synth=None）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T01 节（行 9653-9709）
  源码 _local_whitebox_detect.py（warn_DEFF_platform 列，abs(df-1.56)>0.15）
  《AI几何指纹插件_参数审计与实验报告.txt》202 DEFF_A（实测 1.5920，状态=平台内）
真实模型对照：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测 phi_pairs_all.npy 得
  DEFF_plat=1.5920（_real_metrics.json curvature.DEFF_plat），代入
  |DEFF-1.56|>0.15 判据：|Δ|=0.0320 平台内；与文档审计 1.5920 一致。
  输出标注：[真实实测]（数据齐全）/ [审计回退]（缺失）。真实实测为准。
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
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 T01Config（pydantic 优先；dataclass 回退） ----
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


class T01Config(_ConfigModelBase):
    """T01 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T01 节点键名一一对应；取值优先级：
    环境变量 AIQ_T01_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0              # 算法逻辑常量：本脚本无随机采样，仅供可复现性约定
    PLATFORM: float = 1.56     # 301 DEFF平台 工程平台值（版本 A，NS/AI 实测折中）
    TOL: float = 0.15          # T01 告警容差（约为平台值 9.6%）
    DEV_DOC: float = 0.0320    # 文档偏差 |1.5920 - 1.56|（README ②/⑤）
    DEFF_POINT: float = 1.5920  # 本次逐点 DEFF 实测参考（202 DEFF_A）
    DEFF_MEAN: float = 1.5737   # 白盒 8 帧均值参考（README ⑤）
    WB8: list = [1.5922, 1.5855, 1.5812, 1.5795,
                 1.5446, 1.5517, 1.5646, 1.5902]  # 白盒 8 帧精确序列


# ---- 第二层：配置工厂 ConfigFactory（实例化 T01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T01Config。"""

    def build(self) -> T01Config:
        """构建 T01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T01Config, "T01")


# ---------------- 算法逻辑常量 ----------------
EPS = 1e-9    # 浮点边界消差（偏差 == tol 视为平台内，含边界）


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def deff_alert(deff: float, platform: float, tol: float) -> tuple[bool, float]:
    """判定 DEFF 是否偏离平台触发告警。

    返回 (是否告警, 偏差)。边界值（偏差 == tol）视为平台内；
    NaN/Inf 输入视为越界告警（防御性处理）。
    """
    if not np.isfinite(deff):                  # 非有限输入按最坏情况越界告警
        return True, float("inf")
    dev = abs(deff - platform)                 # 相对平台值的绝对偏差（核心判据量）
    return (dev > tol + EPS), dev              # 严格大于 tol+EPS 才告警：含边界等价于闭区间


# ---- 第三层：验证引擎 T01Validator（8 项验证 + 结构化日志 + 类型化异常） ----
class T01ValidationError(AIQValidationError):
    """T01 DEFF 平台告警判据验证失败。"""


class T01Validator(_EngineBase):
    """T01 验证引擎：顺序执行 8 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 平台区间 [LO, HI] 由配置 PLATFORM±TOL 推导（零硬编码判据边界）。
    """

    def __init__(
        self,
        config: T01Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)   # 本参数无合成环节，synth=None
        self._real_data = real_data                # 惰性注入
        self.lo: float = config.PLATFORM - config.TOL   # 平台区间下限（推导）
        self.hi: float = config.PLATFORM + config.TOL   # 平台区间上限（推导）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _check_case(self, name: str, deff: float, expected_alert: bool) -> dict:
        """单例判据：执行 deff_alert 并与期望告警状态对照（供多个步骤复用）。"""
        cfg = self.config
        alert, dev = deff_alert(deff, cfg.PLATFORM, cfg.TOL)
        ok = bool(alert == expected_alert)
        if not ok:
            raise T01ValidationError(
                f"{name}: DEFF={deff:.4f} 告警状态不符",
                expected="告警" if expected_alert else "平台内",
                actual="告警" if alert else "平台内",
                param_key="T01",
            )
        status = "告警" if alert else "平台内"
        return {"detail": (f"{name}: DEFF={deff:<8.4f} |DEFF-{cfg.PLATFORM}|={dev:<8.4f} "
                           f"-> {status} (期望: {'告警' if expected_alert else '平台内'})"),
                "deff": deff, "dev": dev, "alert": alert}

    # ------------------------------------------------------------ 1) 本次逐点 DEFF
    def validate_point(self) -> dict:
        """1) 本次逐点 DEFF=1.5920（202 DEFF_A 实测，README ⑤）-> 平台内。"""
        cfg = self.config
        return self._check_case("本次逐点 DEFF", cfg.DEFF_POINT, False)

    # ------------------------------------------------------------ 2) 白盒 8 帧均值
    def validate_wb_mean(self) -> dict:
        """2) 白盒 8 帧均值 DEFF=1.5737（README ⑤）-> 平台内。"""
        cfg = self.config
        return self._check_case("白盒 8 帧均值 DEFF", cfg.DEFF_MEAN, False)

    # ------------------------------------------------------------ 3) 白盒 8 帧序列
    def validate_wb_band(self) -> dict:
        """3) 白盒 8 帧 DEFF 序列全部落在平台区间 [LO, HI] 内。"""
        cfg = self.config
        wb8 = np.array(cfg.WB8, dtype=float)
        if not (np.isfinite(wb8).all() and wb8.size > 0):
            raise T01ValidationError(
                f"白盒序列非法（NaN/Inf/空）: {wb8}",
                expected="finite non-empty", actual=wb8.tolist(), param_key="T01",
            )
        in_band = bool(((wb8 >= self.lo) & (wb8 <= self.hi)).all())
        if not in_band:
            raise T01ValidationError(
                f"白盒序列越出平台区间: {wb8.tolist()}",
                expected=[self.lo, self.hi], actual=wb8.tolist(), param_key="T01",
            )
        cv_check = float(np.std(wb8) / np.mean(wb8) * 100.0)   # 帧间 CV，供 T02 参考
        return {
            "detail": (f"白盒8帧序列全部在平台内: min={wb8.min():.4f} max={wb8.max():.4f} "
                       f"区间=[{self.lo:.2f},{self.hi:.2f}] (8帧均值={wb8.mean():.4f}, "
                       f"供 T02 参考 CV={cv_check:.2f}%)"),
            "min": float(wb8.min()), "max": float(wb8.max()),
            "lo": self.lo, "hi": self.hi, "cv_pct": cv_check,
        }

    # ------------------------------------------------------------ 4a) 超上限告警
    def validate_over_hi(self) -> dict:
        """4a) 构造 DEFF=1.75（超上限）-> 告警。"""
        return self._check_case("构造 DEFF=1.75 (超上限)", 1.75, True)

    # ------------------------------------------------------------ 4b) 低于下限告警
    def validate_under_lo(self) -> dict:
        """4b) 构造 DEFF=1.30（低于下限）-> 告警。"""
        return self._check_case("构造 DEFF=1.30 (低于下限)", 1.30, True)

    # ------------------------------------------------------------ 4c) 边界值=上限
    def validate_boundary_hi(self) -> dict:
        """4c) 边界值 DEFF=1.71（=上限）-> 平台内（含边界）。"""
        return self._check_case("边界值 DEFF=1.71 (=上限)", self.hi, False)

    # ------------------------------------------------------------ 4d) 边界值=下限
    def validate_boundary_lo(self) -> dict:
        """4d) 边界值 DEFF=1.41（=下限）-> 平台内（含边界）。"""
        return self._check_case("边界值 DEFF=1.41 (=下限)", self.lo, False)

    # ------------------------------------------------------------ 5) 文档偏差核对
    def validate_doc_dev(self) -> dict:
        """5) 文档偏差核对 |1.5920-1.56|=0.0320（np.isclose rtol=1e-9）。"""
        cfg = self.config
        dev_doc = abs(cfg.DEFF_POINT - cfg.PLATFORM)   # 从文档实测值重新复算
        ok = bool(np.isclose(dev_doc, cfg.DEV_DOC, rtol=1e-9, atol=0.0))
        if not ok:
            raise T01ValidationError(
                f"文档偏差复算失败: 实得 {dev_doc} != {cfg.DEV_DOC}",
                expected=cfg.DEV_DOC, actual=dev_doc, param_key="T01",
            )
        return {"detail": (f"偏差核对 |{cfg.DEFF_POINT:.4f}-{cfg.PLATFORM}|={dev_doc:.4f} "
                           f"(文档 {cfg.DEV_DOC:.4f}, np.isclose rtol=1e-9)"),
                "dev_doc": dev_doc}

    # ------------------------------------------------------------ 6) 防御
    def validate_guards(self) -> dict:
        """6) 工程化防御：NaN/Inf 输入 -> 越界告警。"""
        ok_nan = deff_alert(float("nan"), self.config.PLATFORM, self.config.TOL)[0]
        ok_inf = deff_alert(float("inf"), self.config.PLATFORM, self.config.TOL)[0]
        ok_guard = bool(ok_nan and ok_inf)
        if not ok_guard:
            raise T01ValidationError(
                "NaN/Inf 防御失败：必须返回告警",
                expected={"nan_alert": True, "inf_alert": True},
                actual={"nan_alert": ok_nan, "inf_alert": ok_inf},
                param_key="T01",
            )
        return {"detail": f"防御: NaN/Inf 输入 -> 告警 (NaN={ok_nan}, Inf={ok_inf})",
                "nan_alert": ok_nan, "inf_alert": ok_inf}

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型实测对照：真实 DEFF 代入判据 -> 平台内（真实实测为准）。"""
        cfg = self.config
        rd = self._get_real_data()
        src = rd.source_tag()
        # 真实值优先；缺失时回退到文档审计值（数据层 DEFF_POINT）—— 保证缺数据环境可运行
        deff_real = rd.get("curvature.DEFF_plat", rd.audit("DEFF_plat", cfg.DEFF_POINT))
        alert_real, dev_real = deff_alert(deff_real, cfg.PLATFORM, cfg.TOL)
        ok_real = not alert_real
        if not ok_real:
            raise RealModelMismatchError(
                f"真实 DEFF={deff_real} 应判平台内（|Δ|={dev_real:.4f} ≤ {cfg.TOL}）",
                expected=cfg.TOL, actual=dev_real, param_key="T01",
            )
        return {
            "detail": (f"[{src}] 真实 DEFF={deff_real:.4f} 代入 |DEFF-{cfg.PLATFORM}|>{cfg.TOL} "
                       f"判据: |Δ|={dev_real:.4f} -> 平台内"),
            "source": src, "deff_real": deff_real, "dev_real": dev_real,
        }

    # ------------------------------------------------------------ 8) 真实 vs 审计
    def validate_real_audit(self) -> dict:
        """8) 真实 vs 文档审计值差异：舍入级 < 1e-4，两口径一致。"""
        cfg = self.config
        rd = self._get_real_data()
        deff_real = rd.get("curvature.DEFF_plat", None)
        if deff_real is None:                  # 真实缺失：回退审计值口径，视为一致
            return {"detail": "[审计回退] 真实 DEFF 缺失，跳过真实 vs 审计对照",
                    "fallback": True}
        deff_audit = rd.audit("DEFF_plat", cfg.DEFF_POINT)
        diff_deff = abs(deff_real - deff_audit)
        ok_diff = bool(diff_deff < 1e-4)
        if not ok_diff:
            raise T01ValidationError(
                f"真实 DEFF={deff_real} 与审计 {deff_audit} 偏差 {diff_deff} 超舍入口径",
                expected=1e-4, actual=diff_deff, param_key="T01",
            )
        return {
            "detail": (f"真实 vs 审计: DEFF 真实={deff_real:.6f} 审计={deff_audit:.4f} "
                       f"(|Δ|={diff_deff:.6f} < 1e-4 审计舍入口径, 两口径一致; 真实实测为准)"),
            "deff_real": deff_real, "deff_audit": deff_audit, "diff": diff_deff,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 8 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "point", self.validate_point),
            (2, "wb_mean", self.validate_wb_mean),
            (3, "wb_band", self.validate_wb_band),
            (4, "over_hi", self.validate_over_hi),
            (5, "under_lo", self.validate_under_lo),
            (6, "boundary_hi", self.validate_boundary_hi),
            (7, "boundary_lo", self.validate_boundary_lo),
            (8, "doc_dev", self.validate_doc_dev),
            (9, "guards", self.validate_guards),
            (10, "real", self.validate_real),
            (11, "real_audit", self.validate_real_audit),
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
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """T01 验证编排：四层工厂装配（无合成层）+ --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T01 DEFF平台告警 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配（T01 无数据合成环节，省略合成层）----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T01Validator(cfg, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    lo, hi = cfg.PLATFORM - cfg.TOL, cfg.PLATFORM + cfg.TOL
    print("=" * 74)
    print(f"T01 DEFF平台告警  |DEFF-{cfg.PLATFORM}|>{cfg.TOL}   平台区间 [{lo:.2f}, {hi:.2f}]")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: PLATFORM={cfg.PLATFORM} TOL={cfg.TOL} DEFF_POINT={cfg.DEFF_POINT} "
          f"DEFF_MEAN={cfg.DEFF_MEAN}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
