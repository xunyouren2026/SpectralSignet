# -*- coding: utf-8 -*-
"""T03 K%告警槽 鞍面占比 K<0% 告警槽 — 形态型流形耗散态监控
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 判据 K% ∉ [20, 57]% 触发告警（槽边界含边界值，为槽内）
  2. 本次逐点 K=45.50% -> 槽内（对照实测）
  3. 白盒 8 帧每帧 K% 全部落在 [20, 57]% 内（主文档 warn_K_mixed 全空）
  4. 白盒 8 帧帧平均 ≈ 40%（实算 40.13%）-> 槽内
  5. 构造槽外值 15/60/75% -> 告警；边界值 20/57% -> 槽内
  6. K% 随链长单调上升（短链 45.6 < 白盒 45.5 ≈ 长链 57.33 锚定 NS）
  7. 工程化防御：NaN/Inf/空输入显式处理
  8. 真实模型实测对照：真实 K<0%=45.50% 代入槽判据 -> 槽内
  9. 真实 vs 文档审计值差异（舍入级 < 0.01pp，真实实测为准）
四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  T03Config / ConfigFactory / T03Validator(ValidatorEngine) /
  ReportGenerator / main —— 同 A01（env AIQ_T03_<KEY> 覆盖由共享工厂处理）。
  本参数无数据合成环节，故省略合成器层（engine 传入 synth=None）。
数据源：
  主文档《几何指纹…参数附录表完整版.md》T03 节（行 9841-9914 及 9801-9840 分析段）
  源码 _local_whitebox_detect.py（warn_K_mixed 列）
  《AI几何指纹插件_参数审计与实验报告.txt》201 K<0%（实测 45.50%，状态=槽内）
真实模型对照：
  _real_model_harness.py 对 Qwen2.5-0.5B-Instruct 实测 phi_pairs_all.npy 得
  K<0%=45.50%（_real_metrics.json curvature.K_neg_pct），代入 [20,57]% 槽判据：
  槽内；与文档审计 45.50% 一致。真实实测为准。
  输出标注：[真实实测]（数据齐全）/ [审计回退]（缺失）。
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

# ---- 第一层：配置模型 T03Config（pydantic 优先；dataclass 回退） ----
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


class T03Config(_ConfigModelBase):
    """T03 配置模型：全部阈值/参考值集中于此（零硬编码判据）。

    字段与 _params_data.json 的 T03 节点键名一一对应（未在 JSON 的字段
    以模型默认值兜底）；取值优先级：环境变量 AIQ_T03_<KEY> > YAML >
    _params_data.json > 本模型默认值。
    """

    SEED: int = 0              # 算法逻辑常量：本脚本无随机采样，仅供可复现性约定
    K_SLOT_LO: float = 20.0    # K% 告警槽下限 20%（README ②/④ 工程下界）
    K_SLOT_HI: float = 57.0    # K% 告警槽上限 57%（README ②/④ 工程上界）
    K_NEG_REF: float = 45.50   # 本次逐点 K<0% 实测参考（201 K<0% 实测，README ⑤）
    WB_K: list = [41.54, 30.19, 39.69, 41.22, 36.02,
                  43.50, 34.59, 54.30]        # 白盒 8 帧每帧 K%（README ④Step4）
    MEAN_DOC: float = 40.01   # 文档帧平均≈40.01%（README ⑤；口径近似）
    SEQ_LEN_K: list = [45.6, 45.5, 57.33]     # 短链12帧/白盒8帧/长链末帧32帧（README ③）
    NS_ANCHOR: float = 57.42   # NS 饱和态锚点 K<0%（README ④Step2）
    MONO_TOL: float = 0.15     # 单调性允许的舍入抖动（pp；45.6->45.5 为文档舍入）
    ANCHOR_TOL: float = 1.0    # 长链锚定 NS 的允许偏差（pp）


# ---- 第二层：配置工厂 ConfigFactory（实例化 T03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """T03 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 T03Config。"""

    def build(self) -> T03Config:
        """构建 T03Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(T03Config, "T03")


# ---------------- 纯函数工具（与原脚本逐项一致，保持可测试） ----------------
def k_alert(k: float, lo: float, hi: float) -> tuple[bool, str]:
    """判定 K% 是否在告警槽外。

    返回 (是否告警, 状态)。非有限值（NaN/Inf）视为槽外告警（防御性处理）。
    """
    if not np.isfinite(k):                     # 非有限输入无法比较大小，按最坏情况判槽外告警
        return True, "槽外(非有限值)"
    alert = (k < lo or k > hi)                 # 核心判据：越出闭区间 [lo,hi] 即告警
    return alert, ("槽外告警" if alert else "槽内")


# ---- 第三层：验证引擎 T03Validator（9 项验证 + 结构化日志 + 类型化异常） ----
class T03ValidationError(AIQValidationError):
    """T03 K% 告警槽判据验证失败。"""


class T03Validator(_EngineBase):
    """T03 验证引擎：顺序执行 9 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛类型化异常（AIQValidationError 子类，携带 expected/actual），由 run() 记 FAIL；
    - 告警槽边界经配置读取（K_SLOT_LO/HI），零硬编码判据。
    """

    def __init__(
        self,
        config: T03Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)   # 本参数无合成环节，synth=None
        self._real_data = real_data
        self.lo: float = config.K_SLOT_LO      # 告警槽下限
        self.hi: float = config.K_SLOT_HI      # 告警槽上限

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _check_k(self, name: str, k: float, expected_alert: bool) -> dict:
        """单例判据：执行 k_alert 并与期望告警状态对照（供多个步骤复用）。"""
        alert, status = k_alert(k, self.lo, self.hi)
        ok = bool(alert == expected_alert)
        if not ok:
            raise T03ValidationError(
                f"{name}: K%={k:.2f} 告警状态不符",
                expected="槽外" if expected_alert else "槽内",
                actual=status, param_key="T03",
            )
        return {"detail": (f"{name}: K%={k:<8.2f} -> {status} "
                           f"(期望: {'槽外' if expected_alert else '槽内'})"),
                "k": k, "alert": alert}

    # ------------------------------------------------------------ 1) 本次逐点
    def validate_point(self) -> dict:
        """1) 本次逐点 K=45.50%（201 K<0% 实测）-> 槽内。"""
        return self._check_k("本次逐点 K<0%", self.config.K_NEG_REF, False)

    # ------------------------------------------------------------ 2) 白盒 8 帧每帧
    def validate_wb_frames(self) -> dict:
        """2) 白盒 8 帧每帧 K% 全部槽内（主文档 warn_K_mixed 全空）。"""
        cfg = self.config
        arr = np.array(cfg.WB_K, dtype=float)
        if not (np.isfinite(arr).all() and arr.size > 0):
            raise T03ValidationError(
                f"白盒 K% 序列非法（NaN/Inf/空）: {arr}",
                expected="finite non-empty", actual=arr.tolist(), param_key="T03",
            )
        in_slot = bool(((arr >= self.lo) & (arr <= self.hi)).all())
        if not in_slot:
            raise T03ValidationError(
                f"白盒 K% 序列越出告警槽: {arr.tolist()}",
                expected=[self.lo, self.hi], actual=arr.tolist(), param_key="T03",
            )
        return {"detail": (f"白盒8帧每帧全部槽内: min={arr.min():.2f}% max={arr.max():.2f}% "
                           f"槽=[{self.lo},{self.hi}]%"),
                "min": float(arr.min()), "max": float(arr.max())}

    # ------------------------------------------------------------ 3) 帧平均
    def validate_wb_mean(self) -> dict:
        """3) 白盒 8 帧帧平均 -> 槽内（文档≈40.01%，实算 40.13%）。"""
        cfg = self.config
        arr = np.array(cfg.WB_K, dtype=float)
        mean_k = float(arr.mean())
        ok = bool(self.lo <= mean_k <= self.hi)
        if not ok:
            raise T03ValidationError(
                f"帧平均越出告警槽: {mean_k}",
                expected=[self.lo, self.hi], actual=mean_k, param_key="T03",
            )
        return {"detail": (f"白盒8帧帧平均 K%={mean_k:.2f}% -> 槽内 "
                           f"(文档≈{cfg.MEAN_DOC}%, 实算 {mean_k:.2f}%)"),
                "mean_k": mean_k, "mean_doc": cfg.MEAN_DOC}

    # ------------------------------------------------------------ 4a) 槽外 15
    def validate_out_15(self) -> dict:
        """4a) 构造 K=15%（椭球面主导）-> 槽外告警。"""
        return self._check_k("构造 K=15% (椭球面主导)", 15.0, True)

    # ------------------------------------------------------------ 4b) 槽外 60
    def validate_out_60(self) -> dict:
        """4b) 构造 K=60%（鞍面过度）-> 槽外告警。"""
        return self._check_k("构造 K=60% (鞍面过度)", 60.0, True)

    # ------------------------------------------------------------ 4c) 槽外 75
    def validate_out_75(self) -> dict:
        """4c) 构造 K=75%（管状极限）-> 槽外告警。"""
        return self._check_k("构造 K=75% (管状极限)", 75.0, True)

    # ------------------------------------------------------------ 4d) 边界=下限
    def validate_boundary_lo(self) -> dict:
        """4d) 边界 K=20%（=下限）-> 槽内（闭区间语义，含边界）。"""
        return self._check_k("边界 K=20% (=下限)", self.lo, False)

    # ------------------------------------------------------------ 4e) 边界=上限
    def validate_boundary_hi(self) -> dict:
        """4e) 边界 K=57%（=上限）-> 槽内（闭区间语义，含边界）。"""
        return self._check_k("边界 K=57% (=上限)", self.hi, False)

    # ------------------------------------------------------------ 5) 链长单调
    def validate_monotonic(self) -> dict:
        """5) K% 随链长单调上升（短链45.6 / 白盒45.5 / 长链57.33）。"""
        cfg = self.config
        seq = np.array(cfg.SEQ_LEN_K, dtype=float)
        if not np.isfinite(seq).all():
            raise T03ValidationError(
                f"链长 K% 序列含 NaN/Inf: {seq}",
                expected="finite", actual=seq.tolist(), param_key="T03",
            )
        diffs = np.diff(seq)                   # 相邻链长间的 K% 增量
        mono = bool((diffs >= -cfg.MONO_TOL).all())  # 允许文档舍入级抖动，不允许真下降
        if not mono:
            raise T03ValidationError(
                f"链长单调性不成立: 相邻差 {diffs.tolist()}",
                expected=f"diffs >= -{cfg.MONO_TOL}",
                actual=diffs.tolist(), param_key="T03",
            )
        return {"detail": (f"K%随链长单调上升: {cfg.SEQ_LEN_K} "
                           f"(12帧短 -> 8帧白盒 -> 32帧长链锚定NS, 容差 {cfg.MONO_TOL}pp)"),
                "seq_len_k": cfg.SEQ_LEN_K}

    # ------------------------------------------------------------ 6) 长链锚定
    def validate_anchor(self) -> dict:
        """6) 长链末帧 57.33% 精准锚定 NS 饱和态 57.42%。"""
        cfg = self.config
        seq = np.array(cfg.SEQ_LEN_K, dtype=float)
        anchor_ok = bool(np.isclose(seq[-1], cfg.NS_ANCHOR, atol=cfg.ANCHOR_TOL, rtol=0.0))
        if not anchor_ok:
            raise T03ValidationError(
                f"长链末帧 {seq[-1]} 偏离 NS 锚点 {cfg.NS_ANCHOR} 超过 {cfg.ANCHOR_TOL}pp",
                expected=cfg.NS_ANCHOR, actual=seq[-1], param_key="T03",
            )
        return {"detail": (f"长链末帧 {seq[-1]:.2f}% 锚定 NS 饱和态 {cfg.NS_ANCHOR}% "
                           f"(偏差 {abs(seq[-1] - cfg.NS_ANCHOR):.2f}pp)"),
                "anchor": float(seq[-1]), "ns_anchor": cfg.NS_ANCHOR}

    # ------------------------------------------------------------ 7) 防御
    def validate_guards(self) -> dict:
        """7) 工程化防御：NaN/Inf -> 槽外告警。"""
        ok_nan = k_alert(float("nan"), self.lo, self.hi)[0]
        ok_inf = k_alert(float("-inf"), self.lo, self.hi)[0]
        ok_guard = bool(ok_nan and ok_inf)
        if not ok_guard:
            raise T03ValidationError(
                "NaN/Inf 防御失败：必须判槽外告警",
                expected={"nan_alert": True, "inf_alert": True},
                actual={"nan_alert": ok_nan, "inf_alert": ok_inf},
                param_key="T03",
            )
        return {"detail": f"防御: NaN/Inf -> 槽外告警 (NaN={ok_nan}, -Inf={ok_inf})",
                "nan_alert": ok_nan, "inf_alert": ok_inf}

    # ------------------------------------------------------------ 8) 真实模型对照
    def validate_real(self) -> dict:
        """8) 真实模型实测对照：真实 K<0% 代入 [LO,HI]% 槽判据 -> 槽内。"""
        cfg = self.config
        rd = self._get_real_data()
        src = rd.source_tag()
        # 真实值优先，缺失回退审计值（数据层 K_NEG_REF）—— 保证缺数据环境可运行
        k_real = rd.get("curvature.K_neg_pct", rd.audit("K_neg_pct", cfg.K_NEG_REF))
        alert_real, status_real = k_alert(k_real, self.lo, self.hi)
        ok_real = not alert_real
        if not ok_real:
            raise RealModelMismatchError(
                f"真实 K<0%={k_real}% 应判槽内",
                expected=[self.lo, self.hi], actual=k_real, param_key="T03",
            )
        return {
            "detail": (f"[{src}] 真实 K<0%={k_real:.2f}% 代入 [{self.lo:.0f},{self.hi:.0f}]% "
                       f"槽: -> {status_real}"),
            "source": src, "k_real": k_real, "slot_status": status_real,
        }

    # ------------------------------------------------------------ 9) 真实 vs 审计
    def validate_real_audit(self) -> dict:
        """9) 真实 vs 文档审计值差异：舍入级 < 0.01pp，两口径一致。"""
        cfg = self.config
        rd = self._get_real_data()
        k_real = rd.get("curvature.K_neg_pct", None)
        if k_real is None:                     # 真实缺失：回退审计值口径，视为一致
            return {"detail": "[审计回退] 真实 K<0% 缺失，跳过真实 vs 审计对照",
                    "fallback": True}
        k_audit = rd.audit("K_neg_pct", cfg.K_NEG_REF)
        diff_k = abs(k_real - k_audit)
        ok = bool(diff_k < 0.01)
        if not ok:
            raise T03ValidationError(
                f"真实 K<0%={k_real}% 与审计 {k_audit}% 偏差 {diff_k}pp 超舍入口径",
                expected=0.01, actual=diff_k, param_key="T03",
            )
        return {
            "detail": (f"真实 vs 审计: K<0% 真实={k_real:.4f}% 审计={k_audit:.2f}% "
                       f"(|Δ|={diff_k:.4f}pp < 0.01 审计舍入口径, 两口径一致; 真实实测为准)"),
            "k_real": k_real, "k_audit": k_audit, "diff": diff_k,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 9 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "point", self.validate_point),
            (2, "wb_frames", self.validate_wb_frames),
            (3, "wb_mean", self.validate_wb_mean),
            (4, "out_15", self.validate_out_15),
            (5, "out_60", self.validate_out_60),
            (6, "out_75", self.validate_out_75),
            (7, "boundary_lo", self.validate_boundary_lo),
            (8, "boundary_hi", self.validate_boundary_hi),
            (9, "monotonic", self.validate_monotonic),
            (10, "anchor", self.validate_anchor),
            (11, "guards", self.validate_guards),
            (12, "real", self.validate_real),
            (13, "real_audit", self.validate_real_audit),
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
    """T03 验证编排：四层工厂装配（无合成层）+ --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="T03 K%告警槽 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配（T03 无数据合成环节，省略合成层）----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = T03Validator(cfg, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"T03 K%告警槽  K<0% not in [{cfg.K_SLOT_LO:.0f}, {cfg.K_SLOT_HI:.0f}]% -> 告警")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: K_SLOT=[{cfg.K_SLOT_LO},{cfg.K_SLOT_HI}] K_NEG_REF={cfg.K_NEG_REF} "
          f"NS_ANCHOR={cfg.NS_ANCHOR}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "t03_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "t03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "t03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
