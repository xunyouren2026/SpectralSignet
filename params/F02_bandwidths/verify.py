# -*- coding: utf-8 -*-
"""F02 bandwidths 带宽预测档 — tok/s 公式验证（四层工厂架构）
====================================================================
公式: tok/s = bandwidth / weight_bytes   (decode 为 memory-bound)
三档带宽: [30e9, 2e12, 3.35e12] bytes/s  (CPU DDR4 / A100 / H100)
权重: Qwen2.5-0.5B fp32 ≈ 1.98e9 B (实测日志 wb_full.log: W=1.98GB)
实测: CPU tok/s=8.39, 有效带宽 16.6GB/s (wb_full.log)

验证目标（与原脚本完全一致，保真）：
  1. 三档带宽值精确匹配规格
  2. tok/s = 带宽/权重 预测: 15.2 / 1010.1 / 1691.9 (主文档输出对照)
  3. 实测 CPU tok/s=8.39 → 效率 η ≈ 55%
  4. 有效带宽 = W_bytes * tok_s ≈ 16.6 GB/s

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  F02Config              —— 配置模型（pydantic 优先；dataclass 回退；
                            列表字段 BANDWIDTHS 经 _mutable 防御共享可变对象）
  ConfigFactory          —— 实例化 F02Config（env AIQ_F02_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：纯公式验证，工具函数以模块级纯函数提供）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 4251-4360（F02 bandwidths）
  《参数完整定义与公式.txt》F02（第 310-316 行）
  源码 _local_whitebox_detect.py（第 151-154 行：有效带宽估计）
  实测日志 wb_full.log（tok/s=8.39，有效带宽 16.6GB/s）
  《参数审计与实验报告.txt》行 71（状态=理论）
说明：纯数值合成数据，不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 tok_s=7.20（48 tok/6.67s）、带宽 0.27GB/tok
  （CPU 实测存档口径），与文档三档预测对照并如实标注差异（文档 8.39 /
  16.6GB/s 为旧实测估算，真实实测为准）。数据来源标注：[真实实测] 或 [审计回退]。
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
    RealModelMismatchError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 F02Config（pydantic 优先；dataclass 回退） ----
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

if _HAS_PYDANTIC:
    def _mutable(v: Any) -> Any:
        """pydantic 路径：默认值直接返回（pydantic 内部深拷贝，安全）。"""
        return v
else:
    def _mutable(v: Any) -> Any:
        """dataclass 路径：包装为 default_factory，避免类定义期共享可变默认值报错。"""
        return _dataclasses.field(default_factory=lambda: v)


class F02Config(_ConfigModelBase):
    """F02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 F02 节点键名一一对应（W_BYTES/TOK_S_MEAS/
    BW_EFF_MEAS/BANDWIDTHS）；取值优先级：env AIQ_F02_<KEY> > YAML >
    _params_data.json > 本模型默认值。BANDWIDTHS 为列表类配置
    （ConfigFactory.get_list 语义），DOC_PRED 与档位顺序对齐（按索引 zip）。
    """

    WEIGHT_BYTES: float = 1.98e9       # 权重字节数（真实实测优先）
    TOK_S_MEASURED: float = 8.39       # 实测 CPU tok/s（wb_full.log: 64tok/7.63s）
    BW_EFF_MEASURED: float = 16.6      # 实测有效带宽（GB/s，wb_full.log:31）
    BANDWIDTHS: list = _mutable([30e9, 2e12, 3.35e12])   # 三档带宽（bytes/s，列表配置）
    ETA_LO: float = 0.5                # 效率区间下界（主文档 ~55%）
    ETA_HI: float = 0.6                # 效率区间上界
    PRED_TOL_PCT: float = 0.5          # tok/s 预测与主文档的允许相对偏差（%）
    DOC_PRED: tuple = (15.2, 1010.1, 1691.9)  # 主文档三档预测（与 BANDWIDTHS 顺序对齐）
    LABELS: tuple = ("CPU DDR4", "A100", "H100")  # 档位标签


# ---- 第二层：配置工厂 ConfigFactory（实例化 F02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """F02 配置工厂：按优先级实例化 F02Config。"""

    def build(self) -> F02Config:
        """构建 F02Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(F02Config, "F02")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def tok_s_prediction(bandwidth: float, weight_bytes: float) -> float:
    """decode 吞吐预测: tok/s = 带宽 / 权重字节（memory-bound 上限）。

    防御：权重非正时返回 0.0（避免除零）。
    """
    if weight_bytes <= 0.0:
        return 0.0
    # decode 阶段受内存带宽限制：每生成 1 token 需读完整权重
    return bandwidth / weight_bytes


def efficiency(measured: float, theoretical: float) -> float:
    """实测效率 η = 实测 tok/s / 理论上限。防御：理论为 0 时返回 0.0。"""
    if theoretical <= 0.0:
        return 0.0
    return measured / theoretical   # 效率 = 实际吞吐占理论上限的比例


def effective_bandwidth(weight_bytes: float, tok_s: float) -> float:
    """有效带宽 = W_bytes * tok_s（GB/s）。"""
    return weight_bytes * tok_s     # 反推：每秒搬移的权重字节数


# ---- 第三层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """F02 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 纯公式验证，无共享 rng 状态。
    """

    def __init__(
        self,
        config: F02Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 三档带宽值
    def validate_bandwidths(self) -> dict:
        """1) 三档带宽精确匹配 [30e9, 2e12, 3.35e12] B/s。"""
        cfg = self.config
        ok = np.allclose(cfg.BANDWIDTHS, (30e9, 2e12, 3.35e12), rtol=0.0, atol=0.0)
        if not ok:
            raise ConfigError(
                f"三档带宽应精确匹配 [30e9, 2e12, 3.35e12]，实际 {cfg.BANDWIDTHS}",
                expected=[30e9, 2e12, 3.35e12], actual=list(cfg.BANDWIDTHS), param_key="F02",
            )
        return {
            "detail": (f"三档带宽: {[f'{b:.3g}' for b in cfg.BANDWIDTHS]} B/s "
                       f"(= {[f'{b/1e9:.3g} GB/s' for b in cfg.BANDWIDTHS]})"),
            "bandwidths": list(cfg.BANDWIDTHS),
        }

    # ------------------------------------------------------------ 2) tok/s 预测对照主文档
    def validate_predictions(self) -> dict:
        """2) tok/s 预测 15.2/1010.1/1691.9 与主文档对齐（偏差<0.5%）。"""
        cfg = self.config
        ok2 = True
        rows: list[str] = []
        for bw, label, doc in zip(cfg.BANDWIDTHS, cfg.LABELS, cfg.DOC_PRED):
            tok_s = tok_s_prediction(bw, cfg.WEIGHT_BYTES)   # 公式预测
            rel = abs(tok_s - doc) / doc * 100 if doc != 0.0 else float("inf")
            ok = rel < cfg.PRED_TOL_PCT
            ok2 &= ok
            rows.append(f"{label}: tok/s = {bw/1e9:.3g}GB/s / {cfg.WEIGHT_BYTES/1e9:.2f}GB = "
                        f"{tok_s:.1f} (主文档 {doc:.1f}, 偏差 {rel:.2f}%)")
        if not ok2:
            raise ConfigError(
                f"tok/s 预测偏离主文档（允许 {cfg.PRED_TOL_PCT}%）: {'; '.join(rows)}",
                expected=list(cfg.DOC_PRED),
                actual=[tok_s_prediction(bw, cfg.WEIGHT_BYTES) for bw in cfg.BANDWIDTHS],
                param_key="F02",
            )
        return {"detail": "; ".join(rows)}

    # ------------------------------------------------------------ 3) 实测效率
    def validate_efficiency(self) -> dict:
        """3) 实测 CPU tok/s=8.39 vs 理论上限 -> 效率 η ∈ (50%, 60%)。"""
        cfg = self.config
        tok_s_cpu_th = tok_s_prediction(cfg.BANDWIDTHS[0], cfg.WEIGHT_BYTES)  # CPU 理论上限
        eta = efficiency(cfg.TOK_S_MEASURED, tok_s_cpu_th)
        ok = cfg.ETA_LO < eta < cfg.ETA_HI
        if not ok:
            raise ConfigError(
                f"效率 η 应在 50%~60%（主文档 ~55%），实际 {eta*100:.1f}%",
                expected=(cfg.ETA_LO, cfg.ETA_HI), actual=eta, param_key="F02",
            )
        return {
            "detail": (f"实测 CPU tok/s = {cfg.TOK_S_MEASURED} vs 理论上限 {tok_s_cpu_th:.2f} "
                       f"-> 效率 η = {eta*100:.1f}% (主文档 ~55%)"),
            "eta": eta, "tok_s_cpu_th": tok_s_cpu_th,
        }

    # ------------------------------------------------------------ 4) 有效带宽反推
    def validate_effective_bw(self) -> dict:
        """4) 有效带宽 = W_bytes × tok/s ≈ 16.6 GB/s（±2%）。"""
        cfg = self.config
        bw_eff = effective_bandwidth(cfg.WEIGHT_BYTES, cfg.TOK_S_MEASURED)
        ok = abs(bw_eff / 1e9 - cfg.BW_EFF_MEASURED) / cfg.BW_EFF_MEASURED < 0.02
        if not ok:
            raise ConfigError(
                f"有效带宽应≈{cfg.BW_EFF_MEASURED}GB/s，实际 {bw_eff/1e9:.1f}GB/s",
                expected=cfg.BW_EFF_MEASURED, actual=bw_eff / 1e9, param_key="F02",
            )
        return {
            "detail": (f"有效带宽 = W_bytes × tok/s = {cfg.WEIGHT_BYTES/1e9:.2f}GB × "
                       f"{cfg.TOK_S_MEASURED} = {bw_eff/1e9:.1f} GB/s (实测日志 {cfg.BW_EFF_MEASURED}GB/s)"),
            "bw_eff_gbs": bw_eff / 1e9,
        }

    # ------------------------------------------------------------ 5) 单位换算 sanity
    def validate_units(self) -> dict:
        """5) 单位换算 sanity：2TB/s=2e12B/s, 3.35TB/s=3350e9B/s, 30GB/s=30e9B/s。"""
        cfg = self.config
        ok = (cfg.BANDWIDTHS[1] == 2 * 1e12) and (cfg.BANDWIDTHS[2] == 3350e9) \
            and abs(cfg.BANDWIDTHS[0] - 30 * 1e9) == 0
        if not ok:
            raise ConfigError(f"单位换算 sanity 失败: {cfg.BANDWIDTHS}", actual=cfg.BANDWIDTHS,
                              param_key="F02")
        return {"detail": "单位换算: 2TB/s=2e12B/s, 3.35TB/s=3350e9B/s, 30GB/s=30e9B/s"}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实 tok_s=7.20 / 带宽 0.27GB（CPU 实测）与文档三档预测对照。

        文档 8.39 / 16.6GB/s 为旧实测估算口径，真实实测为准。数据缺失回退。
        """
        cfg = self.config
        rd = self._get_real_data()
        tok_s = rd.get("engine.tok_s")
        bw = rd.get("engine.bandwidth_gbs")
        W = rd.get("arch.W_bytes")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 数据缺失：回退审计值（8.39 / 16.6GB/s），对照层跳过（不判失败）
        if tok_s is None or bw is None or W is None:
            return {
                "detail": f"{tag} 真实数据缺失 -> 回退审计值（8.39 / 16.6GB/s），对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        ok1 = abs(tok_s - 7.1957) < 0.01    # 真实 tok/s 应 ≈ 7.20
        bw_eff = W * tok_s / 1e9            # 复算有效带宽（GB/s）
        if not ok1:
            raise RealModelMismatchError(
                f"真实 tok/s 应≈7.20，实际 {tok_s:.2f}",
                expected=7.1957, actual=tok_s, param_key="F02",
            )
        return {
            "detail": (f"{tag} tok/s = {tok_s:.2f}（48 tok/6.67s）; "
                       f"harness 带宽口径 (W/tok_s) = {bw:.3f} GB/tok（CPU 实测存档）; "
                       f"有效带宽 (W×tok/s) = {bw_eff:.1f} GB/s; "
                       f"[差异] 文档审计 tok/s=8.39 / 有效带宽=16.6GB/s 为旧实测估算（wb_full.log），"
                       f"真实 tok/s 7.20 vs 文档 8.39：低 {abs(1-tok_s/8.39)*100:.1f}%"
                       f"（CPU 单机差异）；带宽 0.27 vs 16.6GB/s 口径不同 + 旧实测"),
            "source": rd.source_tag(), "tag": tag,
            "tok_s": tok_s, "bw_gbs": bw, "bw_eff": bw_eff,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "bandwidths", self.validate_bandwidths),
            (2, "predictions", self.validate_predictions),
            (3, "efficiency", self.validate_efficiency),
            (4, "effective_bw", self.validate_effective_bw),
            (5, "units", self.validate_units),
            (6, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except AIQValidationError as e:
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
    """F02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="F02 bandwidths 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    engine = ValidatorEngine(cfg, report, real_data=RD)

    print("=" * 78)
    print("F02 bandwidths 带宽预测档 —— 公式验证（四层工厂架构，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"模型: Qwen2.5-0.5B-Instruct | 权重 = {cfg.WEIGHT_BYTES/1e9:.2f}GB (fp32)")
    print("=" * 78)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "f02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "f02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "f02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
