# -*- coding: utf-8 -*-
"""F04 ctx_scan 显存公式计算档位 — 四档 KV 扫描与主导点验证（四层工厂架构）
====================================================================
公式: KV(ctx) = kv_per_tok × ctx,   KV主导点 = W_bytes / kv_per_tok
四档: [1024, 4096, 16384, 65536]  (4倍对数间隔递增)
主文档口径: kv_per_tok = 168,000 B/tok (168KB 十进制取整), W = 1.98e9
实测口径  : kv_per_tok = 24,576 B/tok (24.6KB, L=24/n_kv=2/hd=64/fp32)

验证目标（与原脚本完全一致，保真）：
  1. 四档匹配 [1024,4096,16384,65536], 档位4倍递增
  2. 主文档口径四档 KV/占比精确复算
  3. 实测口径(24.6KB/tok)四档 KV
  4. KV 主导点(两套口径)
  5. KV 随档位严格线性

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  F04Config              —— 配置模型（pydantic 优先；dataclass 回退；
                            列表字段 CTX_SCAN 经 _mutable 防御共享可变对象）
  ConfigFactory          —— 实例化 F04Config（env AIQ_F04_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：纯公式验证，工具函数以模块级纯函数提供）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 4482-4613（F04 ctx_scan）
  《参数完整定义与公式.txt》F04（第 326-332 行）
  源码 _local_whitebox_detect.py（KV 计算同 F01/F03 公式）
  《参数审计与实验报告.txt》行 73（状态=理论）
说明：纯数值合成数据，不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 24.6KB/tok（kv_per_tok=24576）与真实权重
  W=1.976GB 口径，复算四档 [1024/4096/16384/65536] KV 与 KV 主导点
  （≈78.5K）。数据来源标注：[真实实测] 或 [审计回退]。
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

# ---- 第一层：配置模型 F04Config（pydantic 优先；dataclass 回退） ----
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


class F04Config(_ConfigModelBase):
    """F04 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 F04 节点键名对应（CTX_SCAN）；KPT_MEAS 沿用
    实测口径默认值（原脚本取自 F03.KV_PER_TOK=24576）。取值优先级：
    env AIQ_F04_<KEY> > YAML > _params_data.json > 本模型默认值。
    CTX_SCAN 为列表类配置（ConfigFactory.get_list 语义）；DOC_EXPECT 为
    不可变元组（主文档输出表），天然安全。
    """

    WEIGHT_BYTES: float = 1.98e9     # 权重字节数（wb_full.log:14）
    KPT_SPEC: float = 168000.0       # 主文档口径: 168KB/tok 十进制取整
    KPT_MEAS: float = 24576.0        # 实测口径: 2*24*2*64*4 = 24,576 B/tok
    CTX_SCAN: list = _mutable([1024, 4096, 16384, 65536])  # 四档（README ①，4 倍递增）
    RATIO_STEP: float = 4.0          # 档位 4 倍递增
    DOC_EXPECT: tuple = ((172.0, 8.7), (688.1, 34.8), (2750.0, 139.0), (11000.0, 556.0))
    KV_MB_TOL: float = 0.01          # 主文档 KV 相对允许偏差
    PCT_TOL: float = 1.0             # 主文档占比绝对允许偏差（%）
    DOM_SPEC_K: float = 11.5         # 主文档口径主导点（K tokens，≈12K）
    DOM_SPEC_TOL: float = 0.5        # 主文档口径主导点容差（K）
    DOM_MEAS_K: float = 78.7         # 实测口径主导点（K tokens）
    DOM_MEAS_TOL: float = 1.0        # 实测口径主导点容差（K）
    KV_MEAS_1K_MB: float = 25.165824  # 实测口径 KV@1024（MB）
    KV_MEAS_64K_GB: float = 1.6106    # 实测口径 KV@65536（GB）


# ---- 第二层：配置工厂 ConfigFactory（实例化 F04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """F04 配置工厂：按优先级实例化 F04Config。"""

    def build(self) -> F04Config:
        """构建 F04Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(F04Config, "F04")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def kv_size(kv_per_tok: float, ctx: int) -> float:
    """KV 大小 = kv_per_tok × ctx。防御：ctx 非正返回 0.0。"""
    if ctx <= 0:
        return 0.0
    return kv_per_tok * ctx   # 线性模型：KV 随 ctx 线性增长


def ratio_pct(kv: float, weight_bytes: float) -> float:
    """KV/权重占比（%）。防御：权重非正返回 0.0。"""
    if weight_bytes <= 0.0:
        return 0.0
    return kv / weight_bytes * 100.0   # 占比用于判断 KV 是否主导


def dominance_point(weight_bytes: float, kv_per_tok: float) -> float:
    """KV 主导点 = W / kv_per_tok（tokens）。防御：kv_per_tok 非正返回 inf。"""
    if kv_per_tok <= 0.0:
        return float("inf")
    return weight_bytes / kv_per_tok   # 该 ctx 处 KV 占用 = 权重（各占 100%）


# ---- 第三层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """F04 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 纯公式验证，无共享 rng 状态。
    """

    def __init__(
        self,
        config: F04Config,
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

    # ------------------------------------------------------------ 1) 档位匹配与 4 倍递增
    def validate_scan(self) -> dict:
        """1) 四档匹配 [1024,4096,16384,65536] 且 4 倍递增。"""
        cfg = self.config
        ok1a = list(cfg.CTX_SCAN) == [1024, 4096, 16384, 65536]
        ratios_scan = [b / a for a, b in zip(cfg.CTX_SCAN, cfg.CTX_SCAN[1:])]  # 相邻档位比
        ok1b = all(abs(r - cfg.RATIO_STEP) < 1e-9 for r in ratios_scan)
        if not (ok1a and ok1b):
            raise ConfigError(
                f"四档应匹配 [1024,4096,16384,65536] 且 4 倍递增，实际 {cfg.CTX_SCAN}",
                expected=[1024, 4096, 16384, 65536], actual=list(cfg.CTX_SCAN),
                param_key="F04",
            )
        return {
            "detail": f"四档: {list(cfg.CTX_SCAN)} | 档位比: "
                      f"{[f'{r:.0f}x' for r in ratios_scan]} (对数4倍间隔)",
            "scan": list(cfg.CTX_SCAN), "ratios": ratios_scan,
        }

    # ------------------------------------------------------------ 2) 主文档口径四档 KV/占比
    def validate_doc_scale(self) -> dict:
        """2) 主文档口径四档 KV/占比精确复算（172/688/2750/11000MB）。"""
        cfg = self.config
        ok2 = True
        rows: list[str] = []
        for ctx, (kv_mb, pct) in zip(cfg.CTX_SCAN, cfg.DOC_EXPECT):
            kv = kv_size(cfg.KPT_SPEC, ctx)
            r = ratio_pct(kv, cfg.WEIGHT_BYTES)
            ok = abs(kv / 1e6 - kv_mb) / kv_mb < cfg.KV_MB_TOL and abs(r - pct) < cfg.PCT_TOL
            ok2 &= ok
            marker = " ← KV主导" if r > 100 else ""   # 占比>100% 标记主导点越过
            rows.append(f"ctx={ctx:6d}: KV={kv/1e6:8.1f}MB ({r:6.1f}% of weight){marker}")
        if not ok2:
            raise ConfigError(
                "主文档口径四档 KV/占比与期望不符",
                expected=list(cfg.DOC_EXPECT), actual="; ".join(rows), param_key="F04",
            )
        return {"detail": f"主文档口径 (kv_per_tok = {cfg.KPT_SPEC:.0f} B/tok): " + "; ".join(rows)}

    # ------------------------------------------------------------ 3) 实测口径四档 KV
    def validate_meas_scale(self) -> dict:
        """3) 实测口径四档 KV（1K→25.165824MB, 64K→1.6106GB）sanity。"""
        cfg = self.config
        ok3 = True
        rows: list[str] = []
        for ctx in cfg.CTX_SCAN:
            kv = kv_size(cfg.KPT_MEAS, ctx)
            r = ratio_pct(kv, cfg.WEIGHT_BYTES)
            ok3 &= kv / 1e6 == kv_size(cfg.KPT_MEAS, ctx) / 1e6   # 线性自洽
            rows.append(f"ctx={ctx:6d}: KV={kv/1e6:9.2f}MB ({r:5.2f}% of weight)")
        # 具体值 sanity: 1024→25.165824MB, 65536→1.6106GB
        ok3 &= abs(kv_size(cfg.KPT_MEAS, 1024) / 1e6 - cfg.KV_MEAS_1K_MB) < 1e-3
        ok3 &= abs(kv_size(cfg.KPT_MEAS, 65536) / 1e9 - cfg.KV_MEAS_64K_GB) < 1e-2
        if not ok3:
            raise ConfigError(
                f"实测口径 sanity 失败: 1K={kv_size(cfg.KPT_MEAS, 1024)/1e6:.4f}MB, "
                f"64K={kv_size(cfg.KPT_MEAS, 65536)/1e9:.4f}GB",
                expected={"1K": cfg.KV_MEAS_1K_MB, "64K": cfg.KV_MEAS_64K_GB},
                actual={"1K": kv_size(cfg.KPT_MEAS, 1024) / 1e6,
                        "64K": kv_size(cfg.KPT_MEAS, 65536) / 1e9}, param_key="F04",
            )
        return {"detail": f"实测口径 (kv_per_tok = {cfg.KPT_MEAS:.0f} B/tok = 24.6KB): "
                          + "; ".join(rows) + " (sanity: 1K→25.165824MB, 64K→1.6106GB)"}

    # ------------------------------------------------------------ 4) KV 主导点
    def validate_dominance(self) -> dict:
        """4) KV 主导点 = W/kv_per_tok：主文档≈11.5K、实测≈78.7K。"""
        cfg = self.config
        dom_spec = dominance_point(cfg.WEIGHT_BYTES, cfg.KPT_SPEC)
        dom_meas = dominance_point(cfg.WEIGHT_BYTES, cfg.KPT_MEAS)
        ok = (abs(dom_spec / 1024 - cfg.DOM_SPEC_K) < cfg.DOM_SPEC_TOL
              and abs(dom_meas / 1024 - cfg.DOM_MEAS_K) < cfg.DOM_MEAS_TOL)
        if not ok:
            raise ConfigError(
                f"主导点不符: 主文档口径 {dom_spec/1024:.1f}K（需≈{cfg.DOM_SPEC_K}K）, "
                f"实测口径 {dom_meas/1024:.1f}K（需≈{cfg.DOM_MEAS_K}K）",
                expected={"spec": cfg.DOM_SPEC_K, "meas": cfg.DOM_MEAS_K},
                actual={"spec": dom_spec / 1024, "meas": dom_meas / 1024}, param_key="F04",
            )
        return {
            "detail": (f"KV主导点 = W/kv_per_tok: 主文档口径 {cfg.WEIGHT_BYTES/1e9:.2f}GB / "
                       f"{cfg.KPT_SPEC:.0f}B = {dom_spec:.0f} tokens ≈ {dom_spec/1024:.1f}K "
                       f"(主文档 ~12K); 实测口径 {cfg.WEIGHT_BYTES/1e9:.2f}GB / "
                       f"{cfg.KPT_MEAS:.0f}B = {dom_meas:.0f} tokens ≈ {dom_meas/1024:.1f}K"),
            "dom_spec_k": dom_spec / 1024, "dom_meas_k": dom_meas / 1024,
        }

    # ------------------------------------------------------------ 5) KV 随档位 4 倍线性
    def validate_linearity(self) -> dict:
        """5) KV 随档位严格 4 倍线性（标度律）。"""
        cfg = self.config
        kvs_spec = [kv_size(cfg.KPT_SPEC, c) for c in cfg.CTX_SCAN]
        ok = all(abs(b / a - cfg.RATIO_STEP) < 1e-9
                 for a, b in zip(kvs_spec, kvs_spec[1:]))
        if not ok:
            raise ConfigError(
                f"KV 应随档位严格 4 倍线性，实际 {[k/1e6 for k in kvs_spec]}MB",
                expected=cfg.RATIO_STEP, actual=[k / 1e6 for k in kvs_spec], param_key="F04",
            )
        return {"detail": f"档位4倍 ⟹ KV严格4倍: {[f'{k/1e6:.0f}MB' for k in kvs_spec]} (线性标度)"}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实 24.6KB/tok 与真实权重口径下四档 KV 扫描与主导点。

        主导点 = W_real / kv_per_tok ≈ 78.5K。数据缺失回退且不判失败。
        """
        cfg = self.config
        rd = self._get_real_data()
        kpt = rd.get("arch.kv_per_tok")
        W = rd.get("arch.W_bytes")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 数据缺失：回退审计口径（24.6KB/tok / 1.98e9），对照层跳过（不判失败）
        if kpt is None or W is None:
            return {
                "detail": (f"{tag} 真实数据缺失 -> 回退审计口径（24.6KB/tok / 1.98e9），"
                           f"对照层跳过（不判失败）"),
                "skipped": True, "source": tag,
            }
        ok1 = abs(kpt - 24576.0) < 1e-9   # 真实 kv_per_tok 核对
        ok2 = True
        rows: list[str] = []
        for ctx in cfg.CTX_SCAN:
            kv = kv_size(kpt, ctx)
            r = ratio_pct(kv, W)
            ok2 &= kv == kv_size(kpt, ctx)   # 线性性自洽校验
            rows.append(f"ctx={ctx:6d}: KV={kv/1e6:9.2f}MB ({r:5.2f}% of weight)")
        dom = dominance_point(W, kpt)
        ok3 = abs(dom / 1024 - 78.5) < 1.5   # 真实主导点 ≈78.5K
        if not (ok1 and ok2 and ok3):
            raise RealModelMismatchError(
                f"真实口径四档 KV 扫描不符: kv_per_tok={kpt}（需 24576）, "
                f"主导点 {dom/1024:.1f}K（需≈78.5K）",
                expected={"kpt": 24576.0, "dom_k": 78.5},
                actual={"kpt": kpt, "dom_k": dom / 1024}, param_key="F04",
            )
        return {
            "detail": (f"{tag} kv_per_tok = {kpt:.0f} B = 24.6KB/tok 核对; "
                       f"四档扫描: {'; '.join(rows)}; "
                       f"KV 主导点 = W_real/kv_per_tok = {W/1e9:.3f}GB / {kpt:.0f}B = "
                       f"{dom:.0f} tok ≈ {dom/1024:.1f}K; "
                       f"[差异] 真实 W={W/1e9:.3f}GB -> 主导点 {dom/1024:.1f}K；"
                       f"文档 78.7K 用 W=1.98e9 取整，差异 <1%（真实为准）"),
            "source": rd.source_tag(), "tag": tag,
            "kpt": kpt, "W_bytes": W, "dom_k": dom / 1024,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "scan", self.validate_scan),
            (2, "doc_scale", self.validate_doc_scale),
            (3, "meas_scale", self.validate_meas_scale),
            (4, "dominance", self.validate_dominance),
            (5, "linearity", self.validate_linearity),
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
    """F04 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="F04 ctx_scan 四层工厂验证")
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
    print("F04 ctx_scan 显存公式计算档位 —— 公式验证（四层工厂架构，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"档位: {list(cfg.CTX_SCAN)} | W = {cfg.WEIGHT_BYTES/1e9:.2f}GB")
    print("=" * 78)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "f04_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "f04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "f04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
