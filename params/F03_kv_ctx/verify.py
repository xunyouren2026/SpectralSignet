# -*- coding: utf-8 -*-
"""F03 kv_ctx KV 预测上下文字节 — 线性预测公式验证（四层工厂架构）
====================================================================
公式: KV_bytes = kv_per_tok × ctx,  kv_per_tok = 2*L*n_kv*head_dim*dtype_bytes
实测模型: Qwen2.5-0.5B (L=24, n_kv=2, head_dim=64, float32)

验证目标（与原脚本完全一致，保真）：
  1. kv_per_tok = 2*24*2*64*4 = 24,576 B = 24.6KB/tok (与日志一致)
  2. kv_ctx = kv_per_tok × ctx: ctx=1024 → 25.165824MB; 实测 24.9MB(seq=1014 复算)
  3. 线性性: KV(2*ctx) = 2*KV(ctx)
  4. 主文档 168KB/tok 口径对照表(1K/4K/16K/32K/128K), 标注单位差异
  5. KV/W: 实测口径 1.3%

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  F03Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 F03Config（env AIQ_F03_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：纯公式验证，工具函数以模块级纯函数提供）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 4361-4481（F03 kv_ctx）
  《参数完整定义与公式.txt》F03（第 318-324 行）
  源码 _local_whitebox_detect.py（第 73 行：kv_per_tok）
  实测日志 wb_full.log（KV/每token=24.6KB；KV=24.9MB@1024；KV/W=1.3%）
  《参数审计与实验报告.txt》行 72（状态=已用 24.6KB/tok）
说明：纯数值合成数据，不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实 kv_per_tok=24576B 精确核对，并用真实 KV 存档
  （24.2MB@seq984）验证线性公式与 KV/W=1.22%。文档审计 24.9MB@1014 为
  旧实测，真实实测为准。数据来源标注：[真实实测] 或 [审计回退]。
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

# ---- 第一层：配置模型 F03Config（pydantic 优先；dataclass 回退） ----
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


class F03Config(_ConfigModelBase):
    """F03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 F03 节点键名对应（KV_PER_TOK/CTX）；
    取值优先级：env AIQ_F03_<KEY> > YAML > _params_data.json > 本模型默认值。
    CTXS_DOC/DOC_VALS 为不可变元组（主文档对照表），天然安全。
    """

    L: int = 24                # Qwen2.5-0.5B 层数（wb_full.log:14）
    N_KV: int = 2              # KV 头数（wb_full.log:14）
    HEAD_DIM: int = 64         # 每头维度（wb_full.log:14）
    DTYPE_BYTES: int = 4       # float32 字节数（H05 dtype，fp16=2）
    KV_PER_TOK: int = 24576    # F03: 目标每 token KV 成本（2*24*2*64*4）
    CTX: int = 1024            # F03: 验证基准 ctx
    WEIGHT_BYTES: float = 1.98e9  # 权重字节数（wb_full.log:14）
    KV_MEAS: float = 24.9e6       # 实测 KV@1024（wb_full.log:20）
    SEQ_MEAS: int = 1014          # 实测前向序列长度（wb_full.log:20）
    KV_PER_TOK_LOG_KB: float = 24.6  # 实测日志 kv_per_tok（KB/tok）
    KVW_MEAS_PCT: float = 1.3        # 实测 KV/W 占比（wb_full.log:20）
    DOC_N_KV: int = 8                # 主文档口径 KV 头数（疑笔误/他型号）
    DOC_HEAD_DIM: int = 112          # 主文档口径头维
    DOC_SPEC_KB: float = 168000.0    # 主文档 168,000 B/tok 十进制取整值
    CTXS_DOC: tuple = (1024, 4096, 16384, 32768, 131072)   # 主文档对照表 ctx
    DOC_VALS: tuple = (168.0, 688.0, 2750.0, 5500.0, 22000.0)  # 主文档打印值（MB/GB）
    DOC_MIN_ERR: float = 3.0         # 与文档值的允许最小偏差（%）


# ---- 第二层：配置工厂 ConfigFactory（实例化 F03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """F03 配置工厂：按优先级实例化 F03Config。"""

    def build(self) -> F03Config:
        """构建 F03Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(F03Config, "F03")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def kv_per_tok(Lv: int, n_kv: int, head_dim: int, dtype_bytes: int) -> int:
    """每 token KV 成本: 2 * L * n_kv * head_dim * dtype_bytes。

    防御：非正整数参数返回 0（避免负数/零维乘积歧义）。
    """
    if Lv <= 0 or n_kv <= 0 or head_dim <= 0:
        return 0
    # 因子 2 = K/V 双缓存；ctx=1 时该式即每 token 字节数
    return 2 * Lv * n_kv * head_dim * dtype_bytes


def predict_kv(ctx: int, kpt: int) -> int:
    """KV 预测: kv_ctx = kv_per_tok × ctx。防御：ctx 非正返回 0。"""
    if ctx <= 0:
        return 0
    return kpt * ctx   # 线性模型：KV 与 ctx 成正比（无固定开销）


# ---- 第三层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """F03 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 纯公式验证，无共享 rng 状态。
    """

    def __init__(
        self,
        config: F03Config,
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

    # ------------------------------------------------------------ 1) kv_per_tok
    def validate_kv_per_tok(self) -> dict:
        """1) kv_per_tok = 2*24*2*64*4 = 24,576 B = 24.6KB/tok。"""
        cfg = self.config
        kpt = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES)
        ok = kpt == cfg.KV_PER_TOK and abs(kpt / 1e3 - cfg.KV_PER_TOK_LOG_KB) < 0.1
        if not ok:
            raise ConfigError(
                f"kv_per_tok 应为 {cfg.KV_PER_TOK} B = 24.6KB/tok，实际 {kpt} B",
                expected=cfg.KV_PER_TOK, actual=kpt, param_key="F03",
            )
        return {"detail": f"kv_per_tok = 2*24*2*64*4 = {kpt} B = {kpt/1e3:.1f}KB/tok "
                          f"(实测日志 24.6KB/tok)", "kv_per_tok": kpt}

    # ------------------------------------------------------------ 2) kv_ctx = kv_per_tok × ctx
    def validate_predict(self) -> dict:
        """2) kv_ctx = 25.165824MB、与实测偏差<2%、seq 复算偏差<1%。"""
        cfg = self.config
        kpt = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES)
        kv_pred = predict_kv(cfg.CTX, kpt)                    # 25,165,824
        kv_pred_mb = kv_pred / 1e6
        kv_seq = predict_kv(cfg.SEQ_MEAS, kpt)                # 24,920,064
        rel_pred = abs(kv_pred - cfg.KV_MEAS) / kv_pred * 100
        rel_seq = abs(kv_pred - kv_seq) / kv_pred * 100
        ok = abs(kv_pred_mb - 25.165824) < 1e-3 and rel_pred < 2.0 and rel_seq < 1.0
        if not ok:
            raise ConfigError(
                f"KV 预测应为 25.165824MB、与实测偏差<2%、seq 复算偏差<1%："
                f"pred={kv_pred_mb:.6f}MB, rel_pred={rel_pred:.2f}%, rel_seq={rel_seq:.2f}%",
                expected={"mb": 25.165824, "rel_pred<": 2.0, "rel_seq<": 1.0},
                actual={"mb": kv_pred_mb, "rel_pred": rel_pred, "rel_seq": rel_seq},
                param_key="F03",
            )
        return {
            "detail": (f"kv_ctx = {kpt} × {cfg.CTX} = {kv_pred} B = {kv_pred_mb:.6f} MB; "
                       f"实测 24.9MB: 相对偏差 {rel_pred:.2f}%; "
                       f"用 seq={cfg.SEQ_MEAS} 复算: {kv_seq} B = {kv_seq/1e6:.4f}MB "
                       f"(偏差 {rel_seq:.2f}% ← 序列取整)"),
            "kv_pred_mb": kv_pred_mb, "rel_pred": rel_pred, "rel_seq": rel_seq,
        }

    # ------------------------------------------------------------ 3) 线性性
    def validate_linearity(self) -> dict:
        """3) 线性性: KV(2048) = 2×KV(1024)。"""
        cfg = self.config
        kpt = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES)
        kv1 = predict_kv(2048, kpt)
        kv2 = predict_kv(1024, kpt) * 2
        ok = kv1 == kv2
        if not ok:
            raise ConfigError(
                f"线性性应成立: KV(2048)={kv1}，2×KV(1024)={kv2}",
                expected=kv2, actual=kv1, param_key="F03",
            )
        return {"detail": f"线性性: KV(2048) = {kv1} B = 2×KV(1024) = {kv2} B"}

    # ------------------------------------------------------------ 4) 主文档 168KB/tok 口径对照表
    def validate_doc_table(self) -> dict:
        """4) 主文档口径对照表（1K/4K/16K/32K/128K），误差<3%。"""
        cfg = self.config
        kpt_spec = kv_per_tok(24, cfg.DOC_N_KV, cfg.DOC_HEAD_DIM, cfg.DTYPE_BYTES)  # 172,032 B
        ok4 = True
        rows: list[str] = []
        for c, d in zip(cfg.CTXS_DOC, cfg.DOC_VALS):
            kv_exact = predict_kv(c, kpt_spec)
            kv_decmb = kv_exact / 1e6      # 十进制 MB
            kv_mib = kv_exact / 2 ** 20    # 二进制 MiB
            doc_approx = cfg.DOC_SPEC_KB * c / 1e6   # 文档取整口径复算
            # 文档 1K 行用 MiB 约定(168MiB), 其余行用 168,000B/tok 十进制取整后四舍五入
            err_mib = abs(kv_mib - d) / d * 100 if d != 0.0 else float("inf")
            err_dec = abs(doc_approx - d) / d * 100 if d != 0.0 else float("inf")
            min_err = min(err_mib, err_dec)   # 取两种口径中更接近文档的
            ok4 &= min_err < cfg.DOC_MIN_ERR
            label = f"{c//1024}K" if c >= 1024 else f"{c}"
            rows.append(f"ctx={label}: 精确公式 {kv_decmb:.1f}MB(十进制)/{kv_mib:.1f}MiB | "
                        f"文档 168,000B/tok 取整 {doc_approx:.1f}MB (文档记 {d:.1f})")
        if not ok4:
            raise ConfigError(
                f"主文档口径对照应有档位误差 <{cfg.DOC_MIN_ERR}%",
                expected=cfg.DOC_MIN_ERR, actual="; ".join(rows), param_key="F03",
            )
        return {
            "detail": "; ".join(rows) + (f" -> 文档 1K 用 MiB(168MiB)、其余用 168,000B/tok "
                                         f"取整(4K=688.1MB), 与精确公式 172,032B 差 ~2.4%, 单位混用"),
            "kpt_spec": kpt_spec,
        }

    # ------------------------------------------------------------ 5) KV/W
    def validate_kvw(self) -> dict:
        """5) KV/W @1024(实测口径) ≈ 1.3%（±0.2%）。"""
        cfg = self.config
        kpt = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES)
        kv_1024 = predict_kv(1024, kpt)
        kvw = kv_1024 / cfg.WEIGHT_BYTES * 100
        ok = abs(kvw - cfg.KVW_MEAS_PCT) < 0.2
        if not ok:
            raise ConfigError(
                f"KV/W 应≈{cfg.KVW_MEAS_PCT}%，实际 {kvw:.2f}%",
                expected=cfg.KVW_MEAS_PCT, actual=kvw, param_key="F03",
            )
        return {
            "detail": (f"KV/W @1024(实测口径) = {kv_1024/1e6:.1f}MB / {cfg.WEIGHT_BYTES/1e9:.2f}GB "
                       f"= {kvw:.2f}% (实测日志 {cfg.KVW_MEAS_PCT}%)"),
            "kvw": kvw,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实 kv_per_tok=24576B 精确核对 + 真实 KV 存档线性公式验证。

        KV = kv_per_tok × seq 精确复现实测 24.18MB@seq984；KV/W=1.22%。
        数据缺失回退且不判失败。
        """
        cfg = self.config
        rd = self._get_real_data()
        kpt = rd.get("arch.kv_per_tok")
        kvb = rd.get("engine.kv_bytes")
        kvw = rd.get("engine.kv_w_ratio")
        W = rd.get("arch.W_bytes")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 数据缺失：回退审计值（24.6KB/tok），对照层跳过（不判失败）
        if kpt is None or kvb is None or kvw is None or W is None:
            return {
                "detail": f"{tag} 真实数据缺失 -> 回退审计值（24.6KB/tok），对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        ok1 = kpt == cfg.KV_PER_TOK          # 真实每 token 成本应精确 = 目标值
        seq_real = kvb // kpt                # 反推实际序列长度
        kv_formula = kpt * seq_real          # 线性公式复算
        ok2 = kv_formula == kvb and seq_real == 984
        kvw_calc = kvb / W                   # 复算 KV/W
        ok3 = abs(kvw_calc - kvw) < 1e-4
        if not (ok1 and ok2 and ok3):
            raise RealModelMismatchError(
                f"真实线性公式核对不符: kv_per_tok={kpt}, seq={seq_real}, "
                f"kv_formula={kv_formula} vs {kvb}, KV/W={kvw_calc:.5f}",
                expected={"kpt": cfg.KV_PER_TOK, "seq": 984, "kv_formula": kvb},
                actual={"kpt": kpt, "seq": seq_real, "kv_formula": kv_formula,
                        "kvw_calc": kvw_calc}, param_key="F03",
            )
        return {
            "detail": (f"{tag} kv_per_tok = {kpt} B = 24.6KB/tok（2*24*2*64*4）; "
                       f"线性公式: {kpt/1e3:.1f}KB/tok × {seq_real} = {kv_formula} B "
                       f"= {kvb/1e6:.2f}MB（实测存档精确一致）; "
                       f"KV/W = {kvb/1e6:.2f}MB / {W/1e9:.2f}GB = {kvw_calc*100:.2f}% "
                       f"(实测存档 {kvw*100:.2f}%); "
                       f"[差异] 真实 KV=24.2MB@seq984 vs 文档审计 24.9MB@seq1014（旧实测）"),
            "source": rd.source_tag(), "tag": tag,
            "kpt": kpt, "seq_real": seq_real, "kv_bytes_real": kvb,
            "kvw_calc": kvw_calc, "kvw": kvw,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "kv_per_tok", self.validate_kv_per_tok),
            (2, "predict", self.validate_predict),
            (3, "linearity", self.validate_linearity),
            (4, "doc_table", self.validate_doc_table),
            (5, "kvw", self.validate_kvw),
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
    """F03 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="F03 kv_ctx 四层工厂验证")
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
    print("F03 kv_ctx KV 预测上下文字节 —— 公式验证（四层工厂架构，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"模型: Qwen2.5-0.5B | L={cfg.L}, n_kv={cfg.N_KV}, head_dim={cfg.HEAD_DIM}, float32")
    print("=" * 78)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "f03_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "f03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "f03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
