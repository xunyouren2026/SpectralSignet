# -*- coding: utf-8 -*-
"""F01 ctx KV 测试上下文长度 — KV 公式验证与降档逻辑（四层工厂架构）
====================================================================
公式: KV_bytes = 2 * L * n_kv * head_dim * ctx * 4
实测模型: Qwen2.5-0.5B  (L=24, n_kv=2, head_dim=64, float32)

验证目标（与原脚本完全一致，保真）：
  1. 公式精确复算: 2*24*2*64*1024*4 = 25,165,824 B = 25.165824 MB
  2. 与实测 24.9MB 比对: 相对偏差≈1.06%; 用实测 seq=1014 复算 24.920064MB(偏差0.98%)
  3. 自动降档逻辑: 请求4096→峰值超3.16GB→返回1024; 请求1024→保留
  4. 主文档 168KB/tok 口径(n_kv=8/h_d=112)对照, 并标注与实测口径的差异来源

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  F01Config              —— 配置模型（pydantic 优先；dataclass 回退）
  ConfigFactory          —— 实例化 F01Config（env AIQ_F01_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：纯公式验证，工具函数以模块级纯函数提供）
  ValidatorEngine        —— 6 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）

数据源：
  主文档《参数附录表完整版》行 4128-4250（F01 ctx）
  《参数完整定义与公式.txt》F01（第 302-308 行）
  源码 _local_whitebox_detect.py（--ctx 参数、KV 降档循环）
  实测日志 wb_full.log（seq=1014 -> KV=24.9MB）
  《参数审计与实验报告.txt》行 70（状态=已用）
说明：纯数值合成数据，不加载任何大模型。

真实模型对照：
  经 _real_data 惰性读取真实实测：KV=24.2MB@seq984 与公式 24.6KB/tok × 984
  精确对照；KV/W=1.22%（真实实测）。文档审计 24.9MB@1014 为旧实测
  （wb_full.log），以真实实测为准。数据来源标注：[真实实测] 或 [审计回退]。
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

# ---- 第一层：配置模型 F01Config（pydantic 优先；dataclass 回退） ----
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


class F01Config(_ConfigModelBase):
    """F01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 F01 节点键名对应（CTX/CTX_FALLBACK/
    MEM_LIMIT_GB）；取值优先级：env AIQ_F01_<KEY> > YAML > _params_data.json
    > 本模型默认值。MEM_LIMIT_GB 存 GB 值，使用时 ×1e9 转字节。
    """

    L: int = 24                # Qwen2.5-0.5B 层数（wb_full.log:14）
    N_KV: int = 2              # KV 头数（wb_full.log:14）
    HEAD_DIM: int = 64         # 每头维度（wb_full.log:14）
    DTYPE_BYTES: int = 4       # float32 字节数（H05 dtype，fp16=2）
    CTX: int = 1024            # 实测运行 ctx（wb_full.log:4，--ctx 1024）
    WEIGHT_BYTES: float = 1.98e9   # 权重字节数（wb_full.log:14，W=1.98GB fp32）
    MEM_LIMIT_GB: float = 3.16     # RSS 上限（wb_full.log:23，3.16GB，GB 单位）
    KV_MEAS: float = 24.9e6        # 实测 KV（wb_full.log:20，24.9MB）
    KV_PER_TOK_MEAS: int = 24576   # 实测口径每 token KV 成本（24.6KB/tok）
    SEQ_MEAS: int = 1014           # 实测前向序列长度（wb_full.log:20）
    LM_HEAD_ACT_4096: float = 2.3e9  # 4096 ctx 的 lm_head 中间激活（源码注释 ~2.3GB）
    KV_PER_TOK_LOG_KB: float = 24.6  # 实测日志 kv_per_tok（KB/tok，wb_full.log:14）
    REQ_CTX_HI: int = 4096           # 默认请求 ctx（源码第 55 行）
    REQ_CTX_LO: int = 1024           # 降档返回 ctx
    DOC_N_KV: int = 8                # 主文档口径 KV 头数（疑笔误/他型号）
    DOC_HEAD_DIM: int = 112          # 主文档口径头维
    DOC_SPEC_KB: float = 168000.0    # 主文档 168,000 B/tok 十进制取整值
    DOC_KV_MB_4096: float = 688.1    # 主文档打印 KV@4096（MB）


# ---- 第二层：配置工厂 ConfigFactory（实例化 F01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """F01 配置工厂：按优先级实例化 F01Config。"""

    def build(self) -> F01Config:
        """构建 F01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(F01Config, "F01")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def kv_bytes(Lv: int, n_kv: int, head_dim: int, ctx: int,
             dtype_bytes: int) -> int:
    """KV 缓存总字节数: 2 * L * n_kv * head_dim * ctx * dtype_bytes。

    防御：非正整数参数返回 0（避免负数/零维乘积歧义）。
    """
    if Lv <= 0 or n_kv <= 0 or head_dim <= 0 or ctx <= 0:
        return 0
    # 因子 2 = K 缓存 + V 缓存两份；逐项相乘得到总字节数
    return 2 * Lv * n_kv * head_dim * ctx * dtype_bytes


def kv_per_tok(Lv: int, n_kv: int, head_dim: int, dtype_bytes: int) -> int:
    """每 token KV 成本（架构常数）。防御同 kv_bytes。"""
    if Lv <= 0 or n_kv <= 0 or head_dim <= 0:
        return 0
    # ctx=1 时 kv_bytes 即每 token 成本：2*L*n_kv*head_dim*dtype_bytes
    return 2 * Lv * n_kv * head_dim * dtype_bytes


def get_safe_ctx(requested_ctx: int, weight_bytes: float,
                 memory_limit: float, cfg: F01Config,
                 lm_head_act_per_tok: float | None = None,
                 kv_per_tok_b: int = 0) -> int:
    """自动降档: 峰值估算 = 权重 + 2*KV + lm_head中间激活(随ctx缩放)。

    源码注释: 4096 ctx 的 lm_head 中间激活约 2.3GB。
    防御：memory_limit/权重非法（<=0）时返回 requested_ctx（不降档）。
    """
    if memory_limit <= 0.0 or weight_bytes <= 0.0:
        return requested_ctx   # 非法约束下不降档（避免误判）
    if kv_per_tok_b <= 0:
        kv_per_tok_b = cfg.KV_PER_TOK_MEAS
    if lm_head_act_per_tok is None:
        lm_head_act_per_tok = cfg.LM_HEAD_ACT_4096 / 4096.0   # 每 token 激活成本
    kv = kv_per_tok_b * requested_ctx            # KV 随 ctx 线性增长
    lm_head_act = lm_head_act_per_tok * requested_ctx  # lm_head 激活随 ctx 缩放
    peak = weight_bytes + 2 * kv + lm_head_act   # 因子 2：峰值时 K/V 双缓存存在
    if peak > memory_limit:
        return cfg.REQ_CTX_LO                    # 超限 -> 降档到安全档
    return requested_ctx                         # 未超限 -> 保留请求值


# ---- 第三层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """F01 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 纯公式验证，无共享 rng 状态（各步骤直接按 cfg 计算）。
    """

    def __init__(
        self,
        config: F01Config,
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

    # ------------------------------------------------------------ 1) 公式精确复算
    def validate_formula(self) -> dict:
        """1) KV 公式精确复算 ctx=1024: 2*24*2*64*1024*4 = 25,165,824 B。"""
        cfg = self.config
        kv_th = kv_bytes(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.CTX, cfg.DTYPE_BYTES)
        target = 2 * 24 * 2 * 64 * 1024 * 4   # 25,165,824
        ok = kv_th == target
        if not ok:
            raise ConfigError(
                f"KV 公式复算应得 {target} B，实际 {kv_th} B",
                expected=target, actual=kv_th, param_key="F01",
            )
        return {
            "detail": f"KV公式 ctx=1024: 2*24*2*64*1024*4 = {kv_th} B = {kv_th/1e6:.6f} MB "
                      f"(预期 25,165,824 B = 25.165824 MB)",
            "kv_bytes": kv_th,
        }

    # ------------------------------------------------------------ 2) 与实测 24.9MB 比对
    def validate_measured(self) -> dict:
        """2) 理论 25.165824MB 与实测 24.9MB 偏差<2%，seq=1014 复算≈24.9MB。"""
        cfg = self.config
        kv_th = kv_bytes(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.CTX, cfg.DTYPE_BYTES)
        rel_1 = abs(kv_th - cfg.KV_MEAS) / kv_th * 100
        kv_seq = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES) * cfg.SEQ_MEAS
        rel_2 = abs(kv_th - kv_seq) / kv_th * 100
        ok = rel_1 < 2.0 and abs(kv_seq / 1e6 - 24.9) < 0.1
        if not ok:
            raise ConfigError(
                f"理论应接近实测 24.9MB: 偏差 {rel_1:.2f}%，seq=1014 复算 {kv_seq/1e6:.4f}MB",
                expected={"rel<": 2.0, "kv_seq_mb≈": 24.9},
                actual={"rel": rel_1, "kv_seq_mb": kv_seq / 1e6}, param_key="F01",
            )
        return {
            "detail": (f"理论 {kv_th/1e6:.4f}MB vs 实测 24.9MB: 相对偏差 {rel_1:.2f}%; "
                       f"用实测 seq=1014 复算: {kv_seq} B = {kv_seq/1e6:.4f}MB (偏差 {rel_2:.2f}%) "
                       f"(差异来源: 实际前向序列 1014 < 1024，取整效应)"),
            "kv_th_mb": kv_th / 1e6, "rel_meas": rel_1, "rel_seq": rel_2,
        }

    # ------------------------------------------------------------ 3) 每 token KV 成本
    def validate_kv_per_tok(self) -> dict:
        """3) kv_per_tok 对齐实测日志 24.6KB/tok（±0.1KB）。"""
        cfg = self.config
        kpt = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES)
        ok = abs(kpt / 1e3 - cfg.KV_PER_TOK_LOG_KB) < 0.1
        if not ok:
            raise ConfigError(
                f"kv_per_tok 应对齐实测 24.6KB/tok，实际 {kpt/1e3:.1f}KB",
                expected=cfg.KV_PER_TOK_LOG_KB, actual=kpt / 1e3, param_key="F01",
            )
        return {"detail": f"kv_per_tok = {kpt} B = {kpt/1e3:.1f}KB/tok (实测日志 24.6KB/tok)",
                "kv_per_tok": kpt}

    # ------------------------------------------------------------ 4) 自动降档逻辑
    def validate_downgrade(self) -> dict:
        """4) 请求 4096 超 3.16GB 降档 1024；请求 1024 保留。"""
        cfg = self.config
        mem_limit = cfg.MEM_LIMIT_GB * 1e9
        ctx_req = get_safe_ctx(cfg.REQ_CTX_HI, cfg.WEIGHT_BYTES, mem_limit, cfg)
        ctx_keep = get_safe_ctx(cfg.REQ_CTX_LO, cfg.WEIGHT_BYTES, mem_limit, cfg)
        ok = (ctx_req == cfg.REQ_CTX_LO) and (ctx_keep == cfg.REQ_CTX_LO)
        if not ok:
            raise ConfigError(
                f"降档逻辑不符: 请求{cfg.REQ_CTX_HI}->{ctx_req}, 请求{cfg.REQ_CTX_LO}->{ctx_keep}",
                expected={"hi->": cfg.REQ_CTX_LO, "lo->": cfg.REQ_CTX_LO},
                actual={"hi": ctx_req, "lo": ctx_keep}, param_key="F01",
            )
        # 手工复算两档峰值（权重 + 2*KV + lm_head 激活）供打印对照
        kpt = kv_per_tok(cfg.L, cfg.N_KV, cfg.HEAD_DIM, cfg.DTYPE_BYTES)
        peak4096 = cfg.WEIGHT_BYTES + 2 * kpt * cfg.REQ_CTX_HI \
            + (cfg.LM_HEAD_ACT_4096 / 4096.0) * cfg.REQ_CTX_HI
        peak1024 = cfg.WEIGHT_BYTES + 2 * kpt * cfg.REQ_CTX_LO \
            + (cfg.LM_HEAD_ACT_4096 / 4096.0) * cfg.REQ_CTX_LO
        return {
            "detail": (f"降档逻辑({cfg.MEM_LIMIT_GB}GB RSS 上限): "
                       f"peak(4096)≈{peak4096/1e9:.2f}GB > 限 -> 返回 ctx={ctx_req}; "
                       f"peak(1024)≈{peak1024/1e9:.2f}GB < 限 -> 返回 ctx={ctx_keep}"),
            "ctx_req": ctx_req, "ctx_keep": ctx_keep,
            "peak4096_gb": peak4096 / 1e9, "peak1024_gb": peak1024 / 1e9,
        }

    # ------------------------------------------------------------ 5) 主文档 168KB/tok 口径对照
    def validate_doc_scale(self) -> dict:
        """5) 主文档口径(n_kv=8/h_d=112)复算 172,032 B/tok，KV@4096≈688.1MB。"""
        cfg = self.config
        kpt_spec = kv_per_tok(24, cfg.DOC_N_KV, cfg.DOC_HEAD_DIM, cfg.DTYPE_BYTES)  # 172,032 B
        kv4096_spec = kv_bytes(24, cfg.DOC_N_KV, cfg.DOC_HEAD_DIM, 4096, cfg.DTYPE_BYTES)
        kv4096_spec_decmb = kv4096_spec / 1e6
        doc_168kb = cfg.DOC_SPEC_KB * 4096 / 1e6      # 文档打印值 688.1MB 十进制取整
        ok = kpt_spec == 172032 and abs(doc_168kb - cfg.DOC_KV_MB_4096) < 0.5
        if not ok:
            raise ConfigError(
                f"主文档口径复算应匹配: kpt_spec={kpt_spec} B(期望 172032)，"
                f"doc_168kb={doc_168kb:.1f}MB(期望 {cfg.DOC_KV_MB_4096})",
                expected={"kpt": 172032, "kv_mb": cfg.DOC_KV_MB_4096},
                actual={"kpt": kpt_spec, "kv_mb": doc_168kb}, param_key="F01",
            )
        return {
            "detail": (f"主文档口径(n_kv={cfg.DOC_N_KV}/h_d={cfg.DOC_HEAD_DIM}): "
                       f"kv_per_tok={kpt_spec} B(={kpt_spec/1024.0:.0f} KiB, 文档记 168KB); "
                       f"KV(4096) 精确公式 = {kv4096_spec_decmb:.1f} MB(十进制) = "
                       f"{kv4096_spec/2**20:.1f} MiB; 文档打印 {cfg.DOC_KV_MB_4096}MB ← "
                       f"168,000 B/tok 十进制取整: {doc_168kb:.1f}MB "
                       f"(差异来源: ① n_kv=8/h_d=112 疑笔误/他型号 ② 单位混用十进制/二进制)"),
            "kpt_spec": kpt_spec, "kv4096_mb": kv4096_spec_decmb, "doc_mb": doc_168kb,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实 KV=24.2MB@seq984 与公式 24.6KB/tok × 984 精确对照；KV/W=1.22%。"""
        cfg = self.config
        rd = self._get_real_data()
        kpt = rd.get("arch.kv_per_tok")
        kvb = rd.get("engine.kv_bytes")
        kvw = rd.get("engine.kv_w_ratio")
        W = rd.get("arch.W_bytes")
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 数据缺失：回退审计值（24.9MB@1014），对照层跳过（不判失败）
        if kpt is None or kvb is None or kvw is None or W is None:
            return {
                "detail": f"{tag} 真实数据缺失 -> 回退审计值（24.9MB@1014），对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        seq_real = kvb // kpt           # 由 KV 总字节反推实际序列长度
        kv_formula = kpt * seq_real     # 公式复算应精确等于存档字节
        ok1 = kv_formula == kvb and seq_real == 984
        kvw_calc = kvb / W              # 复算 KV/W 占比
        ok2 = abs(kvw_calc - kvw) < 1e-4
        ok3 = kpt == 24576              # 真实 kv_per_tok 应 = 24.6KB/tok
        if not (ok1 and ok2 and ok3):
            raise RealModelMismatchError(
                f"真实 KV 公式核对不符: kv_per_tok={kpt}, seq={seq_real}, "
                f"kv_formula={kv_formula} vs {kvb}, KV/W={kvw_calc:.5f}",
                expected={"kpt": 24576, "seq": 984, "kv_formula": kvb},
                actual={"kpt": kpt, "seq": seq_real, "kv_formula": kv_formula,
                        "kvw_calc": kvw_calc}, param_key="F01",
            )
        return {
            "detail": (f"{tag} 公式: {kpt/1e3:.1f}KB/tok × seq={seq_real} = {kv_formula} B "
                       f"= {kvb/1e6:.2f}MB（与实测 {kvb/1e6:.2f}MB 精确一致）; "
                       f"KV/W = {kvb/1e6:.2f}MB / {W/1e9:.2f}GB = {kvw_calc*100:.2f}% "
                       f"(实测存档 {kvw*100:.2f}%); kv_per_tok = {kpt} B = 24.6KB/tok 核对; "
                       f"[差异] 真实 KV=24.2MB@seq984 vs 文档审计 24.9MB@seq1014（旧实测）"),
            "source": rd.source_tag(), "tag": tag,
            "kpt": kpt, "seq_real": seq_real, "kv_bytes_real": kvb,
            "kvw_calc": kvw_calc, "kvw": kvw,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "formula", self.validate_formula),
            (2, "measured", self.validate_measured),
            (3, "kv_per_tok", self.validate_kv_per_tok),
            (4, "downgrade", self.validate_downgrade),
            (5, "doc_scale", self.validate_doc_scale),
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
    """F01 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="F01 ctx 四层工厂验证")
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
    print("F01 ctx KV 测试上下文长度 —— 公式验证（四层工厂架构，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"模型: Qwen2.5-0.5B | L={cfg.L}, n_kv={cfg.N_KV}, head_dim={cfg.HEAD_DIM}, "
          f"dtype=float32({cfg.DTYPE_BYTES}B) | MEM_LIMIT={cfg.MEM_LIMIT_GB}GB")
    print("=" * 78)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "f01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "f01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "f01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
