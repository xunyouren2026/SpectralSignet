# -*- coding: utf-8 -*-
"""G05 max_ctx 黑盒计时最大上下文 — 黑盒 tok/s 计量的范围护栏验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 黑盒计时扫描只执行 ctx <= max_ctx 的档位，超过跳过
  2. 扫描档位 [512,1024,2048,4096,8192] 与文档 5 档行为一致（8192 跳过）
  3. t(ctx) 单调递增、近似线性（黑盒 tok/s 计量的数据基础）
  4. max_ctx 敏感性：max_ctx=8192 时 8192 档变为执行
  5. 真实模型对照（kv_per_tok / RSS 内存预算推算最大可支撑 ctx，惰性读取）
数据源：
  主文档行 5520-5632（G05 max_ctx 章节，README ⑤ 实测表 5620-5631 行）
  源码 plugin.py（黑盒计时循环）
  《参数审计与实验报告.txt》行 80（状态=理论）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  G05Config / ConfigFactory / G05TimerSynthesizer / G05Validator /
  ReportGenerator / main —— 同 G01（env AIQ_G05_<KEY> 覆盖由共享工厂处理）。
说明：纯数值合成数据（算法逻辑验证）+ 真实实测值对照，不加载任何大模型。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
from typing import Any, Callable

import numpy as np

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, RealModelMismatchError, SynthesisError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 G05Config（pydantic 优先；dataclass 回退） ----
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


class G05Config(_ConfigModelBase):
    """G05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 G05 节点键名一一对应；取值优先级：
    环境变量 AIQ_G05_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                  # 合成计时噪声固定种子（规范 §3 / H01 语义）
    MAX_CTX: int = 4096            # max_ctx 固定值（README ①，DeepSeek V4 上限）
    CTX_SCAN: list = [512, 1024, 2048, 4096, 8192]  # 扫描档位（README ②/⑤）
    TIMER_A: float = 0.3           # 固定开销 a（API 往返/预填充，README ②）
    TIMER_B: float = 0.0019        # 每 token 增量 b（拟合文档斜率，README ②）
    TIMER_NOISE: float = 0.02      # 计时噪声标准差（秒）
    RMSE_TOL: float = 0.1          # 线性拟合 RMSE 阈值（秒）
    MEM_BUDGET_GB: float = 16.0    # 内存预算（16GB 演示档，README ⑤）
    MEM_BUDGET_GB_HI: float = 24.0  # 内存预算（24GB 演示档，README ⑤）
    DOC_TIMES: dict = {512: 1.2, 1024: 2.1, 2048: 4.3, 4096: 8.1}  # 文档基准耗时（秒）


# ---- 第二层：配置工厂 ConfigFactory（实例化 G05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """G05 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 G05Config。"""

    def build(self) -> G05Config:
        """构建 G05Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(G05Config, "G05")


# ---- 第三层：合成器 G05TimerSynthesizer（合成黑盒计时器） ----
class G05TimerSynthesizer(_SynthBase):
    """G05 计时器合成器：生成 t = a + b·ctx + 噪声 的合成 API 计时函数。"""

    def __init__(self, cfg: G05Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def make_timer(self) -> Callable[[int], float]:
        """合成 API 计时函数：t = a + b*ctx + 小噪声（README ② 计时模型）。"""
        cfg = self._cfg
        rng = np.random.default_rng(self._seed)

        def timer(ctx: int) -> float:
            t = cfg.TIMER_A + cfg.TIMER_B * ctx + cfg.TIMER_NOISE * rng.standard_normal()
            if not np.isfinite(t):
                raise SynthesisError(f"ctx={ctx} 计时结果 t={t} 非有限", actual=t)
            return float(t)
        return timer


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def blackbox_benchmark(api_timer: Callable[[int], float], ctx_scan: list, max_ctx: int) -> list:
    """黑盒计时：扫描不同上下文长度，不超过 max_ctx（README ② run(ctx) 公式）。"""
    assert callable(api_timer), "api_timer 需为可调用对象"
    assert isinstance(ctx_scan, list) and len(ctx_scan) > 0, "ctx_scan 为空"
    assert max_ctx > 0, f"max_ctx={max_ctx} 非法"
    results = []
    for ctx in ctx_scan:
        if ctx > max_ctx:
            results.append({"ctx": ctx, "time": None, "skipped": True})
            continue
        t = api_timer(ctx)
        assert np.isfinite(t), f"ctx={ctx} 计时结果 t={t} 非有限"
        results.append({"ctx": ctx, "time": float(t), "skipped": False})
    return results


# ---- 第四层：验证引擎 G05Validator（5 项验证 + 结构化日志 + 类型化异常） ----
class G05Validator(_EngineBase):
    """G05 验证引擎：顺序执行 5 项验证（结构化 JSON 日志 + 类型化异常）。"""

    def __init__(
        self,
        config: G05Config,
        synth: G05TimerSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.results: list | None = None  # 扫描结果（共享中间结果）
        self.times: list | None = None    # 已执行档位耗时
        self.ctxs: list | None = None     # 已执行档位 ctx

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 范围护栏
    def validate_guard(self) -> dict:
        """1) 扫描档位执行/跳过：超过 max_ctx 的档位被护栏跳过。"""
        cfg = self.config
        timer = self.synth.make_timer()
        results = blackbox_benchmark(timer, cfg.CTX_SCAN, cfg.MAX_CTX)
        self.results = results
        expected_skip = {512: False, 1024: False, 2048: False, 4096: False, 8192: True}
        parts = []
        ok1 = True
        for r in results:
            status = "跳过（超过 max_ctx）" if r["skipped"] else f"{r['time']:.2f}s"
            parts.append(f"ctx={r['ctx']} -> {status}")
            ok = (r["skipped"] == expected_skip[r["ctx"]])
            ok1 = ok1 and ok
            if not ok:
                raise AIQValidationError(
                    f"ctx={r['ctx']} 执行/跳过状态 {r['skipped']} ≠ 期望 {expected_skip[r['ctx']]}",
                    expected=expected_skip[r["ctx"]], actual=r["skipped"], param_key="G05",
                )
        return {"detail": "[1] 扫描档位: " + "; ".join(parts), "n_ctx": len(results), "ok": ok1}

    # ------------------------------------------------------------ 2) 单调性与线性拟合
    def validate_monotonic_fit(self) -> dict:
        """2) 计时单调性与线性拟合：t(ctx) 单调递增且线性（RMSE < RMSE_TOL）。"""
        cfg = self.config
        assert self.results is not None
        times = [r["time"] for r in self.results if not r["skipped"]]
        ctxs = [r["ctx"] for r in self.results if not r["skipped"]]
        self.times, self.ctxs = times, ctxs
        assert len(times) >= 2, f"有效计时点过少（{len(times)}）"
        mono = all(b >= a for a, b in zip(times, times[1:]))
        slope, intercept = np.polyfit(ctxs, times, 1)
        pred = intercept + slope * np.array(ctxs)
        rmse = float(np.sqrt(np.mean((np.array(times) - pred) ** 2)))
        ok2 = bool(mono) and rmse < cfg.RMSE_TOL
        if not ok2:
            raise AIQValidationError(
                f"计时非单调递增 {times} 或线性拟合 RMSE={rmse:.4f}s ≥ {cfg.RMSE_TOL}s",
                expected={"mono": True, "rmse": f"<{cfg.RMSE_TOL}"},
                actual={"mono": mono, "rmse": rmse}, param_key="G05",
            )
        tok_per_s = 1.0 / np.mean([t / c for t, c in zip(times, ctxs)])
        return {
            "detail": (f"[2] 耗时序列: {[f'{t:.2f}' for t in times]}s; 单调递增: {mono}; "
                       f"线性拟合 t = {intercept:.3f} + {slope:.6f}*ctx, RMSE={rmse:.4f}s; "
                       f"tok/s 估计: 平均 {tok_per_s:.1f} token/s"),
            "mono": mono, "rmse": rmse, "slope": slope, "intercept": intercept,
            "tok_per_s": float(tok_per_s),
        }

    # ------------------------------------------------------------ 3) 文档对照
    def validate_doc_compare(self) -> dict:
        """3) 与文档实测对照（DeepSeek V4）：展示为主，不参与硬性判定。"""
        cfg = self.config
        assert self.ctxs is not None and self.times is not None
        parts = []
        for c in self.ctxs:
            d = cfg.DOC_TIMES.get(c)
            if d:
                parts.append(f"ctx={c} 文档 {d:.1f}s vs 合成 "
                             f"{self.times[self.ctxs.index(c)]:.2f}s")
        return {
            "detail": "[3] 文档对照: " + "; ".join(parts) + " | 文档中 ctx=8192 同样被跳过，行为一致",
            "ok": True,
        }

    # ------------------------------------------------------------ 4) max_ctx 敏感性
    def validate_sensitivity(self) -> dict:
        """4) max_ctx 敏感性：提高上限到最高档位后应全部执行。"""
        cfg = self.config
        timer = self.synth.make_timer()
        r2 = blackbox_benchmark(timer, cfg.CTX_SCAN, max_ctx=cfg.CTX_SCAN[-1])
        n_run = sum(1 for r in r2 if not r["skipped"])
        ok4 = n_run == len(cfg.CTX_SCAN)
        if not ok4:
            raise AIQValidationError(
                f"max_ctx={cfg.CTX_SCAN[-1]} 时执行 {n_run}/{len(cfg.CTX_SCAN)} 档，应为全部",
                expected=len(cfg.CTX_SCAN), actual=n_run, param_key="G05",
            )
        return {
            "detail": (f"[4] max_ctx={cfg.CTX_SCAN[-1]} 时执行 {n_run}/{len(cfg.CTX_SCAN)} 档"
                       f"（{cfg.CTX_SCAN[-1]} 档执行）"),
            "n_run": n_run, "n_total": len(cfg.CTX_SCAN),
        }

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实 KV 工程数据与内存预算推算最大可支撑 ctx（惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        kv_per_tok = rd.get("arch.kv_per_tok", None)
        kv_bytes = rd.get("engine.kv_bytes", None)
        rss_gb = rd.get("engine.rss_gb", None)
        tag = rd.source_tag()
        if kv_per_tok is None or rss_gb is None:  # 真实数据缺失：审计回退
            return {
                "detail": (f"[5] [{tag}] 真实 KV/RSS 数据不可用，回退文档审计 "
                           f"(KV@1024=24.9MB, RSS=3.16GB) 标注"),
                "source": tag, "real_mode": False,
            }
        kv_seq = int(kv_bytes / kv_per_tok) if kv_bytes else None
        audit_tok = rd.audit("tok_s", 8.39)
        real_tok = rd.get("engine.tok_s", None)
        ctx_max_16 = int((cfg.MEM_BUDGET_GB * 1e9 - rss_gb * 1e9) / kv_per_tok)
        ctx_max_24 = int((cfg.MEM_BUDGET_GB_HI * 1e9 - rss_gb * 1e9) / kv_per_tok)
        ok5 = (ctx_max_16 > cfg.MAX_CTX and ctx_max_24 > cfg.MAX_CTX
               and kv_per_tok > 0)
        if not ok5:
            raise RealModelMismatchError(
                f"真实 KV 数据异常: kv_per_tok={kv_per_tok}, "
                f"ctx_max(16GB)={ctx_max_16}, ctx_max(24GB)={ctx_max_24}",
                expected=f">{cfg.MAX_CTX}", actual=(ctx_max_16, ctx_max_24), param_key="G05",
            )
        w_bytes = rd.get("arch.W_bytes", 1)
        extra = ""
        if real_tok is not None:
            extra = (f"; 真实黑盒 tok/s = {real_tok:.2f} vs 文档审计 {audit_tok:.2f} "
                     f"（Δ={real_tok - audit_tok:+.2f}, 以真实实测为准）")
        return {
            "detail": (f"[5] [{tag}] kv_per_tok = {kv_per_tok} B/token; "
                       f"KV@seq{kv_seq} = {kv_bytes / 1e6:.2f} MB (KV/W = "
                       f"{kv_bytes / w_bytes * 100:.2f}%); 真实 RSS = {rss_gb:.3f} GB "
                       f"(psutil 实测); 内存预算 {cfg.MEM_BUDGET_GB:.0f}GB: 最大可支撑 ctx ≈ "
                       f"({cfg.MEM_BUDGET_GB:.0f}GB - {rss_gb:.2f}GB) / {kv_per_tok / 1e3:.2f}KB "
                       f"= {ctx_max_16:,} token; 内存预算 {cfg.MEM_BUDGET_GB_HI:.0f}GB: "
                       f"最大可支撑 ctx ≈ {ctx_max_24:,} token (远超 max_ctx={cfg.MAX_CTX} 档位上限){extra}"),
            "source": tag, "kv_per_tok": kv_per_tok, "kv_seq": kv_seq,
            "kv_bytes": kv_bytes, "rss_gb": rss_gb, "ctx_max_16": ctx_max_16,
            "ctx_max_24": ctx_max_24,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "guard", self.validate_guard),
            (2, "monotonic_fit", self.validate_monotonic_fit),
            (3, "doc_compare", self.validate_doc_compare),
            (4, "sensitivity", self.validate_sensitivity),
            (5, "real_model", self.validate_real_model),
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


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """G05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="G05 max_ctx 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    synth = G05TimerSynthesizer(cfg)
    report = ReportGenerator()
    engine = G05Validator(cfg, synth, report, real_data=RD)

    print("=" * 74)
    print(f"G05 max_ctx 黑盒计时最大上下文验证（四层工厂架构，max_ctx={cfg.MAX_CTX}）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MAX_CTX={cfg.MAX_CTX} CTX_SCAN={cfg.CTX_SCAN} "
          f"RMSE_TOL={cfg.RMSE_TOL} MEM_BUDGET_GB={cfg.MEM_BUDGET_GB}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "g05_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "g05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "g05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
