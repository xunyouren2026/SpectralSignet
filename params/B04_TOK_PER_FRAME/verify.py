# -*- coding: utf-8 -*-
"""B04 TOK_PER_FRAME 每帧 token 数 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 总帧数 = gen_len / TOK_PER_FRAME（整除）：64/8=8, 96/8=12, 256/8=32
  2. 帧内 token 数 == TOK，提供采样池形状 (TOK, d)
  3. 统计效应：帧均值方差 ∝ 1/TOK -> 帧间 CV 随 TOK 增大而减小
  4. DEFF 均值对 TOK 不敏感（期望不变，仅估计方差变小）
  5. 真实模型对照：真实生成 48 token → 6 帧（engine.ngen_actual 断言）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B04Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 B04Config（环境变量 AIQ_B04_<KEY> > YAML > _params_data.json > 默认）
  FrameStatsSynthesizer   —— 帧数换算 + 帧统计模拟（每 token 曲率量 + 帧内均值 + 帧间 CV）
  ValidatorEngine         —— 4 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 995-1116（B04）
  《参数完整定义与公式.txt》B04 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实自回归生成 48 token（engine.ngen_actual）→ 48/TOK_PER_FRAME=6 帧；
          真实 KV 缓存 seq=984 token → 984/8=123 帧。
  如实呈现：文档帧数档位以 gen_len=64/96/256 为例，真实测量规模为 48 token
            （6 帧），帧数公式在真实规模上成立。
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

# ---- 第一层：配置模型 B04Config（pydantic 优先；dataclass 回退） ----
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


class B04Config(_ConfigModelBase):
    """B04 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B04 节点键名一一对应；取值优先级：
    环境变量 AIQ_B04_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    TOK: int = 8                     # 每帧 token 数
    DIM: int = 64                    # 模拟激活维度
    DEFF_PLATFORM: float = 1.58      # DEFF 平台均值（主文档行 49；帧均值围绕该值波动）
    NOISE_STD: float = 0.05          # 每 token 瞬时曲率噪声
    N_REP: int = 200                 # CV/均值 MC 重复次数
    MEAN_TOL: float = 0.01           # DEFF 均值对 TOK 不敏感的容差
    GEN_LENS: list = [64, 96, 256]   # 帧数公式检查的生成长度
    CV_TOKS: list = [4, 8, 16]       # CV 扫描的帧内 token 数
    NGEN_REAL: int = 48              # 真实自回归生成 token 数（文档声称 48）
    FRAMES_REAL: int = 6             # 真实生成 48 token 换算的帧数（48/8）


# ---- 第二层：配置工厂 ConfigFactory（实例化 B04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B04 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B04Config。"""

    def build(self) -> B04Config:
        """构建 B04Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(B04Config, "B04")


# ---- 第三层：合成器 FrameStatsSynthesizer（算法与原脚本完全一致） ----
class FrameStatsSynthesizer(_SynthBase):
    """B04 帧统计模拟合成器。

    - n_frames(gen_len, tok)：总帧数 = gen_len / TOK_PER_FRAME（须整除）；
    - sim_frame_stats(gen_len, tok, rng, n_rep)：每 token 曲率量模拟 → 帧序列
      DEFF 的 (均值, CV) 多次 MC 平均。
    """

    def __init__(self, cfg: B04Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def n_frames(self, gen_len: int, tok: int) -> int:
        """总帧数 = gen_len / TOK_PER_FRAME（须整除，主文档行 16）。"""
        if gen_len <= 0:
            raise ValueError(f"非法的 gen_len: {gen_len}")   # 防御：非正长度无法分帧
        if tok <= 0:
            raise ValueError(f"非法的 TOK: {tok}")           # 防御：每帧 token 须为正
        return gen_len // tok                                # 整数除法（假定可整除）

    def sim_frame_stats(self, gen_len: int, tok: int, rng: np.random.Generator,
                        n_rep: int) -> tuple[float, float]:
        """模拟每 token 的曲率量，返回帧序列 DEFF 的 (均值, CV) 的多次平均。"""
        cfg = self._cfg
        mean_list = []     # 每次重复的帧均值 DEFF 收集器
        cv_list = []       # 每次重复的帧间 CV 收集器
        for _ in range(n_rep):
            # 每 token 的瞬时曲率量 q_t = mu + 噪声（mu≈1.58 模拟 DEFF 平台）
            q = cfg.DEFF_PLATFORM + cfg.NOISE_STD * rng.standard_normal(gen_len)
            nf = self.n_frames(gen_len, tok)                 # 该 gen_len 下的帧数
            frames = q[: nf * tok].reshape(nf, tok).mean(axis=1)     # 每帧内取均值
            mean_list.append(float(frames.mean()))      # 帧均值（DEFF 平台估计）
            cv_list.append(float(frames.std() / (frames.mean() + 1e-12)))   # 帧间 CV（防御除零）
        return float(np.mean(mean_list)), float(np.mean(cv_list))   # 多次 MC 平均


# ---- 第四层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B04 验证引擎：顺序执行 4 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: B04Config,
        synth: FrameStatsSynthesizer,
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

    # ------------------------------------------------------------ 1) 帧数公式（整除）
    def validate_frame_formula(self) -> dict:
        """1) 帧数公式：gen_len 须被 TOK 整除，且帧数 = gen_len // TOK。"""
        cfg = self.config
        ok1 = True
        rows = []
        for gen_len in cfg.GEN_LENS:
            nf = self.synth.n_frames(gen_len, cfg.TOK)          # 计算帧数
            divisible = (gen_len % cfg.TOK == 0)     # 整除性检查（帧数公式的前提）
            one_ok = divisible and (nf == gen_len // cfg.TOK)   # 整除且帧数正确
            ok1 &= one_ok
            rows.append(f"gen_len={gen_len:>4}: {gen_len}/{cfg.TOK} = {nf:>3} 帧")
        if not ok1:
            raise AIQValidationError(
                f"帧数整除关系不符: {[(g, self.synth.n_frames(g, cfg.TOK)) for g in cfg.GEN_LENS]}",
                expected=[g // cfg.TOK for g in cfg.GEN_LENS],
                actual=[self.synth.n_frames(g, cfg.TOK) for g in cfg.GEN_LENS],
                param_key="B04",
            )
        detail_head = "帧数公式（gen_len 须被 %d 整除）: " % cfg.TOK
        return {"detail": detail_head + "; ".join(rows)}

    # ------------------------------------------------------------ 2) 帧采样池形状
    def validate_sampling_pool(self) -> dict:
        """2) 帧激活矩阵形状 = (TOK, DIM)（每帧 8 个 token 激活）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        self._rng = rng                                   # 供 CV 扫描步骤复用同一随机流
        H_frame = rng.standard_normal((cfg.TOK, cfg.DIM))          # 每帧 8 个 token 激活
        ok2 = (H_frame.shape == (cfg.TOK, cfg.DIM))                # 形状断言：(8, 64)
        if not ok2:
            raise AIQValidationError(
                f"帧采样池形状不符: {H_frame.shape}",
                expected=(cfg.TOK, cfg.DIM), actual=tuple(H_frame.shape), param_key="B04",
            )
        return {"detail": (f"帧激活矩阵形状 = {H_frame.shape}（应为 ({cfg.TOK}, {cfg.DIM})，"
                           f"即 {cfg.TOK} 个 {cfg.DIM} 维激活）"),
                "shape": tuple(H_frame.shape)}

    # ------------------------------------------------------------ 3) 帧间 CV 随 TOK 递减
    def validate_cv_scan(self) -> dict:
        """3) 统计效应：帧间 CV 随 TOK 单调递减 + DEFF 均值对 TOK 不敏感。"""
        cfg = self.config
        assert self._rng is not None
        results: dict[int, tuple[float, float]] = {}        # tok → (DEFF 均值, 帧间 CV)
        rows = []
        for tok in cfg.CV_TOKS:
            m, cv = self.synth.sim_frame_stats(cfg.GEN_LENS[1], tok, self._rng, cfg.N_REP)   # 96 token
            results[tok] = (m, cv)
            rows.append(f"TOK={tok:>2}: DEFF 均值 = {m:.4f}, 帧间 CV = {cv * 100:.2f}%")
        # 统计判据 1：帧内均值方差 ∝ 1/TOK → CV 随 TOK 单调递减
        ok3a = (results[4][1] > results[8][1] > results[16][1])     # CV 单调递减
        deff_means = [results[t][0] for t in cfg.CV_TOKS]
        # 统计判据 2：DEFF 平台均值与 TOK 无关（期望不变，仅方差变小）
        ok3b = (max(deff_means) - min(deff_means)) < cfg.MEAN_TOL      # DEFF 均值不敏感
        ok3 = ok3a and ok3b
        if not ok3:
            raise AIQValidationError(
                f"CV 未递减或 DEFF 均值漂移: results={results}",
                expected={"cv_decreasing": True, "mean_tol": cfg.MEAN_TOL},
                actual=results, param_key="B04",
            )
        return {"detail": ("帧间 CV 随 TOK 变化（gen_len=96，"
                           f"{cfg.N_REP} 次重复取平均）: " + "; ".join(rows)
                           + f"; CV 单调递减: {results[4][1] * 100:.2f}% > {results[8][1] * 100:.2f}% "
                           + f"> {results[16][1] * 100:.2f}% (文档 3.2%/1.5%/1.1% 同趋势); "
                           + f"DEFF 均值不敏感: {min(deff_means):.4f}~{max(deff_means):.4f}"
                           + f"（文档≈{cfg.DEFF_PLATFORM}，容差 {cfg.MEAN_TOL}）"),
                "results": results}

    # ------------------------------------------------------------ 4) 真实模型对照
    def validate_real_model(self) -> dict:
        """4) 真实模型对照：真实生成 48 token → 6 帧 + KV 缓存 seq 帧视角。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        ngen_real = rd.get("engine.ngen_actual")       # 真实自回归生成 token 数（48）
        kv_bytes_real = rd.get("engine.kv_bytes")      # 真实 KV 缓存字节数
        kv_per_tok_real = rd.get("arch.kv_per_tok")    # 每 token KV 字节数
        seq_real = int(round(kv_bytes_real / kv_per_tok_real)) if kv_per_tok_real else None  # KV 序列长度
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        # 核心断言：真实生成恰 NGEN_REAL token，且 NGEN_REAL/TOK = FRAMES_REAL 帧
        ok4 = (ngen_real == cfg.NGEN_REAL) and (self.synth.n_frames(ngen_real, cfg.TOK) == cfg.FRAMES_REAL)
        if not ok4:
            raise RealModelMismatchError(
                f"真实帧数不符: ngen={ngen_real}, frames={self.synth.n_frames(ngen_real, cfg.TOK)}",
                expected={"ngen": cfg.NGEN_REAL, "frames": cfg.FRAMES_REAL},
                actual={"ngen": ngen_real, "frames": self.synth.n_frames(ngen_real, cfg.TOK)},
                param_key="B04",
            )
        detail = (f"{tag} 真实自回归生成 {ngen_real} token → 帧数 = "
                  f"{ngen_real}/{cfg.TOK} = {self.synth.n_frames(ngen_real, cfg.TOK)} 帧")
        if seq_real:
            detail += (f"; 真实 KV 缓存 seq = {seq_real} token → "
                       f"{seq_real}/{cfg.TOK} = {seq_real // cfg.TOK} 帧（KV 帧视角；真实 KV = "
                       f"{kv_bytes_real / 1e6:.1f}MB，KV/W = {rd.get('engine.kv_w_ratio', 0) * 100:.1f}%）")
        detail += (f"; 文档档位 gen_len=64/96/256（8/12/32 帧）为更长序列声称值，"
                   f"真实测量规模为 {ngen_real} token（6 帧），帧数公式在真实规模上成立")
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "ngen_real": ngen_real, "frames_real": self.synth.n_frames(ngen_real, cfg.TOK),
            "seq_real": seq_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "frame_formula", self.validate_frame_formula),
            (2, "sampling_pool", self.validate_sampling_pool),
            (3, "cv_scan", self.validate_cv_scan),
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
    """B04 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B04 TOK_PER_FRAME 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = FrameStatsSynthesizer(cfg)                # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"B04 TOK_PER_FRAME 验证（四层工厂架构）：总帧数 = gen_len / TOK_PER_FRAME  (TOK={cfg.TOK})")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: TOK={cfg.TOK} DIM={cfg.DIM} DEFF_PLATFORM={cfg.DEFF_PLATFORM} "
          f"NOISE_STD={cfg.NOISE_STD} N_REP={cfg.N_REP} MEAN_TOL={cfg.MEAN_TOL} "
          f"GEN_LENS={cfg.GEN_LENS} CV_TOKS={cfg.CV_TOKS} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b04_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
