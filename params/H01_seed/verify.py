# -*- coding: utf-8 -*-
"""H01 seed 随机种子 — 固定种子下的指纹管道可复现性验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. set_seed(0) 后两次"指纹管道"运行结果逐元素全等（max|Δ| = 0）
  2. 不同种子（0 vs 1）结果不同（证明种子确实控制随机性）
  3. torch 路径（若可用）：torch.manual_seed(0) 后张量生成全等
  4. 模拟指纹管道：随机采样 -> SVD -> Gamma -> 蒙特卡洛直方图
  5. random/numpy 库序列复现 + 真实测量引擎种子可复现性对照
数据源：
  主文档行 5427-5519（H01 seed 章节，README ⑤ 实测值 5507-5516 行）
  源码 plugin.py（set_seed 逻辑）
  《参数审计与实验报告.txt》行 81（状态=已用）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  H01Config / ConfigFactory / H01PipelineSynthesizer / H01Validator /
  ReportGenerator / main —— 同 G01（env AIQ_H01_<KEY> 覆盖由共享工厂处理）。
说明：torch 为可选路径（环境可用时校验、缺失时跳过），主验证走 numpy/random。
      纯数值合成数据 + 真实实测对照，不加载任何大模型。
=====================================================================
"""
import argparse
import io
import os
import random
import sys
import time
from typing import Any

import numpy as np

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, RealModelMismatchError, ReproducibilityError, SynthesisError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

try:
    import torch  # noqa: E402  # torch 随机源（若环境可用）
    TORCH_OK = True
except Exception:
    TORCH_OK = False

RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 H01Config（pydantic 优先；dataclass 回退） ----
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


class H01Config(_ConfigModelBase):
    """H01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 H01 节点键名一一对应；取值优先级：
    环境变量 AIQ_H01_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0              # H01 固定种子默认值（从数据层读取，README ①）
    N_SEQ: int = 1000          # random/numpy/torch 序列长度
    B: int = 256               # 指纹管道激活规模（token 数）
    D: int = 128               # 指纹管道激活规模（隐藏维）
    N_MC: int = 20000          # 蒙特卡洛采样数（模拟 D08）
    EPS: float = 1e-12         # 除零保护小量


# ---- 第二层：配置工厂 ConfigFactory（实例化 H01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """H01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 H01Config。"""

    def build(self) -> H01Config:
        """构建 H01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(H01Config, "H01")


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def set_seed(seed: int) -> None:
    """固定所有随机源（对应插件 H01 的 set_seed 逻辑，README ④ 推导 1）。"""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


# ---- 第三层：合成器 H01PipelineSynthesizer（指纹管道模拟） ----
class H01PipelineSynthesizer(_SynthBase):
    """H01 指纹管道合成器：随机激活 -> SVD Gamma -> MC 直方图（依赖全局随机源）。

    调用前必须 set_seed（README ③/⑤）；返回 [Gamma, hist[:10]/n_mc...] 一维数组。
    """

    def __init__(self, cfg: H01Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def fingerprint_pipeline(self) -> np.ndarray:
        """模拟指纹计算管道：随机激活 -> 中心化 -> SVD Gamma -> MC 直方图。"""
        cfg = self._cfg
        assert cfg.B >= 2 and cfg.D >= 3, f"激活规模非法 B={cfg.B}, d={cfg.D}"
        assert cfg.N_MC > 0, f"MC 采样数非法 n_mc={cfg.N_MC}"
        H = np.random.randn(cfg.B, cfg.D)
        H += 0.3 * np.random.randn(cfg.B, 3) @ np.random.randn(3, cfg.D)
        if not np.isfinite(H).all():
            raise SynthesisError("合成激活含 NaN/Inf", actual=H.shape)
        Hc = H - H.mean(axis=0, keepdims=True)
        S = np.linalg.svd(Hc, compute_uv=False)
        gamma = float(S[:3].sum() / (S.sum() + cfg.EPS))
        phi = np.degrees(np.arctan2(np.abs(np.random.randn(cfg.N_MC)),
                                    np.abs(np.random.randn(cfg.N_MC)) * 0.8 + 1e-9))
        hist, _ = np.histogram(phi, bins=np.arange(0, 46.5, 0.5))
        return np.array([gamma] + (hist[:10] / cfg.N_MC).tolist())


# ---- 第四层：验证引擎 H01Validator（6 项验证 + 结构化日志 + 类型化异常） ----
class H01Validator(_EngineBase):
    """H01 验证引擎：顺序执行 6 项验证（结构化 JSON 日志 + 类型化异常）。"""

    def __init__(
        self,
        config: H01Config,
        synth: H01PipelineSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.r1: np.ndarray | None = None  # seed=0 第一次指纹输出（共享）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 两次运行可复现
    def validate_reproducibility(self) -> dict:
        """1) seed=0 两次独立运行逐元素全等（H01 核心承诺）。"""
        cfg = self.config
        set_seed(cfg.SEED)
        r1 = self.synth.fingerprint_pipeline()
        set_seed(cfg.SEED)
        r2 = self.synth.fingerprint_pipeline()
        assert r1.shape == r2.shape, f"指纹维度不一致 {r1.shape} vs {r2.shape}"
        diff = float(np.abs(r1 - r2).max())
        ok1 = bool(np.array_equal(r1, r2))
        self.r1 = r1
        if not ok1:
            raise ReproducibilityError(
                f"seed={cfg.SEED} 两次运行不一致，max|Δ|={diff:.3e}",
                expected=0.0, actual=diff, param_key="H01",
            )
        return {
            "detail": (f"[1] seed={cfg.SEED} 两次运行 max|Δ| = {diff:.3e} (应为 0); "
                       f"Gamma: run1={r1[0]:.6f}, run2={r2[0]:.6f}"),
            "diff": diff, "gamma": r1[0],
        }

    # ------------------------------------------------------------ 2) 不同种子不同结果
    def validate_seed_diff(self) -> dict:
        """2) 不同种子（0 vs 1）-> 结果不同（种子确实控制随机性）。"""
        cfg = self.config
        assert self.r1 is not None
        set_seed(1)
        r_other = self.synth.fingerprint_pipeline()
        diff12 = float(np.abs(self.r1 - r_other).max())
        ok2 = diff12 > 0.0
        if not ok2:
            raise ReproducibilityError(
                f"seed=0 与 seed=1 结果相同（max|Δ|={diff12:.3e}）",
                expected=">0", actual=diff12, param_key="H01",
            )
        return {"detail": f"[2] seed=0 vs seed=1: max|Δ| = {diff12:.3e} (应 > 0)", "diff": diff12}

    # ------------------------------------------------------------ 3) random 库复现
    def validate_random_lib(self) -> dict:
        """3) 内置 random 库：固定种子序列逐项全等。"""
        cfg = self.config
        random.seed(cfg.SEED)
        a1 = [random.random() for _ in range(cfg.N_SEQ)]
        random.seed(cfg.SEED)
        a2 = [random.random() for _ in range(cfg.N_SEQ)]
        ok3 = a1 == a2
        if not ok3:
            raise ReproducibilityError(
                "random.seed(0) 序列不一致",
                expected="equal", actual="diff", param_key="H01",
            )
        return {"detail": f"[3] random.seed({cfg.SEED}) 序列逐项全等: {ok3}", "ok": ok3}

    # ------------------------------------------------------------ 4) numpy 序列复现
    def validate_numpy(self) -> dict:
        """4) numpy 随机源：固定种子序列逐元素全等。"""
        cfg = self.config
        np.random.seed(cfg.SEED)
        n1 = np.random.rand(cfg.N_SEQ)
        np.random.seed(cfg.SEED)
        n2 = np.random.rand(cfg.N_SEQ)
        ok4 = bool(np.array_equal(n1, n2))
        if not ok4:
            raise ReproducibilityError(
                "np.random.seed(0) 序列不一致",
                expected="equal", actual="diff", param_key="H01",
            )
        return {"detail": f"[4] np.random.seed({cfg.SEED}) 序列逐项全等: {ok4}", "ok": ok4}

    # ------------------------------------------------------------ 5) torch 路径（可选）
    def validate_torch(self) -> dict:
        """5) torch 随机源（若可用）：torch.manual_seed(0) 张量全等。"""
        cfg = self.config
        if not TORCH_OK:
            return {"detail": "[5] torch 不可用，跳过（numpy/random 已验证）",
                    "torch_ok": False, "ok": True}
        torch.manual_seed(cfg.SEED)
        t1 = torch.rand(cfg.N_SEQ)
        torch.manual_seed(cfg.SEED)
        t2 = torch.rand(cfg.N_SEQ)
        ok5 = bool(torch.equal(t1, t2))
        if not ok5:
            raise ReproducibilityError(
                "torch.manual_seed(0) 序列不一致",
                expected="equal", actual="diff", param_key="H01",
            )
        return {"detail": f"[5] torch.manual_seed({cfg.SEED}) 序列逐项全等: {ok5}",
                "torch_ok": True, "ok": ok5}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实测量引擎源码含种子设置 + 真实 JSON 存在（惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        harness_fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "_real_model_harness.py")
        has_seed_code = False
        if os.path.isfile(harness_fp):
            with open(harness_fp, "r", encoding="utf-8", errors="replace") as f:
                _src = f.read()
            has_seed_code = ("torch.manual_seed(0)" in _src
                             and "np.random.seed(0)" in _src)
        has_json = rd.has_real()
        ok6 = has_seed_code and has_json
        if not ok6:
            raise RealModelMismatchError(
                f"真实测量不可复现: harness 源码含种子设置={has_seed_code}, "
                f"真实 JSON 存在={has_json}",
                expected={"has_seed_code": True, "has_json": True},
                actual={"has_seed_code": has_seed_code, "has_json": has_json},
                param_key="H01",
            )
        return {
            "detail": (f"[6] [{rd.source_tag()}] harness 源码含 torch.manual_seed(0)/"
                       f"np.random.seed(0): {has_seed_code}; 真实测量 JSON 存在: {has_json}; "
                       f"实测生成时间: {rd.get('generated_at', 'N/A')}, "
                       f"harness 版本: {rd.get('harness_version', 'N/A')}"),
            "source": rd.source_tag(), "has_seed_code": has_seed_code, "has_json": has_json,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "reproducibility", self.validate_reproducibility),
            (2, "seed_diff", self.validate_seed_diff),
            (3, "random_lib", self.validate_random_lib),
            (4, "numpy", self.validate_numpy),
            (5, "torch", self.validate_torch),
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


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """H01 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="H01 seed 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    synth = H01PipelineSynthesizer(cfg)
    report = ReportGenerator()
    engine = H01Validator(cfg, synth, report, real_data=RD)

    print("=" * 74)
    print(f"H01 seed 随机种子复现性验证（四层工厂架构，seed={cfg.SEED}）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: SEED={cfg.SEED} N_SEQ={cfg.N_SEQ} B={cfg.B} D={cfg.D} N_MC={cfg.N_MC}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "h01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "h01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "h01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
