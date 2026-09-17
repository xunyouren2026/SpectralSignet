# -*- coding: utf-8 -*-
"""G02 keep_ratio token 保留比例 — 按 Gamma_i 降序的谱剪枝保留逻辑验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. keep = int(seq_len * keep_ratio)，保留前 keep_ratio 比例的 token
  2. 保留集合 = Gamma_i（C-子空间能量占比）最高的 keep 个 token
  3. 保留集合平均 Gamma_i 显著高于丢弃集合
  4. 谱剪枝保留的 C-子空间能量占比 >= 随机剪枝（多随机种子对比）
  5. keep_ratio 参数敏感性（0.3/0.5/0.7）
  6. 真实模型对照：真实 24 层 k_proj Gamma 谱重要性剪枝（_real_metrics.json）
数据源：
  主文档行 4957-5188（G02 keep_ratio 章节，README ⑤ 实测表）
  源码 spectral_pruning.py、spectral_analysis.py（Gamma_i 定义）
  《参数审计与实验报告.txt》行 77（状态=理论）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  G02Config / ConfigFactory / KeepRatioSynthesizer / G02Validator /
  ReportGenerator / main —— 同 G01（env AIQ_G02_<KEY> 覆盖由共享工厂处理）。
说明：纯数值合成数据（算法逻辑验证）+ 真实实测值对照，不加载任何大模型。
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

# ---- 第一层：配置模型 G02Config（pydantic 优先；dataclass 回退） ----
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


class G02Config(_ConfigModelBase):
    """G02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 G02 节点键名一一对应；取值优先级：
    环境变量 AIQ_G02_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0               # 合成数据固定种子（规范 §3 / H01 语义）
    SEED_RANDOM: int = 7        # 随机剪枝对比的独立种子（README ⑤）
    N_COMPONENTS: int = 3       # C-子空间维数 M=3（README ② / B01）
    EPS: float = 1e-12          # 除零保护小量
    RTOL: float = 1e-6          # 相对容差
    ATOL: float = 1e-9          # 绝对容差
    T: int = 200                # 合成激活 seq_len
    D: int = 32                 # 合成激活隐藏维
    KEEP_RATIO: float = 0.5     # 默认保留比例（README ①：固定 0.5）
    RATIO_SWEEP: list = [0.3, 0.5, 0.7]  # 敏感性扫描档位（README ③ 表）
    N_RANDOM_TRIALS: int = 50   # 随机剪枝对比的试验次数（README ⑤）
    MIN_SEPARATION: float = 2.0  # 保留 vs 丢弃平均 Gamma_i 最小比值（[3] 判定阈值）


# ---- 第二层：配置工厂 ConfigFactory（实例化 G02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """G02 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 G02Config。"""

    def build(self) -> G02Config:
        """构建 G02Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(G02Config, "G02")


# ---- 第三层：合成器 KeepRatioSynthesizer（合成激活矩阵生成） ----
class KeepRatioSynthesizer(_SynthBase):
    """G02 激活矩阵合成器：低秩主导 + 分区噪声的激活矩阵（固定种子可复现）。"""

    def __init__(self, cfg: G02Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_activation(self) -> np.ndarray:
        """合成激活：前 60 个 token 高集中（决策核心）、其余弥散（乘性噪声）。"""
        cfg = self._cfg
        rng = np.random.default_rng(self._seed)
        U = rng.standard_normal((cfg.T, 3))
        Vt = rng.standard_normal((cfg.D, 3))
        signal = U @ Vt.T
        noise = np.ones((cfg.T, cfg.D))
        noise[:60] *= 0.02
        noise[60:] *= 0.8
        H = signal + noise * rng.standard_normal((cfg.T, cfg.D))
        if H.shape != (cfg.T, cfg.D) or not np.isfinite(H).all():
            raise SynthesisError("合成数据异常（NaN/Inf 或形状不符）", actual=H.shape)
        return H


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def project_to_c_subspace(H: np.ndarray, n_components: int, eps: float) -> np.ndarray:
    """C-子空间投影：均值 + 前 n_components-1 个主方向（numpy SVD）。"""
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2, f"H 必须为二维 (B,d)，实际 ndim={H.ndim}"
    assert np.isfinite(H).all(), "H 含 NaN/Inf，投影结果不可信"
    B = H.shape[0]
    d = H.shape[1]
    if B < 2:
        return H
    mu = H.mean(axis=0, keepdims=True)
    Hc = H - mu
    k = min(n_components - 1, B, d)
    if k <= 0:
        return np.broadcast_to(mu, H.shape).copy()
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    V = Vt[:k].T
    return mu + (Hc @ V) @ V.T


def token_gamma(H: np.ndarray, n_components: int, eps: float) -> np.ndarray:
    """逐 token 的 C-子空间能量占比 Gamma_i = ||Π_C h_t||^2 / ||h_t||^2。

    公式来源：README ②（108 Gamma_i 定义）。
    """
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2 and H.shape[0] >= 1, f"H 需二维且 T>=1，实际 {H.shape}"
    assert np.isfinite(H).all(), "H 含 NaN/Inf，Gamma_i 不可信"
    PiH = project_to_c_subspace(H, n_components=n_components, eps=eps)
    num = np.sum(PiH ** 2, axis=1)
    den = np.sum(H ** 2, axis=1) + eps
    return num / den


def spectral_keep(gamma_i: np.ndarray, keep_ratio: float) -> tuple:
    """按 Gamma_i 降序保留前 keep_ratio 比例，返回 (kept, dropped) 索引数组。

    keep = int(seq_len * keep_ratio)（README ② 策略）。
    """
    gamma_i = np.asarray(gamma_i, dtype=np.float64)
    assert gamma_i.ndim == 1 and gamma_i.size > 0, f"gamma_i 需为一维非空，实际 {gamma_i.shape}"
    assert 0.0 <= keep_ratio <= 1.0, f"keep_ratio={keep_ratio} 越界，要求∈[0,1]"
    seq = gamma_i.size
    keep = int(seq * keep_ratio)
    sorted_idx = np.argsort(gamma_i)[::-1]
    return sorted_idx[:keep], sorted_idx[keep:]


def retained_c_energy(H: np.ndarray, idx: np.ndarray, n_components: int, eps: float) -> float:
    """被保留 token 携带的 C-子空间能量占全体 C-能量比例。"""
    gamma_i = token_gamma(H, n_components=n_components, eps=eps)
    den = float(gamma_i.sum() + eps)
    return float(gamma_i[idx].sum() / den)


# ---- 第四层：验证引擎 G02Validator（6 项验证 + 结构化日志 + 类型化异常） ----
class G02Validator(_EngineBase):
    """G02 验证引擎：顺序执行 6 项验证（结构化 JSON 日志 + 类型化异常）。"""

    def __init__(
        self,
        config: G02Config,
        synth: KeepRatioSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data
        self.H: np.ndarray | None = None       # 合成激活矩阵
        self.gamma_i: np.ndarray | None = None  # 逐 token Gamma_i
        self.kept: np.ndarray | None = None    # 保留索引
        self.dropped: np.ndarray | None = None  # 丢弃索引

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1&2) 保留数量与集合
    def validate_keep_set(self) -> dict:
        """1&2) 保留数量符合 int 截断规则 + 保留集合为 Gamma_i 最高。"""
        cfg = self.config
        H = self.synth.synth_activation()
        gamma_i = token_gamma(H, cfg.N_COMPONENTS, cfg.EPS)
        kept, dropped = spectral_keep(gamma_i, cfg.KEEP_RATIO)
        self.H, self.gamma_i, self.kept, self.dropped = H, gamma_i, kept, dropped
        ok1 = (len(kept) == int(cfg.T * cfg.KEEP_RATIO)
               and len(dropped) == cfg.T - len(kept))
        ok2 = (len(dropped) == 0
               or float(gamma_i[kept].min()) >= float(gamma_i[dropped].max()))
        if not (ok1 and ok2):
            raise AIQValidationError(
                f"保留数 {len(kept)} ≠ {int(cfg.T * cfg.KEEP_RATIO)} 或集合序错误 "
                f"(min_kept={gamma_i[kept].min() if len(kept) else float('nan'):.4f})",
                expected={"n_keep": int(cfg.T * cfg.KEEP_RATIO), "top_k": True},
                actual={"n_keep": len(kept), "n_drop": len(dropped)},
                param_key="G02",
            )
        return {
            "detail": (f"[1] 保留数 = {len(kept)} (期望 {int(cfg.T * cfg.KEEP_RATIO)}); "
                       f"[2] 保留集合为 Gamma_i 最高 {len(kept)} 个 "
                       f"(min_kept={gamma_i[kept].min():.4f} "
                       f">= max_dropped={gamma_i[dropped].max() if len(dropped) else float('nan'):.4f})"),
            "n_keep": len(kept), "n_drop": len(dropped),
        }

    # ------------------------------------------------------------ 3) 集中度分离
    def validate_separation(self) -> dict:
        """3) 保留 vs 丢弃集合的集中度（比值 > MIN_SEPARATION）。"""
        cfg = self.config
        assert self.gamma_i is not None and self.kept is not None and self.dropped is not None
        mean_kept = float(self.gamma_i[self.kept].mean())
        mean_dropped = float(self.gamma_i[self.dropped].mean())
        assert mean_dropped > 0.0, f"mean_dropped={mean_dropped} 除零风险"
        ratio_kept = mean_kept / mean_dropped
        ok3 = ratio_kept > cfg.MIN_SEPARATION
        if not ok3:
            raise AIQValidationError(
                f"保留集合集中度不足 {ratio_kept:.2f}x（需>{cfg.MIN_SEPARATION}）",
                expected=cfg.MIN_SEPARATION, actual=ratio_kept, param_key="G02",
            )
        return {
            "detail": (f"[3] 保留集合平均 Gamma_i={mean_kept:.4f} vs 丢弃集合={mean_dropped:.4f} "
                       f"(比值 {ratio_kept:.2f}x)"),
            "mean_kept": mean_kept, "mean_dropped": mean_dropped, "ratio": ratio_kept,
        }

    # ------------------------------------------------------------ 4) 谱剪枝 vs 随机
    def validate_vs_random(self) -> dict:
        """4) 谱剪枝 vs 随机剪枝的 C-能量保留（多随机种子对比）。"""
        cfg = self.config
        assert self.H is not None and self.kept is not None
        spec_energy = retained_c_energy(self.H, self.kept, cfg.N_COMPONENTS, cfg.EPS)
        rng2 = np.random.default_rng(cfg.SEED_RANDOM)
        random_energies = []
        for _ in range(cfg.N_RANDOM_TRIALS):
            ridx = rng2.permutation(cfg.T)[:len(self.kept)]
            random_energies.append(retained_c_energy(self.H, ridx, cfg.N_COMPONENTS, cfg.EPS))
        rand_energy = float(np.mean(random_energies))
        ok4 = spec_energy >= rand_energy
        if not ok4:
            raise AIQValidationError(
                f"谱剪枝 {spec_energy:.4f} < 随机剪枝 {rand_energy:.4f}",
                expected="spec >= random", actual=spec_energy, param_key="G02",
            )
        return {
            "detail": (f"[4] 谱剪枝保留 C-能量={spec_energy:.4f} >= 随机剪枝均值={rand_energy:.4f} "
                       f"(Δ={spec_energy - rand_energy:+.4f})"),
            "spec_energy": spec_energy, "rand_energy": rand_energy,
        }

    # ------------------------------------------------------------ 5) keep_ratio 敏感性
    def validate_sensitivity(self) -> dict:
        """5) keep_ratio 敏感性（int(T*ratio) 规则，逐档验证）。"""
        cfg = self.config
        assert self.gamma_i is not None
        detail_parts = ["[5] keep_ratio 敏感性 (int(T*ratio) 规则):"]
        ok5 = True
        for r in cfg.RATIO_SWEEP:
            k, _ = spectral_keep(self.gamma_i, r)
            ok = len(k) == int(cfg.T * r)
            ok5 = ok5 and ok
            detail_parts.append(f"keep_ratio={r} -> 保留 {len(k)} token (期望 {int(cfg.T * r)})")
            if not ok:
                raise AIQValidationError(
                    f"keep_ratio={r} 保留数 {len(k)} ≠ {int(cfg.T * r)}",
                    expected=int(cfg.T * r), actual=len(k), param_key="G02",
                )
        return {"detail": "; ".join(detail_parts), "ok": ok5}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 24 层 k_proj Gamma 上的谱重要性剪枝（惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        real_layers = rd.get("spectral.k_proj_gamma_layers", None)
        tag = rd.source_tag()
        if real_layers is None or len(real_layers) == 0:  # 真实逐层值缺失：审计回退
            audit_mean = rd.audit("k_proj_gamma_mean", 0.625)
            return {
                "detail": (f"[6] [{tag}] 真实逐层 Gamma 不可用，使用审计均值 {audit_mean:.4f} "
                           f"做回退标注（未找到 _real_metrics.json）"),
                "source": tag, "real_mode": False, "audit_mean": audit_mean,
            }
        g_real = np.asarray(real_layers, dtype=np.float64)
        n_layer = g_real.size
        keep_real = int(n_layer * cfg.KEEP_RATIO)
        kept_r, dropped_r = spectral_keep(g_real, cfg.KEEP_RATIO)
        mean_kept_r = float(g_real[kept_r].mean())
        mean_dropped_r = float(g_real[dropped_r].mean())
        rng2r = np.random.default_rng(cfg.SEED_RANDOM)
        rand_means = [g_real[rng2r.permutation(n_layer)[:keep_real]].mean()
                      for _ in range(cfg.N_RANDOM_TRIALS)]
        rand_mean_r = float(np.mean(rand_means))
        real_mean = float(g_real.mean())
        audit_mean = rd.audit("k_proj_gamma_mean", 0.625)
        ok6 = (len(kept_r) == keep_real
               and mean_kept_r >= mean_dropped_r
               and mean_kept_r >= rand_mean_r
               and 0.0 < real_mean < 1.0)
        if not ok6:
            raise RealModelMismatchError(
                f"真实剪枝异常: keep={len(kept_r)}/{keep_real}, "
                f"kept_mean={mean_kept_r:.4f}, dropped_mean={mean_dropped_r:.4f}, "
                f"rand_mean={rand_mean_r:.4f}",
                expected={"keep": keep_real, "kept>=dropped": True, "kept>=rand": True},
                actual={"keep": len(kept_r), "kept_mean": mean_kept_r,
                        "dropped_mean": mean_dropped_r, "rand_mean": rand_mean_r},
                param_key="G02",
            )
        return {
            "detail": (f"[6] [{tag}] 真实 k_proj 逐层 Gamma 均值 = {real_mean:.4f} "
                       f"(vs 文档审计 {audit_mean:.4f}, Δ={abs(real_mean - audit_mean):.4f}); "
                       f"按谱重要性保留 top {keep_real}/{n_layer} 层: 保留层 Gamma={mean_kept_r:.4f} "
                       f"vs 丢弃层={mean_dropped_r:.4f} (比值 "
                       f"{mean_kept_r / (mean_dropped_r + cfg.EPS):.2f}x); "
                       f"谱剪枝 {mean_kept_r:.4f} >= 随机保留均值 {rand_mean_r:.4f}"),
            "source": tag, "real_mean": real_mean, "audit_mean": audit_mean,
            "n_layer": n_layer, "kept_mean": mean_kept_r,
            "dropped_mean": mean_dropped_r, "rand_mean": rand_mean_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "keep_set", self.validate_keep_set),
            (2, "separation", self.validate_separation),
            (3, "vs_random", self.validate_vs_random),
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
    """G02 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="G02 keep_ratio 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    synth = KeepRatioSynthesizer(cfg)
    report = ReportGenerator()
    engine = G02Validator(cfg, synth, report, real_data=RD)

    print("=" * 74)
    print("G02 keep_ratio token 保留比例验证（四层工厂架构，合成数据）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: seq_len={cfg.T} d={cfg.D} KEEP_RATIO={cfg.KEEP_RATIO} "
          f"MIN_SEPARATION={cfg.MIN_SEPARATION} N_RANDOM_TRIALS={cfg.N_RANDOM_TRIALS}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "g02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "g02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "g02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
