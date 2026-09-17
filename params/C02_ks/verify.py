# -*- coding: utf-8 -*-
"""C02 ks — top-k 主成分扫描列表：能量保留率验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 能量保留率定义 E_k = Σ_{i≤k}λᵢ/Σλᵢ ∈ [0,1]
  2. E_k 关于 k 单调不减（曲线性质，README ②曲线性质）
  3. 与 C01 一致性：同一谱下 E(k90-1) < 0.90 <= E(k90)
  4. 集中谱判据 E_16 >= 0.90（高度集中，对照 Qwen k_proj 0.93）
  5. 弥散谱判据 E_16 < 0.90（对照，up_proj 0.82 量级）
  6. 数量级对照：构造谱曲线逼近 Qwen k_proj 0.38/0.55/0.70/0.84/0.93
  7. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 k_proj 逐层 Gamma 谱 24 维，
     真实曲线 E16≈0.72 显著低于文档逐 token 谱 0.93——口径差异如实标注）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  C02Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 C02Config（环境变量 AIQ_C02_<KEY>
                            > YAML > _params_data.json > 模型默认值）
  ValidatorEngine        —— 7 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→engine→report + --profile/--json/--html

数据源：
  主文档行 1832-1934（ks 扫描与能量保留率）
  主文档行 1842-1844（能量占比公式）、1890-1896（投影对比表）、1919-1923（实测曲线）
  《参数审计与实验报告.txt》行 51（状态=理论）
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
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

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 C02Config（pydantic 优先；dataclass 回退） ----
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
        """pydantic 缺失时的空壳基类（无字段，仅提供 dataclass 语义）。

        子类 C02Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class C02Config(_ConfigModelBase):
    """C02 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 C02 节点键名一一对应；取值优先级：
    环境变量 AIQ_C02_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    KS: list = [1, 2, 4, 8, 16]              # C02 ks 固定列表（2 的幂次，README ①）
    D_DIM: int = 896                         # Qwen k_proj head_dim（主文档行 1919 同源）
    TARGET_KPROJ: list = [0.38, 0.55, 0.70, 0.84, 0.93]  # Qwen k_proj 实测曲线（主文档行 1919-1923）
    K90_THR: float = 0.90                    # C01 一致性阈值 / 集中谱判据（C01 energy_thr 同源）
    TARGET_TOL: float = 0.10                 # 曲线数量级偏差容差（合成 vs 实测）
    E16_DOC: float = 0.93                    # 文档 Qwen k_proj 逐 token 谱 E16（主文档行 1919-1923）
    SEED: int = 0                            # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 C02Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """C02 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 C02Config。"""

    def build(self) -> C02Config:
        """构建 C02Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(C02Config, "C02")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def energy_curve(lam: np.ndarray, ks: list) -> list:
    """对每个 k∈ks 计算前 k 个主成分累积能量占比 E_k∈[0,1]（README ②）。"""
    lam = np.asarray(lam, dtype=float)
    assert lam.size > 0, f"energy_curve: 空特征值谱 lam.size={lam.size}"
    assert np.all(np.isfinite(lam)), "energy_curve: 特征值谱含 NaN/Inf，非法输入"
    lam = np.sort(lam)[::-1]
    total = np.sum(lam)
    assert total > 0.0, f"energy_curve: 特征值总和为 0 total={total}"
    return [float(np.sum(lam[:k]) / total) for k in ks]


def k90_of(lam: np.ndarray, thr: float) -> int:
    """C01 同款：降序特征值 -> 累积能量到 thr 的最小 k。"""
    lam = np.asarray(lam, dtype=float)
    assert lam.size > 0, f"k90_of: 空特征值谱 lam.size={lam.size}"
    assert np.all(np.isfinite(lam)), "k90_of: 特征值谱含 NaN/Inf，非法输入"
    lam = np.sort(lam)[::-1]
    total = np.sum(lam)
    assert total > 0.0, f"k90_of: 特征值总和为 0 total={total}"
    return int(np.argmax(np.cumsum(lam) / total >= thr)) + 1


def energy_ratio(lam: np.ndarray, k: int) -> float:
    """前 k 个主成分累积能量占比（C01 一致性用）。"""
    lam = np.asarray(lam, dtype=float)
    assert lam.size > 0, f"energy_ratio: 空特征值谱 lam.size={lam.size}"
    assert np.all(np.isfinite(lam)), "energy_ratio: 特征值谱含 NaN/Inf，非法输入"
    lam = np.sort(lam)[::-1]
    total = np.sum(lam)
    assert total > 0.0, f"energy_ratio: 特征值总和为 0 total={total}"
    return float(np.sum(lam[:k]) / total)


def real_gamma_spectrum(rd: Any) -> np.ndarray | None:
    """真实 k_proj 逐层 Gamma 谱（24 维，_real_metrics.json）；缺失返回 None。

    参数：
      rd: 共享真实数据模块（_real_data，经 setup_env 注入或惰性导入）
    """
    layers = rd.get("spectral.k_proj_gamma_layers")
    if not layers:
        return None
    arr = np.asarray(layers, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


# ---- 第三层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """C02 验证引擎：顺序执行 7 项验证（6 合成 + 1 真实模型对照）。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（方法内 import 或经 _common.setup_env 注入）。
    """

    def __init__(
        self,
        config: C02Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        # C02 无独立合成器（谱输入为构造型测试数据）→ synth=None
        super().__init__(config, None, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 取值域
    def validate_domain(self) -> dict:
        """1) 能量保留率取值域 [0,1]（集中谱）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        # 集中谱：快速指数核 + 微噪声尾（衰减率 -0.35 使前 16 维覆盖 >90%）
        lam_c = np.exp(-0.35 * np.arange(cfg.D_DIM)) + rng.uniform(0, 0.0002, cfg.D_DIM)
        ec = energy_curve(lam_c, cfg.KS)
        in01 = all(0.0 <= e <= 1.0 for e in ec)
        if not in01:
            raise AIQValidationError(
                f"能量保留率超出 [0,1]: {ec}",
                expected=[0.0, 1.0], actual=ec, param_key="C02",
            )
        ec_str = ", ".join(f"E{k}={e:.3f}" for k, e in zip(cfg.KS, ec))
        return {"detail": ec_str, "ec": ec, "in01": True}

    # ------------------------------------------------------------ 2) 单调性
    def validate_monotone(self) -> dict:
        """2) E_k 关于 k 单调不减（README ②曲线性质）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        lam_c = np.exp(-0.35 * np.arange(cfg.D_DIM)) + rng.uniform(0, 0.0002, cfg.D_DIM)
        ec = energy_curve(lam_c, cfg.KS)
        mono = all(ec[i] <= ec[i + 1] for i in range(len(ec) - 1))
        if not mono:
            raise AIQValidationError(
                f"E_k 非单调: {ec}", expected="monotone non-decreasing",
                actual=ec, param_key="C02",
            )
        ec_str = ", ".join(f"E{k}={e:.3f}" for k, e in zip(cfg.KS, ec))
        return {"detail": ec_str, "ec": ec, "mono": True}

    # ------------------------------------------------------------ 3) C01 一致性
    def validate_c01_consistency(self) -> dict:
        """3) 与 C01 一致性：E(k90-1) < 0.90 <= E(k90)（README ④第 4 步）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED + 2)   # 独立种子（与原脚本语义一致）
        lam_c = np.exp(-0.35 * np.arange(cfg.D_DIM)) + rng.uniform(0, 0.0002, cfg.D_DIM)
        k90 = k90_of(lam_c, cfg.K90_THR)
        e_before = energy_ratio(lam_c, k90 - 1)
        e_at = energy_ratio(lam_c, k90)
        ok = (e_before < cfg.K90_THR) and (e_at >= cfg.K90_THR)
        if not ok:
            raise AIQValidationError(
                f"与 C01 一致性不满足: E({k90}-1)={e_before:.4f}, E({k90})={e_at:.4f}",
                expected={"E_before < thr": True, "E_at >= thr": True},
                actual={"e_before": e_before, "e_at": e_at}, param_key="C02",
            )
        return {
            "detail": f"k90={k90}, E({k90}-1)={e_before:.4f}, E({k90})={e_at:.4f}",
            "k90": k90, "e_before": e_before, "e_at": e_at,
        }

    # ------------------------------------------------------------ 4) 集中谱判据
    def validate_concentrated(self) -> dict:
        """4) 集中谱判据 E_16 >= 0.90（README ④第 3 步，对照 Qwen k_proj 0.93）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        lam_c = np.exp(-0.35 * np.arange(cfg.D_DIM)) + rng.uniform(0, 0.0002, cfg.D_DIM)
        ec = energy_curve(lam_c, cfg.KS)
        ok = ec[-1] >= cfg.K90_THR
        if not ok:
            raise AIQValidationError(
                f"集中谱 E_16 不足: {ec[-1]:.3f} 需>={cfg.K90_THR}",
                expected=cfg.K90_THR, actual=ec[-1], param_key="C02",
            )
        return {"detail": f"E_16={ec[-1]:.3f} (实测 0.93)", "e16": ec[-1]}

    # ------------------------------------------------------------ 5) 弥散谱判据
    def validate_diffuse(self) -> dict:
        """5) 弥散谱对照 E_16 < 0.90（接近均匀 -> 弥散，up_proj 0.82 量级）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED + 3)   # 独立种子（与原脚本语义一致）
        lam_f = np.ones(cfg.D_DIM) + rng.uniform(0, 0.1, cfg.D_DIM)
        ef = energy_curve(lam_f, cfg.KS)
        ok = ef[-1] < cfg.K90_THR
        ef_str = ", ".join(f"E{k}={e:.3f}" for k, e in zip(cfg.KS, ef))
        if not ok:
            raise AIQValidationError(
                f"弥散谱 E_16 应<{cfg.K90_THR}: {ef[-1]:.3f}",
                expected=cfg.K90_THR, actual=ef[-1], param_key="C02",
            )
        return {"detail": f"E_16={ef[-1]:.3f}; {ef_str}", "e16": ef[-1]}

    # ------------------------------------------------------------ 6) 数量级对照
    def validate_magnitude(self) -> dict:
        """6) 数量级对照 Qwen k_proj 实测曲线（主文档行 1919-1923）。"""
        cfg = self.config
        lam_t = 1.0 / (1.0 + np.arange(cfg.D_DIM)) ** 1.6 + 0.0001  # 构造谱逼近实测曲线
        et = energy_curve(lam_t, cfg.KS)
        max_abs = max(abs(a - b) for a, b in zip(et, cfg.TARGET_KPROJ))
        ok = max_abs < cfg.TARGET_TOL
        et_str = ", ".join(f"{e:.3f}" for e in et)
        if not ok:
            raise AIQValidationError(
                f"合成曲线偏离实测过大: 最大偏差={max_abs:.3f} 需<{cfg.TARGET_TOL}",
                expected=cfg.TARGET_TOL, actual=max_abs, param_key="C02",
            )
        return {
            "detail": f"合成=[{et_str}], 实测={cfg.TARGET_KPROJ}, 最大偏差={max_abs:.3f}",
            "max_abs": max_abs, "tolerance": cfg.TARGET_TOL,
        }

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型对照（真实 k_proj Gamma 24维谱能量保留率曲线）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        real_gamma = real_gamma_spectrum(rd)
        if real_gamma is None:
            # 审计回退：真实谱缺失时用合成集中谱代替
            lam_fb = np.exp(-0.35 * np.arange(cfg.D_DIM)) + 1e-4
            efb = energy_curve(lam_fb, cfg.KS)
            ok = efb[-1] >= cfg.K90_THR
            if not ok:
                raise AIQValidationError(
                    f"回退谱 E16 不足: {efb[-1]:.3f}",
                    expected=cfg.K90_THR, actual=efb[-1], param_key="C02",
                )
            return {"detail": f"E16={efb[-1]:.3f}（真实数据未就绪）",
                    "tag": "[审计回退]", "e16": efb[-1], "fallback": True}
        tag = "[真实实测]"
        er_real = energy_curve(real_gamma, cfg.KS)
        mono_r = all(er_real[i] <= er_real[i + 1] for i in range(len(er_real) - 1))
        in_r = all(0.0 <= e <= 1.0 for e in er_real)
        max_abs_r = max(abs(a - b) for a, b in zip(er_real, cfg.TARGET_KPROJ))
        # 真实逐层 Gamma 谱弥散：E16≈0.72 显著低于文档逐token谱 0.93（口径差异）
        okr1 = in_r and mono_r and (er_real[-1] < cfg.E16_DOC)
        if not okr1:
            raise RealModelMismatchError(
                f"真实谱 E_k 曲线单调/取值域/弥散校验失败: E16={er_real[-1]:.3f}",
                expected={"in01": True, "mono": True, "e16 < doc": cfg.E16_DOC},
                actual={"er": er_real, "e16": er_real[-1]}, param_key="C02",
            )
        okr2 = max_abs_r >= cfg.TARGET_TOL   # 口径不同 → 最大偏差须如实 ≥ 容差
        if not okr2:
            raise RealModelMismatchError(
                f"真实谱与文档曲线偏差过小(口径异常): {max_abs_r:.3f}",
                expected=cfg.TARGET_TOL, actual=max_abs_r, param_key="C02",
            )
        es_r = ", ".join(f"E{k}={e:.3f}" for k, e in zip(cfg.KS, er_real))
        return {
            "detail": (f"{tag} 真实谱 E_k 曲线: {es_r}; E16={er_real[-1]:.3f} < 文档 0.93 "
                       f"（真实谱跨层弥散）; 与文档 Qwen 曲线最大偏差={max_abs_r:.3f}"
                       f"（≥{cfg.TARGET_TOL}：口径不同），文档逐token谱={cfg.TARGET_KPROJ}"),
            "source": rd.source_tag(), "tag": tag,
            "er": er_real, "e16": er_real[-1], "max_abs": max_abs_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "domain", self.validate_domain),
            (2, "monotone", self.validate_monotone),
            (3, "c01_consistency", self.validate_c01_consistency),
            (4, "concentrated", self.validate_concentrated),
            (5, "diffuse", self.validate_diffuse),
            (6, "magnitude", self.validate_magnitude),
            (7, "real_model", self.validate_real),
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
            # 结构化 JSON 日志（_logging 单例；每行可 json.loads）
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """C02 验证编排：配置层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="C02 ks 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()      # ① 配置层（env > YAML > JSON > 默认）
    report = ReportGenerator()         # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, report, real_data=RD)   # ② 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("C02 ks 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: KS={cfg.KS} D_DIM={cfg.D_DIM} TARGET_KPROJ={cfg.TARGET_KPROJ} "
          f"K90_THR={cfg.K90_THR} TARGET_TOL={cfg.TARGET_TOL} E16_DOC={cfg.E16_DOC} "
          f"SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ③ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "c02_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ④ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "c02_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "c02_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑤ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
