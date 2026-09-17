# -*- coding: utf-8 -*-
"""D09 lam_grid — λ↔E[DEFF] 映射搜索：恒等式与反解验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 网格完整性：linspace(0.05,1.0,40)，40 点
  2. 解析恒等式 DEFF = (|κ₁|+|κ₂|)²/(κ₁²+κ₂²) = 1 + sin2φ
  3. 均匀 φ∈[0,45°] 下 E[DEFF] = 1 + 2/π ≈ 1.63662（用户指定解析值）
  4. MC 建立 λ→E[DEFF] 映射（mc_n=400000, D08），与解析闭式对比
     E(λ) = 1 + (-4λ·lnλ)/(π(1-λ²))，E(1)=1+2/π；含 λ=1 极限与单调性
  5. 反解 target=1.5711 -> λ*（线性插值）；与实测 λ=0.852 对比
     -> 各向异性高斯模型下 λ*≈0.44，偏差≈48% > 15% -> 未闭环（T06 一致）
  6. 映射表摘录：数值全部有限（可追溯反解）
  7. 真实模型对照（真实 λ=0.8516、DEFF=1.5920 代入闭环判据：
     E(λ_real)=1.634 vs 实测 1.5920，闭合偏差≈2.6%，未闭环）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D09Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D09Config（环境变量 AIQ_D09_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            GRID_LAM 经 get_grid 建模，跨节点在 build() 解析）
  LamSynthesizer         —— λ→E[DEFF] MC 映射合成（各向异性高斯）
  ValidatorEngine        —— 7 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 2938-3073（D09 六步流程）
  《参数审计与实验报告.txt》行 8、15、60、109、114、140、154（状态=已用）
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
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 D09Config（pydantic 优先；dataclass 回退） ----
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

        子类 D09Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D09Config(_ConfigModelBase):
    """D09 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（TARGET_NS）
    在 ConfigFactory.build() 中按来源节点显式解析（D10）。
    """

    MC_N: int = 400000                  # D08 固定采样数（主文档行 2983）
    LAM_MEAS: float = 0.852             # 实测 λ（审计报告 0.8516 / 主文档 0.852）
    TARGET_NS: float = 1.5711           # D10 NS 平台目标（D10.TARGET_NS 同源）
    GRID_LAM: tuple = (0.05, 1.0, 40)   # λ 网格：(start, stop, count)，linspace(0.05,1,40)
    IDENT_TOL: float = 1e-12            # 恒等式 DEFF=1+sin2φ 数值容差（EPS 量级）
    UNIFORM_REL: float = 0.005          # 均匀 φ 下 E[DEFF] 相对误差上界
    MAP_ERR_MAX: float = 2e-3           # 映射 MC vs 闭式最大误差
    E1_TOL: float = 0.01                # λ=1 极限 E(1)≈1+2/π 容差
    N_INV_MAX: int = 2                  # MC 映射局部反转数上限（允许≤2）
    LAM_LO: float = 0.30                # 反解 λ* 审计区间下界（0.41-0.52）
    LAM_HI: float = 0.60                # 反解 λ* 审计区间上界
    DEV_MIN: float = 0.15               # 未闭环偏差判据（D11.CLOSURE_THR 同源）
    CLOS_LO: float = 0.01               # 真实闭合偏差下界（非示意 0.16% 的判据）
    CLOS_HI: float = 0.10               # 真实闭合偏差上界
    SEED: int = 0                       # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D09Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D09 配置工厂：实例化 D09Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D09Config:
        cfg = self.build_model(D09Config, "D09")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D10）
        cfg.TARGET_NS = self.get_float("D10", "TARGET_NS", 1.5711)
        # D 组网格参数：get_grid 保留 (start, stop, count) 语义
        cfg.GRID_LAM = self.get_grid("D09", "GRID_LAM", (0.05, 1.0, 40))
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def E_deff_analytic(lam) -> np.ndarray:
    """各向异性高斯模型闭式解：E[DEFF](λ)=1+(-4λ·lnλ)/(π(1-λ²))，E(1)=1+2/π。

    边界防御：lam 须在 (0,1]（λ≤0 时 lnλ 无定义）。
    """
    lam = np.asarray(lam, dtype=float)
    assert lam.size > 0, f"E_deff_analytic: 空输入 lam.size={lam.size}"
    assert np.all(np.isfinite(lam)), "E_deff_analytic: 输入含 NaN/Inf"
    assert np.all(lam > 0.0), f"E_deff_analytic: λ 必须>0，得到 {lam.min()}"
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(lam < 1.0,
                        (-4.0 * lam * np.log(lam)) / (np.pi * (1.0 - lam ** 2)),
                        2.0 / np.pi)
    return 1.0 + term


# ---- 第三层：合成器 LamSynthesizer（算法与原脚本完全一致） ----
class LamSynthesizer(_SynthBase):
    """D09 合成器：λ→E[DEFF] MC 映射（各向异性高斯 (κ1,κ2) 对）。"""

    def __init__(self, cfg: D09Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def build_lookup(self, seed: int | None = None) -> tuple:
        """D09 映射：λ∈linspace(0.05,1,40) -> E[DEFF]（各向异性高斯 MC）。

        边界防御：MC 结果须全部有限。
        返回 (lam_grid, deff_lookup)。
        """
        cfg = self._cfg
        lam_grid = np.linspace(*cfg.GRID_LAM)   # (start, stop, count) 展开
        rng = np.random.default_rng(self._seed if seed is None else seed)
        deff_lookup = []
        for lam in lam_grid:
            k1s = rng.normal(0.0, 1.0, cfg.MC_N)
            k2s = rng.normal(0.0, lam, cfg.MC_N)
            deff = (np.abs(k1s) + np.abs(k2s)) ** 2 / (k1s ** 2 + k2s ** 2)
            deff_lookup.append(float(deff.mean()))
        deff_arr = np.array(deff_lookup)
        assert np.all(np.isfinite(deff_arr)), "build_lookup: MC 映射含 NaN/Inf"
        return lam_grid, deff_arr


# ---- 第四层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D09 验证引擎：顺序执行 7 项验证（6 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D09Config,
        synth: LamSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real 方法内 import）
        self.lam_grid: np.ndarray | None = None   # 共享中间结果（供后续步骤复用）
        self.deff_lookup: np.ndarray | None = None

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    def _lookup(self, seed: int | None = None) -> tuple:
        """惰性构建 λ→E[DEFF] 映射（首次构建后缓存复用）。"""
        if self.deff_lookup is None:
            self.lam_grid, self.deff_lookup = self.synth.build_lookup(seed)
        return self.lam_grid, self.deff_lookup

    # ------------------------------------------------------------ 1) 网格完整性
    def validate_grid(self) -> dict:
        """1) λ 网格完整性（README ①）。"""
        cfg = self.config
        lam_grid = np.linspace(*cfg.GRID_LAM)
        dlam = np.diff(lam_grid)
        ok = (len(lam_grid) == 40) and bool(np.allclose(dlam, dlam[0]))
        if not ok:
            raise AIQValidationError(
                f"λ 网格不符: n={len(lam_grid)}",
                expected=40, actual=len(lam_grid), param_key="D09",
            )
        return {
            "detail": f"n={len(lam_grid)}, 范围[{lam_grid[0]},{lam_grid[-1]}], "
                      f"步长≈{dlam[0]:.4f}",
            "n": len(lam_grid),
        }

    # ------------------------------------------------------------ 2) 恒等式
    def validate_identity(self) -> dict:
        """2) 恒等式 DEFF = 1 + sin2φ（README ②核心恒等式）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        k1 = rng.normal(0, 1, 200000)
        k2 = rng.normal(0, 1, 200000)
        deff_direct = (np.abs(k1) + np.abs(k2)) ** 2 / (k1 ** 2 + k2 ** 2)
        phi = np.arctan(np.abs(k2) / np.abs(k1))
        deff_id = 1.0 + np.sin(2.0 * phi)
        max_err = float(np.max(np.abs(deff_direct - deff_id)))
        ok = max_err < cfg.IDENT_TOL
        if not ok:
            raise AIQValidationError(
                f"恒等式 DEFF=1+sin2φ 失效: 最大误差={max_err:.2e}",
                expected=cfg.IDENT_TOL, actual=max_err, param_key="D09",
            )
        return {
            "detail": f"最大误差={max_err:.2e} (允许<{cfg.IDENT_TOL})",
            "max_err": max_err,
        }

    # ------------------------------------------------------------ 3) 均匀 φ 期望
    def validate_uniform_expectation(self) -> dict:
        """3) 均匀 φ 下 E[DEFF] = 1+2/π（README ②解析期望）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        phi_u = rng.uniform(0.0, np.pi / 4.0, cfg.MC_N)
        e_mc_uniform = float(np.mean(1.0 + np.sin(2.0 * phi_u)))
        e_analytic_uniform = 1.0 + 2.0 / np.pi
        rel_uniform = abs(e_mc_uniform - e_analytic_uniform) / e_analytic_uniform
        ok = rel_uniform < cfg.UNIFORM_REL
        if not ok:
            raise AIQValidationError(
                f"均匀 φ 期望偏离: MC={e_mc_uniform:.5f} vs 解析={e_analytic_uniform:.5f}",
                expected=cfg.UNIFORM_REL, actual=rel_uniform, param_key="D09",
            )
        return {
            "detail": f"MC={e_mc_uniform:.5f} vs 解析={e_analytic_uniform:.5f}, "
                      f"相对误差={rel_uniform * 100:.3f}%<0.5%",
            "e_mc": e_mc_uniform, "rel": rel_uniform,
        }

    # ------------------------------------------------------------ 4) 映射 vs 闭式
    def validate_mapping(self) -> dict:
        """4) λ→E[DEFF] 映射（MC 400000）vs 闭式（README ⑤表第 4 行）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup(seed=cfg.SEED + 1)
        an = E_deff_analytic(lam_grid)
        err_map = np.abs(deff_lookup - an)
        max_map_err = float(err_map.max())
        e_at_1 = np.interp(1.0, lam_grid, deff_lookup)
        mono_an = bool(np.all(np.diff(E_deff_analytic(lam_grid)) > 0))
        n_inv = int(np.sum(np.diff(deff_lookup) <= 0))
        ok = (max_map_err < cfg.MAP_ERR_MAX
              and abs(e_at_1 - (1 + 2 / np.pi)) < cfg.E1_TOL
              and mono_an and n_inv <= cfg.N_INV_MAX)
        if not ok:
            raise AIQValidationError(
                f"映射校验失败: 最大误差={max_map_err:.4f}, λ=1→E={e_at_1:.4f}, "
                f"反转数={n_inv}",
                expected={"map_err <": cfg.MAP_ERR_MAX, "e1_tol": cfg.E1_TOL,
                          "mono": True, "n_inv <=": cfg.N_INV_MAX},
                actual={"max_err": max_map_err, "e1": e_at_1,
                        "mono": mono_an, "n_inv": n_inv},
                param_key="D09",
            )
        return {
            "detail": f"最大误差={max_map_err:.4f}(<{cfg.MAP_ERR_MAX}), λ=1→E={e_at_1:.4f}"
                      f"(解析 1+2/π=1.6366), 闭式严格单调={mono_an}, "
                      f"MC 局部反转数={n_inv}(允许≤{cfg.N_INV_MAX})",
            "max_err": max_map_err, "e1": e_at_1, "mono": mono_an, "n_inv": n_inv,
        }

    # ------------------------------------------------------------ 5) 反解
    def validate_inversion(self) -> dict:
        """5) 反解 1.5711 -> λ*（README ⑤表第 6 行 / 审计 T06 未闭环）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup(seed=cfg.SEED + 1)
        lam_star = float(np.interp(cfg.TARGET_NS, deff_lookup, lam_grid))
        dev = abs(lam_star - cfg.LAM_MEAS) / cfg.LAM_MEAS
        ok = (cfg.LAM_LO < lam_star < cfg.LAM_HI) and (dev > cfg.DEV_MIN)
        if not ok:
            raise AIQValidationError(
                f"反解未闭环判据不满足: λ*={lam_star:.3f}, 偏差={dev * 100:.1f}%",
                expected={"lam* in": [cfg.LAM_LO, cfg.LAM_HI], "dev >": cfg.DEV_MIN},
                actual={"lam_star": lam_star, "dev": dev}, param_key="D09",
            )
        return {
            "detail": f"λ*={lam_star:.3f} (审计区间 {cfg.LAM_LO}-{cfg.LAM_HI}), "
                      f"实测λ={cfg.LAM_MEAS}, 偏差={dev * 100:.1f}%>{cfg.DEV_MIN * 100:.0f}%; "
                      f"⚠️ 主文档叙述 0.853/0.16% 为示意值，与真实 MC 不符",
            "lam_star": lam_star, "dev": dev,
        }

    # ------------------------------------------------------------ 6) 映射表
    def validate_table(self) -> dict:
        """6) 映射表摘录：数值全部有限（README ④第 5 步映射表）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup(seed=cfg.SEED + 1)
        an = E_deff_analytic(lam_grid)
        fin_ok = bool(np.all(np.isfinite(deff_lookup)) and np.all(np.isfinite(an)))
        if not fin_ok:
            raise AIQValidationError(
                "映射表含 NaN/Inf", expected="all finite",
                actual="has NaN/Inf", param_key="D09",
            )
        rows = "; ".join(f"λ={lam_grid[i]:.3f}:{deff_lookup[i]:.4f}/{an[i]:.4f}"
                         for i in [0, 5, 10, 15, 20, 25, 30, 35, 39])
        return {"detail": f"{rows} (MC/闭式)", "finite": True}

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型对照（真实 λ=0.8516、DEFF=1.5920 代入闭环判据）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        lam_grid, deff_lookup = self._lookup(seed=cfg.SEED + 1)
        if not rd.has_real():
            # 审计回退：真实数据缺失时回退审计 λ=0.852
            lam_star = float(np.interp(cfg.TARGET_NS, deff_lookup, lam_grid))
            dev = abs(lam_star - cfg.LAM_MEAS) / cfg.LAM_MEAS
            ok = dev > cfg.DEV_MIN
            if not ok:
                raise AIQValidationError(
                    f"回退反解偏差不足: {dev * 100:.1f}%",
                    expected=cfg.DEV_MIN, actual=dev, param_key="D09",
                )
            return {"detail": f"λ*={lam_star:.3f} 偏差={dev * 100:.1f}%",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        lam_real = rd.get("curvature.lambda_ratio", cfg.LAM_MEAS)
        deff_real = rd.get("curvature.DEFF_plat", cfg.TARGET_NS)
        # 1) 真实 λ 代入闭式 E(λ) vs 实测 DEFF
        e_at_lam_real = float(E_deff_analytic(lam_real))
        clos_dev = abs(e_at_lam_real - deff_real) / deff_real
        okr1 = (clos_dev > cfg.CLOS_LO) and (clos_dev < cfg.CLOS_HI)
        if not okr1:
            raise RealModelMismatchError(
                f"真实闭合偏差越界: {clos_dev * 100:.2f}%",
                expected=[cfg.CLOS_LO, cfg.CLOS_HI], actual=clos_dev, param_key="D09",
            )
        # 2) 真实 DEFF 反解 λ* vs 实测 λ
        lam_star_real = float(np.interp(deff_real, deff_lookup, lam_grid))
        dev_real = abs(lam_star_real - lam_real) / lam_real
        okr2 = dev_real > cfg.DEV_MIN
        if not okr2:
            raise RealModelMismatchError(
                f"真实反解未闭环偏差不足: {dev_real * 100:.1f}%",
                expected=cfg.DEV_MIN, actual=dev_real, param_key="D09",
            )
        return {
            "detail": (f"{tag} 真实 λ={lam_real:.4f}→E(λ)={e_at_lam_real:.4f} vs 实测 "
                       f"DEFF={deff_real:.4f}，闭合偏差={clos_dev * 100:.2f}%"
                       f"（⚠️ 文档示意 0.16% 与真实 {clos_dev * 100:.1f}% 不符，如实标注）; "
                       f"真实 DEFF 反解 λ*={lam_star_real:.3f} vs 实测 {lam_real:.4f}，"
                       f"偏差={dev_real * 100:.1f}% > {cfg.DEV_MIN * 100:.0f}% -> 未闭环（T06 一致）"),
            "source": rd.source_tag(), "tag": tag,
            "lam_real": lam_real, "deff_real": deff_real,
            "clos_dev": clos_dev, "lam_star_real": lam_star_real, "dev_real": dev_real,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "grid", self.validate_grid),
            (2, "identity", self.validate_identity),
            (3, "uniform_expectation", self.validate_uniform_expectation),
            (4, "mapping", self.validate_mapping),
            (5, "inversion", self.validate_inversion),
            (6, "table", self.validate_table),
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


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """D09 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D09 lam_grid 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = LamSynthesizer(cfg)                # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D09 lam_grid 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MC_N={cfg.MC_N} LAM_MEAS={cfg.LAM_MEAS} TARGET_NS={cfg.TARGET_NS} "
          f"GRID_LAM={cfg.GRID_LAM} MAP_ERR_MAX={cfg.MAP_ERR_MAX} "
          f"DEV_MIN={cfg.DEV_MIN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d09_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d09_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d09_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
