# -*- coding: utf-8 -*-
"""B05 SAMPLES_PER_FRAME 每帧曲率采样点数 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 有放回采样逻辑：8-token 池 -> 18 个采样点，形状 (18, d)
  2. 平均每 token 被采样次数 ≈ 18/8 = 2.25
  3. 经验关系 B05 ≈ 2*B04 + 2（B04=8 -> B05=18）
  4. 二次曲面拟合稳定性：18 点比 10 点更稳定（噪声下 κ 估计方差更小）
  5. 真实模型对照：真实曲率对（phi_pairs_all.npy 6303×2）有放回采样 18 点估计 DEFF 稳定性

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B05Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 B05Config（环境变量 AIQ_B05_<KEY> > YAML > _params_data.json > 默认）
  BootstrapSynthesizer    —— 有放回采样（bootstrap）+ 二次曲面拟合稳定性（Hessian 主曲率 std）
  ValidatorEngine         —— 5 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 1117-1260（B05）
  《参数完整定义与公式.txt》B05 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实曲率对数据集 phi_pairs_all.npy（6303×2，真实逐点 (κ1,κ2)）：
          有放回采样 18 点估计 DEFF，与全局真实 DEFF 平台≈1.5920 对照稳定性。
  如实呈现：合成层用高斯池演示采样稳定性；真实层用真实曲率点云验证同一性质。
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

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注，CFG 供路径探测）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 B05Config（pydantic 优先；dataclass 回退） ----
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


class B05Config(_ConfigModelBase):
    """B05 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B05 节点键名一一对应；取值优先级：
    环境变量 AIQ_B05_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    B04: int = 8                     # 每帧 token 数（B04；有放回采样的池大小，数据层键 B04_TOK）
    B05: int = 18                    # 每帧曲率采样点数（数据层键 N_SAMPLE）
    DIM: int = 64                    # 模拟激活维度
    N_REP_AVG: int = 100000          # 平均每 token 采样次数的 MC 重复数
    AVG_TOL: float = 0.05            # 平均采样次数容差
    N_TRIALS_FIT: int = 300          # 二次曲面拟合稳定性重复次数
    K1_TRUE: float = 2.0             # 主曲率真值 κ1（鞍面正曲率）
    K2_TRUE: float = -0.8            # 主曲率真值 κ2（鞍面负曲率）
    NOISE_FIT: float = 0.02          # 二次曲面拟合噪声标准差
    SAMPLE_PTS: list = [10, 18, 30]  # 拟合稳定性扫描采样点数


# ---- 第二层：配置工厂 ConfigFactory（实例化 B05Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B05 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B05Config。"""

    def build(self) -> B05Config:
        """构建 B05Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(B05Config, "B05")


# ---- 第三层：合成器 BootstrapSynthesizer（算法与原脚本完全一致） ----
class BootstrapSynthesizer(_SynthBase):
    """B05 有放回采样 + 二次曲面拟合稳定性合成器。

    - bootstrap_sample(H_frame, n_sample, rng)：从帧内 token 池有放回采样；
    - fit_principal_curvatures(n_pts, noise_std, seed, n_trials)：原点邻域二次曲面拟合，
      返回主曲率 (κ1,κ2) 估计的 std（统计稳定性度量）。
    """

    def __init__(self, cfg: B05Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def bootstrap_sample(self, H_frame: np.ndarray, n_sample: int,
                         rng: np.random.Generator) -> np.ndarray:
        """从帧内 token 池有放回采样 n_sample 次。
        有放回（bootstrap）允许同一 token 多次入样，扩增每帧采样点数。"""
        if H_frame.shape[0] == 0:
            raise ValueError("空采样池")      # 防御：空池无法采样
        idx = rng.integers(0, H_frame.shape[0], size=n_sample)   # 池索引有放回抽取
        return H_frame[idx]

    def fit_principal_curvatures(self, n_pts: int, noise_std: float, seed: int,
                                 n_trials: int) -> tuple[float, float]:
        """在原点邻域拟合二次曲面 z=0.5*(k1*x^2+k2*y^2)，返回估计 (k1,k2) 的 std。
        用 N_TRIALS_FIT 次独立重复估计主曲率，衡量估计的统计稳定性（std 越小越稳）。"""
        cfg = self._cfg
        r = np.random.default_rng(seed)
        k1s = []     # 各次重复的 κ1 估计收集器
        k2s = []     # 各次重复的 κ2 估计收集器
        for _ in range(n_trials):
            x = r.normal(0, 0.03, n_pts)                 # 邻域横坐标
            y = r.normal(0, 0.03, n_pts)                 # 邻域纵坐标
            # 二次曲面高度 + 噪声（κ 真值注入，验证能否恢复）
            z = 0.5 * (cfg.K1_TRUE * x**2 + cfg.K2_TRUE * y**2) + r.normal(0, noise_std, n_pts)
            A = np.stack([x**2, y**2, x * y, x, y, np.ones_like(x)], axis=1)  # 6 系数设计矩阵
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)  # 最小二乘解 6 系数
            a, b, c = coef[0], coef[1], coef[2]           # 二次项系数（xy 交叉项）
            Hm = np.array([[2 * a, c], [c, 2 * b]])     # Hessian 矩阵（曲率张量）
            ev = np.linalg.eigvalsh(Hm)                 # 特征值 = 主曲率
            k1s.append(ev[1])                           # κ1（较大）
            k2s.append(ev[0])                           # κ2（较小）
        return float(np.std(k1s)), float(np.std(k2s))   # 主曲率估计的统计波动


# ---- 第四层：验证引擎 ValidatorEngine（5 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B05 验证引擎：顺序执行 5 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）；真实曲率数据经 CFG 路径探测。
    """

    def __init__(
        self,
        config: B05Config,
        synth: BootstrapSynthesizer,
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

    # ------------------------------------------------------------ 1) 经验关系
    def validate_empirical(self) -> dict:
        """1) 经验关系 B05 ≈ 2*B04+2（B04=8 -> B05=18）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        self._rng = rng                                   # 供采样步骤复用同一随机流
        empirical = 2 * cfg.B04 + 2         # 经验公式：B05 ≈ 2·B04 + 2
        ok1 = (empirical == cfg.B05)        # 公式还原断言
        if not ok1:
            raise AIQValidationError(
                f"经验关系不符: 2*{cfg.B04}+2={empirical} != B05={cfg.B05}",
                expected=empirical, actual=cfg.B05, param_key="B05",
            )
        return {"detail": f"经验关系 B05 ≈ 2*B04+2: 2*{cfg.B04}+2 = {empirical} vs B05={cfg.B05}",
                "empirical": empirical}

    # ------------------------------------------------------------ 2) 有放回采样形状
    def validate_sampling_shape(self) -> dict:
        """2) 有放回采样：8-token 池 -> 18 点，形状 (18, DIM)。"""
        cfg = self.config
        assert self._rng is not None
        H_frame = self._rng.standard_normal((cfg.B04, cfg.DIM))          # 每帧 8 个 token 激活
        S = self.synth.bootstrap_sample(H_frame, cfg.B05, self._rng)     # 有放回采样 18 点
        ok2 = (S.shape == (cfg.B05, cfg.DIM))                            # 形状断言：(18, 64)
        if not ok2:
            raise AIQValidationError(
                f"采样形状不符: {S.shape}",
                expected=(cfg.B05, cfg.DIM), actual=tuple(S.shape), param_key="B05",
            )
        return {"detail": (f"采样池 {H_frame.shape} -> 有放回采样 {cfg.B05} 次 -> {S.shape}"
                           f"（应为 ({cfg.B05},{cfg.DIM})）"),
                "shape": tuple(S.shape)}

    # ------------------------------------------------------------ 3) 平均每 token 采样次数
    def validate_avg_count(self) -> dict:
        """3) 平均每 token 采样次数 ≈ B05/B04 = 2.25（向量化 MC）。"""
        cfg = self.config
        assert self._rng is not None
        # 一次生成全部 MC 索引：N_REP_AVG 次重复 × B05 个采样，向量化统计
        idx_all = self._rng.integers(0, cfg.B04, size=(cfg.N_REP_AVG, cfg.B05))
        cnt = np.bincount(idx_all.ravel(), minlength=cfg.B04)          # 每 token 出现总次数
        avg = float(cnt.mean() / cfg.N_REP_AVG)          # 单次采样中每 token 平均被抽次数
        ok3 = abs(avg - cfg.B05 / cfg.B04) < cfg.AVG_TOL         # 期望 B05/B04 = 2.25
        # 多样性保持：单次 18 次采样覆盖了多少个不同 token
        cover = int(np.sum(np.bincount(self._rng.integers(0, cfg.B04, size=cfg.B05),
                                       minlength=cfg.B04) > 0))
        if not ok3:
            raise AIQValidationError(
                f"平均采样次数不符: {avg:.4f} vs 期望 {cfg.B05 / cfg.B04}",
                expected=cfg.B05 / cfg.B04, actual=avg, param_key="B05",
            )
        return {"detail": (f"平均每 token 采样次数 = {avg:.4f}（期望 {cfg.B05}/{cfg.B04} = "
                           f"{cfg.B05 / cfg.B04:.2f}）; 单次 18 次采样覆盖不同 token 数 = "
                           f"{cover}/{cfg.B04}（多样性保持）"),
                "avg": avg, "cover": cover}

    # ------------------------------------------------------------ 4) 二次曲面拟合稳定性
    def validate_fit_stability(self) -> dict:
        """4) 拟合稳定性：采样点越多估计越稳（std 越小），18 点显著优于 10 点。"""
        cfg = self.config
        rows = []
        fit = {n: self.synth.fit_principal_curvatures(n, cfg.NOISE_FIT, 7, cfg.N_TRIALS_FIT)
               for n in cfg.SAMPLE_PTS}
        for n, (s1, s2) in fit.items():
            rows.append(f"采样 {n:>2} 点: std(κ1)={s1:.5f}, std(κ2)={s2:.5f}")
        s1_10 = fit[cfg.SAMPLE_PTS[0]][0]      # 10 点时 κ1 估计 std
        s1_18 = fit[cfg.SAMPLE_PTS[1]][0]      # 18 点时 κ1 估计 std
        s1_30 = fit[cfg.SAMPLE_PTS[2]][0]      # 30 点时 κ1 估计 std
        # 稳定性判据：采样点越多估计越稳定（std 越小），18 点显著优于 10 点
        ok4 = (s1_18 < s1_10) and (s1_30 <= s1_18 + 1e-12)
        if not ok4:
            raise AIQValidationError(
                f"拟合稳定性趋势不符: 10点={s1_10:.5f}, 18点={s1_18:.5f}, 30点={s1_30:.5f}",
                expected="s1_18 < s1_10 且 s1_30 <= s1_18",
                actual={"s1_10": s1_10, "s1_18": s1_18, "s1_30": s1_30},
                param_key="B05",
            )
        return {"detail": (f"二次曲面拟合稳定性（真值 κ1={cfg.K1_TRUE}, κ2={cfg.K2_TRUE}, "
                           f"加噪 σ={cfg.NOISE_FIT}, {cfg.N_TRIALS_FIT} 次重复）: " + "; ".join(rows)
                           + f"; std(κ1): 10 点 = {s1_10:.5f} > 18 点 = {s1_18:.5f} "
                           + f">= 30 点 = {s1_30:.5f}"),
                "s1_10": s1_10, "s1_18": s1_18, "s1_30": s1_30}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实曲率点云（phi_pairs_all.npy）有放回采样 18 点估计 DEFF 稳定性。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        # 数据文件经 CFG.phi_pairs_path() 定位：插件内 params/ 副本优先，回退工作区 AIQ/
        npy_path = CFG.phi_pairs_path()
        ok5 = False
        deff_glob = mean_est = std_est = float("nan")   # 预置哨兵值（缺失时保持 NaN）
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        if os.path.isfile(npy_path):                    # 数据存在才做真实对照
            pairs = np.load(npy_path)                   # 真实逐点 (κ1,κ2) 点云
            k1r, k2r = pairs[:, 0], pairs[:, 1]
            # 全局真实 DEFF：逐点 DEFF 均值（真实平台基准）
            deff_glob = float(np.mean((abs(k1r) + abs(k2r))**2 / (k1r**2 + k2r**2 + 1e-16)))
            rngr = np.random.default_rng(cfg.SEED)
            # 2000 次有放回采样 18 点，估计每次的 DEFF（向量化）
            idx_all = rngr.integers(0, pairs.shape[0], size=(2000, cfg.B05))
            deff_m = ((abs(k1r[idx_all]) + abs(k2r[idx_all]))**2
                      / (k1r[idx_all]**2 + k2r[idx_all]**2 + 1e-16)).mean(axis=1)
            mean_est = float(deff_m.mean())   # 采样估计均值
            std_est = float(deff_m.std())     # 采样估计波动
            err = abs(mean_est - deff_glob) / deff_glob   # 与全局平台相对偏差
            deff_doc = rd.audit("DEFF_plat", 1.5920)      # 文档声称平台
            ok5 = (err < 0.02) and (std_est < 0.10)       # 偏差<2% 且波动<0.10
            if not ok5:
                raise RealModelMismatchError(
                    f"真实曲率采样稳定性不符: 全局={deff_glob:.4f}, 估计均值={mean_est:.4f}, std={std_est:.4f}",
                    expected={"err<0.02": True, "std<0.10": True},
                    actual={"deff_glob": deff_glob, "mean_est": mean_est, "std_est": std_est},
                    param_key="B05",
                )
            detail = (f"{tag} 真实曲率点云 {pairs.shape[0]}×{pairs.shape[1]} "
                      f"（phi_pairs_all.npy）有放回采样 {cfg.B05} 点 × 2000 次: "
                      f"DEFF 估计均值 = {mean_est:.4f}，std = {std_est:.4f}; "
                      f"全局真实 DEFF 平台 = {deff_glob:.4f}（文档声称 {deff_doc:.4f}），"
                      f"采样估计偏差 = {err * 100:.2f}%")
        else:
            raise RealModelMismatchError(
                f"phi_pairs_all.npy 缺失（{npy_path}），真实曲率采样稳定性验证无法执行",
                expected="数据文件存在", actual=npy_path, param_key="B05",
            )
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "deff_glob": deff_glob, "mean_est": mean_est, "std_est": std_est,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 5 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "empirical", self.validate_empirical),
            (2, "sampling_shape", self.validate_sampling_shape),
            (3, "avg_count", self.validate_avg_count),
            (4, "fit_stability", self.validate_fit_stability),
            (5, "real_model", self.validate_real_model),
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
    """B05 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B05 SAMPLES_PER_FRAME 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = BootstrapSynthesizer(cfg)                 # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("B05 SAMPLES_PER_FRAME 验证（四层工厂架构）：有放回采样 + 拟合稳定性")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: B04={cfg.B04} B05={cfg.B05} DIM={cfg.DIM} N_REP_AVG={cfg.N_REP_AVG} "
          f"N_TRIALS_FIT={cfg.N_TRIALS_FIT} K1_TRUE={cfg.K1_TRUE} K2_TRUE={cfg.K2_TRUE} "
          f"NOISE_FIT={cfg.NOISE_FIT} SAMPLE_PTS={cfg.SAMPLE_PTS} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b05_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b05_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b05_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
