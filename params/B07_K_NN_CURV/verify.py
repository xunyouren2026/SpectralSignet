# -*- coding: utf-8 -*-
"""B07 K_NN_CURV 版本A曲率近邻数 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. kNN=15：每个采样点取最近 15 个邻居（k 远大于 6 系数 -> 超定拟合）
  2. 在 15 点邻域上拟合二次曲面，恢复 Hessian 特征值（主曲率 κ1, κ2）
  3. 与真值对比并标注相对误差；K=8/15/25/40 扫描演示稳定性趋势
  4. 输出 DEFF、Gauss 曲率 K、平均曲率 Hmed 一致性
  5. 真实模型对照：真实曲率对（phi_pairs_all.npy 6303×2）200 锚点 × kNN=15 邻域统计

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  B07Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 B07Config（环境变量 AIQ_B07_<KEY> > YAML > _params_data.json > 默认）
  ManifoldSynthesizer     —— 真实流形嵌入（z=0.5(κ1x²+κ2y²)）+ kNN 邻域拟合 + K 值扫描
  ValidatorEngine         —— 4 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  《参数附录表完整版》行 1357-1443（B07）
  《参数完整定义与公式.txt》B07 项
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实曲率对数据集 phi_pairs_all.npy（6303×2，真实逐点 (κ1,κ2)）：
          200 个锚点 × kNN=15 邻域统计 K<0%/DEFF/Hmed，与全局真实值对照。
  口径说明：真实点云 (κ1,κ2) 为曲率值本身而非流形嵌入，kNN 用于验证「局部邻域
            曲率指标与全局一致」；合成层仍验证二次曲面拟合恢复 κ1/κ2。
  如实呈现：全局真实 DEFF≈1.5920 / K<0%≈45.50% / Hmed≈1.3152 作为邻域对照基准。
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

# ---- 第一层：配置模型 B07Config（pydantic 优先；dataclass 回退） ----
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


class B07Config(_ConfigModelBase):
    """B07 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 B07 节点键名一一对应；取值优先级：
    环境变量 AIQ_B07_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    K_NN: int = 15                   # 版本A曲率近邻数（远大于 6 系数 → 超定拟合）
    K1_TRUE: float = 3.0             # 主曲率真值 κ1（鞍面正曲率）
    K2_TRUE: float = -1.2            # 主曲率真值 κ2（鞍面负曲率）
    N_TOTAL: int = 300               # 流形点集规模
    NOISE_STD: float = 0.0002        # 曲率拟合噪声
    N_TRIALS_SCAN: int = 200         # K 值扫描重复次数
    REL_ERR1_TOL: float = 0.10       # κ1 相对误差容差
    REL_ERR2_TOL: float = 0.20       # κ2 相对误差容差
    K_SCAN: list = [8, 15, 25, 40]   # K 值扫描档位


# ---- 第二层：配置工厂 ConfigFactory（实例化 B07Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """B07 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 B07Config。"""

    def build(self) -> B07Config:
        """构建 B07Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(B07Config, "B07")


# ---- 第三层：合成器 ManifoldSynthesizer（算法与原脚本完全一致） ----
class ManifoldSynthesizer(_SynthBase):
    """B07 流形嵌入 + kNN 邻域拟合合成器。

    - generate_manifold(n_total, noise_std, rng)：真实流形 z=0.5(κ1x²+κ2y²) 嵌入 R³；
    - fit_neighborhood(xy, pts, k, target)：按 2D 参数距离取最近 k 个邻居，
      邻域二次曲面拟合 → (κ1, κ2, DEFF, Gauss K, Hmed)；
    - scan_k(k, seed, n_trials)：给定 K 重复 n_trials 次，返回 κ1 估计的 std 与中位相对误差。
    """

    def __init__(self, cfg: B07Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def generate_manifold(self, n_total: int, noise_std: float,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """真实流形 z = 0.5*(κ1*x^2 + κ2*y^2) 嵌入 R^3，参数化坐标为 (x, y)。
        返回 (点集 (n,3), 参数坐标 (n,2))。"""
        cfg = self._cfg
        X = rng.normal(0, 0.3, n_total)          # 横坐标（含噪采样）
        Y = rng.normal(0, 0.3, n_total)          # 纵坐标（含噪采样）
        Z = 0.5 * (cfg.K1_TRUE * X**2 + cfg.K2_TRUE * Y**2) + rng.normal(0, noise_std, n_total)   # 曲面高度
        return np.stack([X, Y, Z], axis=1), np.stack([X, Y], axis=1)   # (R³ 点集, 参数坐标)

    def fit_neighborhood(self, xy: np.ndarray, pts: np.ndarray, k: int,
                         target: np.ndarray) -> tuple[float, float, float, float, float]:
        """对目标点 p0=原点，按 (x,y) 2D 参数距离取最近 k 个邻居，在邻域上拟合二次曲面，
        返回 (κ1, κ2, DEFF, Gauss K, Hmed)。"""
        d2 = np.sum((xy - target) ** 2, axis=1)      # 到目标点的 2D 欧氏距离平方
        nbr_idx = np.argsort(d2)[:k]                 # 最近的 k 个邻居索引
        neigh_xy = xy[nbr_idx]                       # 邻居参数坐标
        neigh_z = pts[nbr_idx, 2]                    # 邻居曲面高度
        x, y = neigh_xy[:, 0], neigh_xy[:, 1]
        A = np.stack([x**2, y**2, x * y, x, y, np.ones_like(x)], axis=1)   # 6 系数设计矩阵
        coef, *_ = np.linalg.lstsq(A, neigh_z, rcond=None)   # 邻域最小二乘拟合
        a, b, c = coef[0], coef[1], coef[2]          # 二次项系数
        Hm = np.array([[2 * a, c], [c, 2 * b]])          # Hessian
        ev = np.linalg.eigvalsh(Hm)                      # 特征值 = 主曲率
        k1_est, k2_est = ev[1], ev[0]                    # κ1(大), κ2(小)
        deff_v = (abs(k1_est) + abs(k2_est))**2 / (k1_est**2 + k2_est**2)   # 有效维数比
        gk = k1_est * k2_est                 # Gauss 曲率（鞍面为负）
        hm = (k1_est + k2_est) / 2           # 平均曲率
        return float(k1_est), float(k2_est), float(deff_v), float(gk), float(hm)

    def scan_k(self, k: int, seed: int, n_trials: int) -> tuple[float, float]:
        """对给定 K 重复 n_trials 次拟合，返回 κ1 估计的 std 与中位相对误差。
        衡量 K 取值的稳定性（std/中位误差越小越稳定）。"""
        cfg = self._cfg
        rng = np.random.default_rng(seed)
        ests = []
        for _ in range(n_trials):
            pts_t, xy_t = self.generate_manifold(cfg.N_TOTAL, cfg.NOISE_STD, rng)   # 每次重新生成点集
            d2t = np.sum(xy_t**2, axis=1)          # 到原点的距离平方
            idx_t = np.argsort(d2t)[:k]            # 最近 k 个邻居
            xt, yt = xy_t[idx_t, 0], xy_t[idx_t, 1]
            zt = pts_t[idx_t, 2]
            At = np.stack([xt**2, yt**2, xt * yt, xt, yt, np.ones_like(xt)], axis=1)   # 设计矩阵
            ct, *_ = np.linalg.lstsq(At, zt, rcond=None)   # 最小二乘解
            Ht = np.array([[2 * ct[0], ct[2]], [ct[2], 2 * ct[1]]])   # Hessian
            evt = np.linalg.eigvalsh(Ht)             # 特征值
            ests.append(evt[1])                # κ1 估计
        ests = np.array(ests)
        return float(ests.std()), float(np.median(np.abs(ests - cfg.K1_TRUE) / cfg.K1_TRUE))   # (std, 中位相对误差)


# ---- 第四层：验证引擎 ValidatorEngine（4 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """B07 验证引擎：顺序执行 4 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）；真实曲率数据经 CFG 路径探测。
    """

    def __init__(
        self,
        config: B07Config,
        synth: ManifoldSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1-2) kNN=15 邻域拟合恢复
    def validate_knn_fit(self) -> dict:
        """1-2) kNN=15 邻域拟合恢复 κ1/κ2（相对误差在容差内）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        pts, xy = self.synth.generate_manifold(cfg.N_TOTAL, cfg.NOISE_STD, rng)   # 合成流形点集
        target = np.zeros(2)     # 目标点 p0 = 原点
        d2 = np.sum(xy**2, axis=1)                 # 各点到原点距离平方
        radius = float(np.sqrt(d2[np.argsort(d2)[:cfg.K_NN]].max()))   # kNN 邻域半径
        k1_est, k2_est, deff_v, gk, hm = self.synth.fit_neighborhood(xy, pts, cfg.K_NN, target)   # 邻域拟合
        rel_err1 = abs(k1_est - cfg.K1_TRUE) / abs(cfg.K1_TRUE)   # κ1 相对误差
        rel_err2 = abs(k2_est - cfg.K2_TRUE) / abs(cfg.K2_TRUE)   # κ2 相对误差
        ok1 = (rel_err1 < cfg.REL_ERR1_TOL) and (rel_err2 < cfg.REL_ERR2_TOL)   # 恢复精度断言
        if not ok1:
            raise AIQValidationError(
                f"kNN 拟合恢复误差过大: κ1={rel_err1 * 100:.3f}%, κ2={rel_err2 * 100:.3f}%",
                expected={"rel_err1": cfg.REL_ERR1_TOL, "rel_err2": cfg.REL_ERR2_TOL},
                actual={"rel_err1": rel_err1, "rel_err2": rel_err2},
                param_key="B07",
            )
        return {"detail": (f"点集 {cfg.N_TOTAL} 个 -> 取最近 {cfg.K_NN} 个邻居，"
                           f"邻域半径 ≈ {radius:.4f}; 恢复: κ1 = {k1_est:.4f} (真值 {cfg.K1_TRUE}), "
                           f"κ2 = {k2_est:.4f} (真值 {cfg.K2_TRUE}); 相对误差: κ1 {rel_err1 * 100:.3f}% "
                           f"(容差 {cfg.REL_ERR1_TOL * 100:.0f}%), κ2 {rel_err2 * 100:.3f}% "
                           f"(容差 {cfg.REL_ERR2_TOL * 100:.0f}%)"),
                "radius": radius, "k1_est": k1_est, "k2_est": k2_est,
                "rel_err1": rel_err1, "rel_err2": rel_err2, "deff_v": deff_v, "gk": gk, "hm": hm}

    # ------------------------------------------------------------ 3) 指标一致性
    def validate_metrics(self) -> dict:
        """3) 指标一致性：DEFF∈[1,2] 且 Gauss 曲率为负（鞍面）。"""
        cfg = self.config
        # 复用项1-2 的结果（重新拟合一次保持纯函数语义）
        rng = np.random.default_rng(cfg.SEED)
        pts, xy = self.synth.generate_manifold(cfg.N_TOTAL, cfg.NOISE_STD, rng)
        _, _, deff_v, gk, hm = self.synth.fit_neighborhood(xy, pts, cfg.K_NN, np.zeros(2))
        ok2 = (1.0 <= deff_v <= 2.0) and (gk < 0.0)   # DEFF∈[1,2] 且 Gauss 曲率为负（鞍面）
        if not ok2:
            raise AIQValidationError(
                f"指标一致性不符: DEFF={deff_v:.4f}, Gauss K={gk:.4f}",
                expected={"DEFF∈[1,2]": True, "Gauss<0": True},
                actual={"deff": deff_v, "gk": gk},
                param_key="B07",
            )
        return {"detail": (f"DEFF = {deff_v:.4f} (∈[1,2]); Gauss K = κ1·κ2 = {gk:.4f} (<0 为鞍面); "
                           f"Hmed = {hm:.4f}"),
                "deff_v": deff_v, "gk": gk, "hm": hm}

    # ------------------------------------------------------------ 4) K 值扫描稳定性
    def validate_k_scan(self) -> dict:
        """4) K 值扫描：标准 K_NN(15) 优于最小档 K_SCAN[0](8)（std 与中位相对误差均更小）。"""
        cfg = self.config
        rows = []
        scan = {k: self.synth.scan_k(k, 7, cfg.N_TRIALS_SCAN) for k in cfg.K_SCAN}   # 各 K 的 (std, 中位相对误差)
        ok3 = True
        for k, (std, med) in scan.items():
            tag = "  <- 标准配置" if k == cfg.K_NN else ""   # 标记标准 K=15
            rows.append(f"K={k:>2}: std(κ1) = {std:.5f}, 中位相对误差 = {med * 100:.2f}%{tag}")
        # 稳定性判据：标准 K_NN(15) 优于最小档 K_SCAN[0](8)（邻居越多拟合越稳，但过多会引入远处偏差）
        ok3 = (scan[cfg.K_NN][0] < scan[cfg.K_SCAN[0]][0]) and (scan[cfg.K_NN][1] < scan[cfg.K_SCAN[0]][1])
        if not ok3:
            raise AIQValidationError(
                f"K={cfg.K_NN} 未比 K={cfg.K_SCAN[0]} 更稳定: "
                f"std={scan[cfg.K_NN][0]:.5f}/{scan[cfg.K_SCAN[0]][0]:.5f}",
                expected="K_NN std/中位误差 < K_SCAN[0]",
                actual=scan, param_key="B07",
            )
        return {"detail": (f"K 值扫描：κ1 估计稳定性（{cfg.N_TRIALS_SCAN} 次重复，模拟文档 CV 趋势）: "
                           + "; ".join(rows)
                           + f"; K={cfg.K_NN} vs K={cfg.K_SCAN[0]}: std "
                           f"{scan[cfg.K_NN][0]:.5f}<{scan[cfg.K_SCAN[0]][0]:.5f}, 中位相对误差 "
                           f"{scan[cfg.K_NN][1] * 100:.2f}%<{scan[cfg.K_SCAN[0]][1] * 100:.2f}%"),
                "scan": scan}

    # ------------------------------------------------------------ 5) 真实模型对照
    def validate_real_model(self) -> dict:
        """5) 真实模型对照：真实曲率点云（phi_pairs_all.npy）200 锚点 × kNN=15 邻域统计。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        # 数据文件经 CFG.phi_pairs_path() 定位：插件内 params/ 副本优先，回退工作区 AIQ/
        npy_path = CFG.phi_pairs_path()
        ok4 = False
        deff_glob = deff_nbr_mean = float("nan")   # 预置哨兵值
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        if os.path.isfile(npy_path):
            pairs = np.load(npy_path).astype(float)     # 真实曲率点云 (κ1,κ2)
            k1r, k2r = pairs[:, 0], pairs[:, 1]
            # 全局真实指标：逐点 DEFF 均值、K<0% 比例、Hmed
            deff_all = (abs(k1r) + abs(k2r))**2 / (k1r**2 + k2r**2 + 1e-16)
            deff_glob = float(deff_all.mean())
            kneg_glob = float(100.0 * np.mean(k1r * k2r < 0))
            hmed_glob = float(np.median(np.abs((k1r + k2r) / 2)))
            rngr = np.random.default_rng(cfg.SEED)
            anchors = rngr.choice(pairs.shape[0], size=200, replace=False)   # 200 个随机锚点
            nm = []      # 各锚点邻域指标收集器
            for a in anchors:
                d2 = (pairs[:, 0] - pairs[a, 0])**2 + (pairs[:, 1] - pairs[a, 1])**2   # 邻域距离
                nb = np.argsort(d2)[:cfg.K_NN]          # kNN=15 个最近邻居
                k1n, k2n = pairs[nb, 0], pairs[nb, 1]
                deffn = (abs(k1n) + abs(k2n))**2 / (k1n**2 + k2n**2 + 1e-16)   # 邻域 DEFF
                nm.append([float(100.0 * np.mean(k1n * k2n < 0)), float(deffn.mean()),
                           float(np.median(np.abs((k1n + k2n) / 2)))])   # [K<0%, DEFF, Hmed]
            nm = np.array(nm)
            deff_nbr_mean = float(nm[:, 1].mean())   # 邻域 DEFF 均值
            err = abs(deff_nbr_mean - deff_glob) / deff_glob   # 邻域 vs 全局偏差
            ok4 = (err < 0.05) and bool(np.isfinite(nm).all())   # 偏差<5% 且全部有限
            if not ok4:
                raise RealModelMismatchError(
                    f"真实曲率 kNN 邻域与全局偏差过大: 全局={deff_glob:.4f}, 邻域={deff_nbr_mean:.4f}",
                    expected={"err<0.05": True, "finite": True},
                    actual={"deff_glob": deff_glob, "deff_nbr_mean": deff_nbr_mean},
                    param_key="B07",
                )
            detail = (f"{tag} 真实曲率点云 {pairs.shape[0]}×{pairs.shape[1]} "
                      f"（phi_pairs_all.npy），200 个锚点 × kNN={cfg.K_NN} 邻域; "
                      f"邻域均值: K<0% = {nm[:, 0].mean():.2f}%, DEFF = {deff_nbr_mean:.4f}, "
                      f"Hmed = {nm[:, 2].mean():.4f}; 全局真实: DEFF = {deff_glob:.4f}（文档声称 "
                      f"{rd.audit('DEFF_plat', 1.5920):.4f}），K<0% = {kneg_glob:.2f}%，"
                      f"Hmed = {hmed_glob:.4f}; 邻域 DEFF 与全局偏差 = {err * 100:.2f}%"
                      f"（局部 kNN 统计与全局一致）")
        else:
            raise RealModelMismatchError(
                f"phi_pairs_all.npy 缺失（{npy_path}），真实 kNN 邻域统计验证无法执行",
                expected="数据文件存在", actual=npy_path, param_key="B07",
            )
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "deff_glob": deff_glob, "deff_nbr_mean": deff_nbr_mean,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 4 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "knn_fit", self.validate_knn_fit),
            (2, "metrics", self.validate_metrics),
            (3, "k_scan", self.validate_k_scan),
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
    """B07 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="B07 K_NN_CURV 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = ManifoldSynthesizer(cfg)                  # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print(f"B07 K_NN_CURV 验证（四层工厂架构）：kNN={cfg.K_NN} 二次拟合 Hessian 特征值 (κ1, κ2)")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: K_NN={cfg.K_NN} K1_TRUE={cfg.K1_TRUE} K2_TRUE={cfg.K2_TRUE} "
          f"N_TOTAL={cfg.N_TOTAL} NOISE_STD={cfg.NOISE_STD} N_TRIALS_SCAN={cfg.N_TRIALS_SCAN} "
          f"REL_ERR1_TOL={cfg.REL_ERR1_TOL} REL_ERR2_TOL={cfg.REL_ERR2_TOL} K_SCAN={cfg.K_SCAN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "b07_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "b07_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "b07_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
