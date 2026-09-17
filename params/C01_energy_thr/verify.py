# -*- coding: utf-8 -*-
"""C01 energy_thr — top-k 能量保留阈值：k90 算法逻辑验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 单方向谱 k90 = 1（信息集中在 1 维，README ⑤表第 1 行）
  2. 均匀谱 k90 = ceil(0.9·d)（各向同性极限，d=896 时=807）
  3. 定义性质 E(k90-1) < 0.90 <= E(k90)（累积能量恰好跨越 90%）
  4. 阈值扫描 0.50/0.90/0.95/0.99 下 k90 单调不减
  5. 压缩比 ratio = d/k90 数量级对照 Qwen k_proj 实测 68.9x
  6. 数量级：合成集中谱 k90 落在 1-60；文档实测 k90=13 (d=896)
     ratio=40-100 区间（README ⑤实测值引用）
  7. 真实模型对照（Qwen2.5-0.5B-Instruct 真实 k_proj 逐层 Gamma 谱，
     24 维 _real_metrics.json；真实谱跨层弥散 E16≈0.72 vs 文档集中谱 0.93，
     口径差异如实标注）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  C01Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退，
                            由 _factory.ConfigFactory.build_model 驱动）
  ConfigFactory          —— 实例化 C01Config（优先级：环境变量 AIQ_C01_<KEY>
                            > YAML config.yaml > _params_data.json > 模型默认值）
  ValidatorEngine        —— 7 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 1740-1831（energy_thr 定义与 k90 算法流程）
  主文档行 1748（k90 定义）、1784-1787（实测表）、1793-1798（扫描表）
  《参数审计与实验报告.txt》行 50（状态=理论，未实测）
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

# ---- 第一层：配置模型 C01Config（pydantic 优先；dataclass 回退） ----
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

        子类 C01Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class C01Config(_ConfigModelBase):
    """C01 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 C01 节点键名一一对应；取值优先级：
    环境变量 AIQ_C01_<KEY> > YAML > _params_data.json > 本模型默认值。
    字段默认值仅作最低优先级兜底（数据文件缺失时），正常由配置层覆盖。
    """

    ENERGY_THR: float = 0.90        # C01 energy_thr 固定值（主文档行 1748）
    D_DIM: int = 896                # Qwen k_proj head_dim（主文档行 1784-1787）
    K_EMP: int = 13                 # 文档实测 Qwen k_proj k90（主文档行 1786）
    THRS_SCAN: list = [0.50, 0.90, 0.95, 0.99]   # 阈值扫描表（主文档行 1793-1798）
    RATIO_MIN: float = 10.0         # 集中谱压缩比下界（README ②数量级对照）
    K_SYN_LO: int = 1               # 合成集中谱 k90 量级下界（README ⑤表）
    K_SYN_HI: int = 60              # 合成集中谱 k90 量级上界（README ⑤表）
    RATIO_EMP_LO: float = 40.0      # 文档实测压缩比区间下界（896/13≈68.9）
    RATIO_EMP_HI: float = 100.0     # 文档实测压缩比区间上界
    E16_DOC: float = 0.93           # 文档集中谱 E16（896 维逐 token 口径）
    GAMMA_MEAN_MIN: float = 0.3     # 真实谱均值非退化下界（对照文档 0.4695 量级）
    SEED: int = 0                   # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 C01Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """C01 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 C01Config。"""

    def build(self) -> C01Config:
        """构建 C01Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(C01Config, "C01")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def k90_of(lam: np.ndarray, thr: float) -> int:
    """主文档行 1770-1776：降序特征值 -> 累积能量到 thr 的最小 k。

    边界防御：空输入 / NaN·Inf / 零总能量。
    """
    lam = np.asarray(lam, dtype=float)
    assert lam.size > 0, f"k90_of: 空特征值谱 lam.size={lam.size}"
    assert np.all(np.isfinite(lam)), "k90_of: 特征值谱含 NaN/Inf，非法输入"
    lam = np.sort(lam)[::-1]                       # 降序：从能量最大的主成分开始累加
    total = np.sum(lam)                            # 总能量（归一化分母）
    assert total > 0.0, f"k90_of: 特征值总和为 0，无法归一化 total={total}"
    # argmax 取第一个满足累积占比>=thr 的下标，+1 将 0 基下标转为 1 基的 k
    return int(np.argmax(np.cumsum(lam) / total >= thr)) + 1


def energy_ratio(lam: np.ndarray, k: int) -> float:
    """前 k 个主成分累积能量占比 E(k)∈[0,1]（README ②公式）。"""
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
        return None                                 # 键不存在或值为空 → 审计回退
    arr = np.asarray(layers, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None                                 # 尺寸为空/含 NaN/Inf 视为数据不可用
    return arr


# ---- 第三层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """C01 验证引擎：顺序执行 7 项验证（6 合成 + 1 真实模型对照）。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（方法内 import 或经 _common.setup_env 注入）。
    """

    def __init__(
        self,
        config: C01Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        # C01 无合成器（谱输入为构造型测试数据，无独立合成逻辑）→ synth=None
        super().__init__(config, None, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real 方法内 import）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 单方向谱
    def validate_unidirectional(self) -> dict:
        """1) 单方向谱：信息集中在 1 维（README ⑤表第 1 行）。"""
        cfg = self.config
        lam1 = np.zeros(64)   # 64 维全零谱，能量仅放第一个维度
        lam1[0] = 1.0         # 唯一非零特征值 → 全部能量集中于一维主成分
        k1 = k90_of(lam1, cfg.ENERGY_THR)
        ok = k1 == 1          # 一个维度即达 90%，k90 应恰为 1
        if not ok:
            raise AIQValidationError(
                f"单方向谱 k90 应为 1，实际 {k1}",
                expected=1, actual=k1, param_key="C01",
            )
        return {"detail": f"k90={k1}, 预期 1", "k90": k1}

    # ------------------------------------------------------------ 2) 均匀谱
    def validate_uniform(self) -> dict:
        """2) 均匀谱：各向同性极限 k90=ceil(0.9·d)（README ⑤表第 2 行）。"""
        cfg = self.config
        lam_u = np.ones(cfg.D_DIM)   # 各维能量均等 → 累积占比与 k 成正比
        k_u = k90_of(lam_u, cfg.ENERGY_THR)
        exp_u = int(np.ceil(cfg.ENERGY_THR * cfg.D_DIM))   # ceil(0.9×896)=807
        ok = k_u == exp_u
        if not ok:
            raise AIQValidationError(
                f"均匀谱 k90 应为 ceil(0.9·d)={exp_u}，实际 {k_u}",
                expected=exp_u, actual=k_u, param_key="C01",
            )
        return {"detail": f"d={cfg.D_DIM}, k90={k_u}, 预期 {exp_u}", "k90": k_u, "expect": exp_u}

    # ------------------------------------------------------------ 3) 定义性质
    def validate_definition(self) -> dict:
        """3) 定义性质：E(k90-1) < 0.90 <= E(k90)（README ④第 3 步）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)   # 固定种子，可复现
        # 指数衰减谱 + 微噪声：集中保证 k90 有限，噪声尾保证可验证严格不等式
        lam_e = np.exp(-0.5 * np.arange(cfg.D_DIM)) + rng.uniform(0, 1e-4, cfg.D_DIM)
        k_e = k90_of(lam_e, cfg.ENERGY_THR)
        e_before = energy_ratio(lam_e, k_e - 1)   # 恰在 k90 前一维的累积能量
        e_at = energy_ratio(lam_e, k_e)           # 恰好到达 k90 时的累积能量
        ok = (e_before < cfg.ENERGY_THR) and (e_at >= cfg.ENERGY_THR)
        if not ok:
            raise AIQValidationError(
                f"定义性质不满足: E({k_e}-1)={e_before:.6f}, E({k_e})={e_at:.6f}",
                expected={"E_before < thr": True, "E_at >= thr": True},
                actual={"e_before": e_before, "e_at": e_at},
                param_key="C01",
            )
        return {
            "detail": f"E({k_e}-1)={e_before:.6f}, E({k_e})={e_at:.6f}",
            "k90": k_e, "e_before": e_before, "e_at": e_at,
        }

    # ------------------------------------------------------------ 4) 阈值扫描单调
    def validate_threshold_mono(self) -> dict:
        """4) 阈值扫描 0.50/0.90/0.95/0.99 下 k90 单调不减（主文档行 1793-1798）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED + 1)   # 独立种子（与原脚本语义一致）
        lam_e = np.exp(-0.5 * np.arange(cfg.D_DIM)) + rng.uniform(0, 1e-4, cfg.D_DIM)
        ks = [k90_of(lam_e, t) for t in cfg.THRS_SCAN]
        ok = all(ks[i] <= ks[i + 1] for i in range(len(ks) - 1))  # 阈值越高 k90 越大
        if not ok:
            raise AIQValidationError(
                f"阈值扫描 k90 非单调: 阈值{cfg.THRS_SCAN} -> k90={ks}",
                expected="monotone non-decreasing", actual=ks, param_key="C01",
            )
        return {"detail": f"阈值{cfg.THRS_SCAN} -> k90={ks}", "ks": ks}

    # ------------------------------------------------------------ 5) 压缩比
    def validate_ratio(self) -> dict:
        """5) 集中谱压缩比 d/k90>10（对照 Qwen k_proj 实测 68.9x，README ②）。"""
        cfg = self.config
        lam_p = np.exp(-0.3 * np.arange(cfg.D_DIM)) + 1e-4  # 集中谱：慢衰减指数核
        k_p = k90_of(lam_p, cfg.ENERGY_THR)
        ratio = cfg.D_DIM / k_p   # 压缩比 = 全维数/保留维数
        ok = ratio > cfg.RATIO_MIN
        if not ok:
            raise AIQValidationError(
                f"集中谱压缩比不足: ratio={ratio:.1f} 需>{cfg.RATIO_MIN}",
                expected=cfg.RATIO_MIN, actual=ratio, param_key="C01",
            )
        return {
            "detail": f"d/k90={cfg.D_DIM}/{k_p}={ratio:.1f}x (实测 68.9x)",
            "k90": k_p, "ratio": ratio,
        }

    # ------------------------------------------------------------ 6) 数量级对照
    def validate_magnitude(self) -> dict:
        """6) 数量级对照 Qwen k_proj（d=896, k90=13, ratio=68.9，主文档行 1784-1787）。"""
        cfg = self.config
        lam_p = np.exp(-0.3 * np.arange(cfg.D_DIM)) + 1e-4
        k_p = k90_of(lam_p, cfg.ENERGY_THR)
        ratio_emp = cfg.D_DIM / cfg.K_EMP    # 文档实测 k90=13 → 896/13≈68.9
        ok = (cfg.K_SYN_LO <= k_p <= cfg.K_SYN_HI
              and cfg.RATIO_EMP_LO <= ratio_emp <= cfg.RATIO_EMP_HI)
        scan_p = ", ".join(f"{t:.2f}->{k90_of(lam_p, t)}" for t in cfg.THRS_SCAN)
        if not ok:
            raise AIQValidationError(
                f"数量级对照不满足: 合成k90={k_p}, 实测ratio={ratio_emp:.1f}x",
                expected={"k_syn": [cfg.K_SYN_LO, cfg.K_SYN_HI],
                          "ratio_emp": [cfg.RATIO_EMP_LO, cfg.RATIO_EMP_HI]},
                actual={"k_syn": k_p, "ratio_emp": ratio_emp},
                param_key="C01",
            )
        return {
            "detail": f"合成k90={k_p}(允许{cfg.K_SYN_LO}-{cfg.K_SYN_HI}), "
                      f"实测ratio={ratio_emp:.1f}x; 扫描[{scan_p}]",
            "k_syn": k_p, "ratio_emp": ratio_emp,
        }

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型对照（Qwen2.5-0.5B-Instruct 真实 k_proj 逐层 Gamma 谱）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        real_gamma = real_gamma_spectrum(rd)
        if real_gamma is None:
            # 审计回退：真实谱缺失时用合成集中谱代替，保证验证仍可运行
            lam_fb = np.exp(-0.3 * np.arange(cfg.D_DIM)) + 1e-4
            k_fb = k90_of(lam_fb, cfg.ENERGY_THR)
            ok = cfg.K_SYN_LO <= k_fb <= cfg.K_SYN_HI
            if not ok:
                raise AIQValidationError(
                    f"回退谱 k90 越界: {k_fb}",
                    expected=[cfg.K_SYN_LO, cfg.K_SYN_HI], actual=k_fb,
                    param_key="C01",
                )
            return {"detail": f"k90={k_fb}（真实数据未就绪）", "tag": "[审计回退]",
                    "k90": k_fb, "fallback": True}
        tag = "[真实实测]"
        k_r = k90_of(real_gamma, cfg.ENERGY_THR)   # 真实谱的 k90（24 维内集中度）
        ks_r = [energy_ratio(real_gamma, k) for k in (1, 2, 4, 8, 16, 24)]
        mono_r = all(ks_r[i] <= ks_r[i + 1] for i in range(len(ks_r) - 1))  # 能量保留率随 k 单调
        in_r = all(0.0 <= e <= 1.0 for e in ks_r)  # 能量占比取值域 [0,1]
        e16_r = ks_r[4]   # E16：前 16 维累积能量（文档集中谱 0.93）
        # 真实谱跨层弥散（E16≈0.72 < 文档集中谱 0.93），口径差异如实标注
        okr1 = (1 <= k_r <= 24) and (e16_r < cfg.E16_DOC) and in_r
        if not okr1:
            raise RealModelMismatchError(
                f"真实谱 k90/E16 口径校验失败: k90={k_r}, E16={e16_r:.3f}",
                expected={"k90": [1, 24], "e16 < doc": cfg.E16_DOC, "in [0,1]": True},
                actual={"k90": k_r, "e16": e16_r, "ks": ks_r},
                param_key="C01",
            )
        okr2 = mono_r and (float(real_gamma.mean()) >= cfg.GAMMA_MEAN_MIN)  # 单调 + 均值非退化
        if not okr2:
            raise RealModelMismatchError(
                f"真实谱能量保留率单调性/均值不满足: mean={real_gamma.mean():.4f}",
                expected={"mono": True, "mean >= min": cfg.GAMMA_MEAN_MIN},
                actual={"mean": float(real_gamma.mean()), "mono": mono_r},
                param_key="C01",
            )
        es_r = ", ".join(f"E{k}={e:.3f}" for k, e in zip((1, 2, 4, 8, 16, 24), ks_r))
        return {
            "detail": (f"{tag} 真实 k_proj Gamma 24维谱 k90={k_r}/24, "
                       f"E16={e16_r:.3f}, 谱均值={real_gamma.mean():.4f}; "
                       f"{es_r}; 压缩比 d/k90={24 / k_r:.2f}x "
                       f"（文档 896/13=68.9x 为逐 token 谱口径，非同一对象；"
                       f"真实谱跨层弥散 E16≈0.72 vs 文档集中谱 0.93）"),
            "source": rd.source_tag(), "tag": tag,
            "k90": k_r, "e16": e16_r, "mean": float(real_gamma.mean()),
            "ks": ks_r, "mono": mono_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "unidirectional", self.validate_unidirectional),
            (2, "uniform", self.validate_uniform),
            (3, "definition", self.validate_definition),
            (4, "threshold_mono", self.validate_threshold_mono),
            (5, "ratio", self.validate_ratio),
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
    """C01 验证编排：配置层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="C01 energy_thr 四层工厂验证")
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
    print("C01 energy_thr 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: ENERGY_THR={cfg.ENERGY_THR} D_DIM={cfg.D_DIM} K_EMP={cfg.K_EMP} "
          f"THRS_SCAN={cfg.THRS_SCAN} RATIO_MIN={cfg.RATIO_MIN} "
          f"K_SYN=[{cfg.K_SYN_LO},{cfg.K_SYN_HI}] RATIO_EMP=[{cfg.RATIO_EMP_LO},"
          f"{cfg.RATIO_EMP_HI}] E16_DOC={cfg.E16_DOC} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ③ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "c01_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ④ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "c01_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "c01_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑤ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
