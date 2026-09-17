# -*- coding: utf-8 -*-
"""D11 closure_thr — 闭环判据阈值：判定逻辑与敏感性验证（四层工厂架构）
====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 阈值固定值 = 0.15（15%）
  2. 判据函数边界：dev<15% 闭环 / dev>15% 未闭环（构造 14.9%/15.1% 边界切换）
  3. 主文档叙述值应用：λ*=0.853→0.16%强闭环; 0.910→6.86%弱闭环; 0.65→23.7%未闭环
  4. 真实 MC 反解应用（D09/D10 各向异性高斯模型）：
       λ*_NS≈0.44 vs 实测 0.852 -> 偏差≈48% > 15% -> 未闭环（T06 审计一致）
  5. 阈值敏感性表：5%→NS✓/AI✗, 15%→两者✓, 20%→两者✓（主文档行 3294-3298）
  6. 跨域口径：AI DEFF 1.5920 vs NS 1.5711 差 1.3% -> 严格跨域口径未闭环
  7. 真实模型对照（真实 λ=0.8516、DEFF=1.5920 代入 15% 闭环判据：
     真实 DEFF 反解 λ*≈0.51，偏差≈40% > 15% -> 未闭环）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  D11Config              —— 配置模型（pydantic 校验；缺失时 dataclass 回退）
  ConfigFactory          —— 实例化 D11Config（环境变量 AIQ_D11_<KEY>
                            > YAML > _params_data.json > 模型默认值；
                            跨节点取值在 build() 中解析）
  LookupSynthesizer      —— λ→E[DEFF] MC 映射合成（各向异性高斯）
  ValidatorEngine        —— 7 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                 —— 仅编排 cfg→synth→engine→report + --profile/--json/--html

数据源：
  主文档行 3208-3327（D11 五步流程 + 阈值敏感性表）
  《参数审计与实验报告.txt》行 8、62、140、154（状态=已用，T06 未闭环）
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

# ---- 第一层：配置模型 D11Config（pydantic 优先；dataclass 回退） ----
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

        子类 D11Config 自动继承 dataclass 行为：ConfigFactory.build_model
        检测到 dataclass 后走 _build_dataclass 运行时校验路径。
        """

    _ConfigModelBase = _DataclassBase


class D11Config(_ConfigModelBase):
    """D11 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 节点键名对应；跨参数节点取值（MC_N/
    DEFF_NS/DEFF_AI/GRID_LAM）在 ConfigFactory.build() 中按来源节点
    显式解析（D08/D10/D09）。
    """

    THR: float = 0.15                   # D11 固定值（主文档行 3211、规格行 239）
    DEV_REF: float = 0.852              # 实测 λ（审计报告 0.8516 / 主文档 0.852，D11.DEV_REF）
    MC_N: int = 400000                  # D08 固定采样数（D08.MC_N 同源）
    DEFF_NS: float = 1.5711             # 跨域平台 NS 值（D10.TARGET_NS 同源）
    DEFF_AI: float = 1.5920             # 跨域平台 AI 值（D10.TARGET_AI 同源）
    GRID_LAM: tuple = (0.05, 1.0, 40)   # λ 网格（单源在 D09.GRID_LAM）
    CROSS_DOMAIN_MIN: float = 0.005     # 真实跨域口径差下界（严格跨域未闭环判据）
    SEED: int = 7                       # H01 固定随机种子（算法逻辑常量，保留）


# ---- 第二层：配置工厂 ConfigFactory（实例化 D11Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """D11 配置工厂：实例化 D11Config 并对跨参数节点字段按来源显式解析。"""

    def build(self) -> D11Config:
        cfg = self.build_model(D11Config, "D11")   # pydantic/dataclass 回退
        # 跨参数节点：保持 _params_data.json 单一数据源（D08/D10/D09）
        cfg.MC_N = self.get_int("D08", "MC_N", 400000)
        cfg.DEFF_NS = self.get_float("D10", "TARGET_NS", 1.5711)
        cfg.DEFF_AI = self.get_float("D10", "TARGET_AI", 1.5920)
        cfg.GRID_LAM = self.get_grid("D09", "GRID_LAM", (0.05, 1.0, 40))
        return cfg


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def deviation(lam_star: float, lam_meas: float) -> float:
    """相对偏差 |λ*-λ实测|/λ实测（README ②公式）。

    边界防御：实测基准须为正。
    """
    assert lam_meas > 0.0, f"deviation: 实测基准 lam_meas={lam_meas} 必须>0"
    return abs(lam_star - lam_meas) / lam_meas


def judge(lam_star: float, lam_meas: float, thr: float) -> tuple:
    """闭环判据：dev<thr -> '闭环'，否则 '未闭环'（README ①）。"""
    dev = deviation(lam_star, lam_meas)
    return dev, ("闭环" if dev < thr else "未闭环")


# ---- 第三层：合成器 LookupSynthesizer（算法与原脚本完全一致） ----
class LookupSynthesizer(_SynthBase):
    """D11 合成器：λ→E[DEFF] MC 映射（各向异性高斯 (κ1,κ2) 对）。"""

    def __init__(self, cfg: D11Config) -> None:
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
            k1 = rng.normal(0.0, 1.0, cfg.MC_N)
            k2 = rng.normal(0.0, lam, cfg.MC_N)
            deff = (np.abs(k1) + np.abs(k2)) ** 2 / (k1 ** 2 + k2 ** 2)
            deff_lookup.append(float(deff.mean()))
        deff_arr = np.array(deff_lookup)
        assert np.all(np.isfinite(deff_arr)), "build_lookup: MC 映射含 NaN/Inf"
        return lam_grid, deff_arr


# ---- 第四层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """D11 验证引擎：顺序执行 7 项验证（6 合成 + 1 真实模型对照）。"""

    def __init__(
        self,
        config: D11Config,
        synth: LookupSynthesizer,
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

    # ------------------------------------------------------------ 1) 阈值固定值
    def validate_threshold(self) -> dict:
        """1) 闭环判据阈值 = 0.15 (15%)（README ①）。"""
        cfg = self.config
        ok = np.isclose(cfg.THR, 0.15, rtol=0, atol=1e-12)
        if not ok:
            raise AIQValidationError(
                f"闭环判据阈值不符: {cfg.THR}",
                expected=0.15, actual=cfg.THR, param_key="D11",
            )
        return {"detail": f"thr={cfg.THR} (主文档行 3211)", "thr": cfg.THR}

    # ------------------------------------------------------------ 2) 判据边界
    def validate_boundary(self) -> dict:
        """2) 判据边界切换 14.9% 闭环 / 15.1% 未闭环（README ⑤表第 2 行）。"""
        cfg = self.config
        dev_lo = 0.149
        dev_hi = 0.151
        ok = (dev_lo < cfg.THR) and not (dev_hi < cfg.THR)
        if not ok:
            raise AIQValidationError(
                f"判据边界切换失效: dev_lo={dev_lo}, dev_hi={dev_hi}",
                expected={"dev_lo < thr": True, "dev_hi >= thr": True},
                actual={"dev_lo": dev_lo, "dev_hi": dev_hi}, param_key="D11",
            )
        return {
            "detail": f"dev={dev_lo:.3f}→{'闭环' if dev_lo < cfg.THR else '未闭环'}; "
                      f"dev={dev_hi:.3f}→{'闭环' if dev_hi < cfg.THR else '未闭环'}",
            "dev_lo": dev_lo, "dev_hi": dev_hi,
        }

    # ------------------------------------------------------------ 3) 主文档叙述值
    def validate_doc_values(self) -> dict:
        """3) 主文档叙述值：0.853 闭环 / 0.910 闭环 / 0.65 未闭环（示意，README ⑤表第 3 行）。"""
        cfg = self.config
        r_ns = judge(0.853, cfg.DEV_REF, cfg.THR)[1]
        r_ai = judge(0.910, cfg.DEV_REF, cfg.THR)[1]
        r_lo = judge(0.65, cfg.DEV_REF, cfg.THR)[1]
        ok = (r_ns == "闭环") and (r_ai == "闭环") and (r_lo == "未闭环")
        if not ok:
            raise AIQValidationError(
                f"主文档叙述值判定不符: {r_ns}/{r_ai}/{r_lo}",
                expected={"0.853": "闭环", "0.910": "闭环", "0.65": "未闭环"},
                actual={"0.853": r_ns, "0.910": r_ai, "0.65": r_lo}, param_key="D11",
            )
        d3 = "; ".join(f"λ*={ls:.3f}:{deviation(ls, cfg.DEV_REF) * 100:.2f}%→"
                       f"{judge(ls, cfg.DEV_REF, cfg.THR)[1]}"
                       for ls in [0.853, 0.910, 0.65])
        return {"detail": d3 + " (示意值)", "r_ns": r_ns, "r_ai": r_ai, "r_lo": r_lo}

    # ------------------------------------------------------------ 4) 真实 MC 反解
    def validate_mc_inversion(self) -> dict:
        """4) 真实 MC 反解：NS/AI 均未闭环（README ⑤表第 4 行 / 审计 T06）。"""
        cfg = self.config
        lam_grid, deff_lookup = self._lookup()
        ls_ns = float(np.interp(cfg.DEFF_NS, deff_lookup, lam_grid))
        ls_ai = float(np.interp(cfg.DEFF_AI, deff_lookup, lam_grid))
        dev_ns, res_ns = judge(ls_ns, cfg.DEV_REF, cfg.THR)
        dev_ai, res_ai = judge(ls_ai, cfg.DEV_REF, cfg.THR)
        ok = (res_ns == "未闭环") and (res_ai == "未闭环")
        if not ok:
            raise AIQValidationError(
                f"真实 MC 反解未闭环判据不满足: NS={res_ns}, AI={res_ai}",
                expected={"NS": "未闭环", "AI": "未闭环"},
                actual={"ns": res_ns, "ai": res_ai}, param_key="D11",
            )
        return {
            "detail": f"NS: λ*={ls_ns:.3f} 偏差={dev_ns * 100:.1f}%→{res_ns}; "
                      f"AI: λ*={ls_ai:.3f} 偏差={dev_ai * 100:.1f}%→{res_ai} "
                      f"(审计区间 0.41-0.52 vs 实测 0.852)",
            "ls_ns": ls_ns, "ls_ai": ls_ai, "dev_ns": dev_ns, "dev_ai": dev_ai,
        }

    # ------------------------------------------------------------ 5) 阈值敏感性
    def validate_sensitivity(self) -> dict:
        """5) 阈值敏感性表 5%/15%/20%（主文档行 3294-3298，示意值 0.853/0.910）。"""
        cfg = self.config
        sens = []
        for thr in [0.05, 0.15, 0.20]:
            r_ns5 = judge(0.853, cfg.DEV_REF, thr)[1]
            r_ai5 = judge(0.910, cfg.DEV_REF, thr)[1]
            sens.append((thr, r_ns5, r_ai5))
        ok = (sens[0][1] == "闭环" and sens[0][2] == "未闭环"
              and sens[1][2] == "闭环" and sens[2][2] == "闭环")
        if not ok:
            raise AIQValidationError(
                f"阈值敏感性表不符: {sens}",
                expected={"5%: NS✓/AI✗": True, "15%: AI✓": True, "20%: AI✓": True},
                actual=sens, param_key="D11",
            )
        sens_str = "; ".join(
            f"thr={thr * 100:.0f}%:NS={'✓' if rn == '闭环' else '✗'}/"
            f"AI={'✓' if ra == '闭环' else '✗'}" for thr, rn, ra in sens)
        return {"detail": sens_str + " (主文档行 3294-3298)", "sens": sens}

    # ------------------------------------------------------------ 6) 跨域口径
    def validate_cross_domain(self) -> dict:
        """6) 跨域口径：平台差>0（严格跨域未闭环，README ④第 5 步）。"""
        cfg = self.config
        deff_dev = abs(cfg.DEFF_AI - cfg.DEFF_NS) / cfg.DEFF_NS
        ok = deff_dev > 0.0
        if not ok:
            raise AIQValidationError(
                "跨域平台差应为正", expected="> 0", actual=deff_dev, param_key="D11",
            )
        return {
            "detail": f"|DEFF_AI-DEFF_NS|/DEFF_NS={deff_dev * 100:.2f}% "
                      f"(T06 未闭环 = 反解偏差>15% + 跨域平台差 1.3% 双重原因)",
            "deff_dev": deff_dev,
        }

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real(self) -> dict:
        """7) 真实模型对照（真实 λ=0.8516、DEFF=1.5920 代入闭环判据）。"""
        cfg = self.config
        rd = self._get_real_data()   # 惰性导入 / 注入的 _real_data
        lam_grid, deff_lookup = self._lookup()
        if not rd.has_real():
            # 审计回退：真实数据缺失时回退审计 λ=0.852
            ls_ns = float(np.interp(cfg.DEFF_NS, deff_lookup, lam_grid))
            ls_ai = float(np.interp(cfg.DEFF_AI, deff_lookup, lam_grid))
            _, res_ns = judge(ls_ns, cfg.DEV_REF, cfg.THR)
            _, res_ai = judge(ls_ai, cfg.DEV_REF, cfg.THR)
            ok = (res_ns == "未闭环") and (res_ai == "未闭环")
            if not ok:
                raise AIQValidationError(
                    f"回退反解未闭环判据不满足: NS={res_ns}, AI={res_ai}",
                    expected={"NS": "未闭环", "AI": "未闭环"},
                    actual={"ns": res_ns, "ai": res_ai}, param_key="D11",
                )
            return {"detail": f"NS/AI 未闭环（真实数据未就绪）",
                    "tag": "[审计回退]", "fallback": True}
        tag = "[真实实测]"
        lam_real = rd.get("curvature.lambda_ratio", cfg.DEV_REF)
        deff_real = rd.get("curvature.DEFF_plat", cfg.DEFF_AI)
        ls_ai_real = float(np.interp(deff_real, deff_lookup, lam_grid))
        dev_real, res_real = judge(ls_ai_real, lam_real, cfg.THR)
        okr1 = res_real == "未闭环"
        if not okr1:
            raise RealModelMismatchError(
                f"真实反解判定不符: {res_real}",
                expected="未闭环", actual=res_real, param_key="D11",
            )
        # 真实跨域口径：实测 DEFF vs NS 目标
        deff_dev_r = abs(deff_real - cfg.DEFF_NS) / cfg.DEFF_NS
        okr2 = deff_dev_r > cfg.CROSS_DOMAIN_MIN
        if not okr2:
            raise RealModelMismatchError(
                f"真实跨域口径差不足: {deff_dev_r * 100:.2f}%",
                expected=cfg.CROSS_DOMAIN_MIN, actual=deff_dev_r, param_key="D11",
            )
        return {
            "detail": (f"{tag} 真实 DEFF={deff_real:.4f} 反解 λ*={ls_ai_real:.3f} "
                       f"vs 实测 λ={lam_real:.4f}, 偏差={dev_real * 100:.1f}% -> "
                       f"{res_real}（T06 一致）; 真实跨域口径 "
                       f"|{deff_real:.4f}-{cfg.DEFF_NS}|/{cfg.DEFF_NS}="
                       f"{deff_dev_r * 100:.2f}%（严格跨域口径差，叠加反解偏差构成未闭环）"),
            "source": rd.source_tag(), "tag": tag,
            "lam_real": lam_real, "deff_real": deff_real,
            "ls_ai_real": ls_ai_real, "dev_real": dev_real, "res_real": res_real,
            "deff_dev_r": deff_dev_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "threshold", self.validate_threshold),
            (2, "boundary", self.validate_boundary),
            (3, "doc_values", self.validate_doc_values),
            (4, "mc_inversion", self.validate_mc_inversion),
            (5, "sensitivity", self.validate_sensitivity),
            (6, "cross_domain", self.validate_cross_domain),
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
    """D11 验证编排：配置层→合成层→验证层→报告层 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="D11 closure_thr 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()              # ① 配置层（env > YAML > JSON > 默认）
    synth = LookupSynthesizer(cfg)             # ② 合成层
    report = ReportGenerator()                 # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)   # ③ 验证层

    print("=" * 74)
    print("D11 closure_thr 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: THR={cfg.THR} DEV_REF={cfg.DEV_REF} MC_N={cfg.MC_N} "
          f"DEFF_NS={cfg.DEFF_NS} DEFF_AI={cfg.DEFF_AI} GRID_LAM={cfg.GRID_LAM} "
          f"CROSS_DOMAIN_MIN={cfg.CROSS_DOMAIN_MIN} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "d11_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "d11_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "d11_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
