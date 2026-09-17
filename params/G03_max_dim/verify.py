# -*- coding: utf-8 -*-
"""G03 max_dim 跳过超大输出层阈值 — 以维度 d 为判据的防 OOM 截断验证
====================================================================
验证目标（编号列表，与原脚本逐项一致）：
  1. 对典型层维度执行 capture 判定：d <= max_dim 捕获，d > max_dim 跳过
  2. 边界行为：d = 8192 捕获（8192 > 8192 为 False），d = 8193 跳过
  3. 协方差矩阵内存估计：C = H^T H/(B-1) in R^{d x d}，bytes = d^2 * 4
     - d=896   (Qwen 隐藏层) -> ~3.2 MB 安全
     - d=152064 (lm_head)   -> ~92.5 GB OOM
  4. 模拟插件 _auto_layer_names 的 max_dim 过滤逻辑，返回正确目标层列表
  5. 捕获层 SVD 谱分析可运行性（小内存验证，Gamma 有限）
  6. 真实模型对照（Qwen2.5-0.5B-Instruct：hidden=896 捕获，lm_head=151936 跳过）
数据源：
  主文档行 5188-5308（G03 max_dim 章节，README ⑤ 实测表）
  源码 plugin.py（第 48-56、115-117 行 max_dim 检查）
  《参数审计与实验报告.txt》行 78（状态=理论）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  G03Config / ConfigFactory / G03Validator / ReportGenerator / main
  无独立合成器（验证主体为纯判定函数；SVD 小矩阵在引擎内生成）。
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
from _errors import AIQValidationError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 G03Config（pydantic 优先；dataclass 回退） ----
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


class G03Config(_ConfigModelBase):
    """G03 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 G03 节点键名一一对应；取值优先级：
    环境变量 AIQ_G03_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    MAX_DIM: int = 8192        # max_dim 固定值（README ①）：超大输出层跳过阈值
    BYTES_PER_ELEM_F32: int = 4  # float32 协方差元素字节数（README ②）
    SEED: int = 0              # 合成数据固定种子（规范 §3 / H01 语义）
    N_COMPONENTS: int = 3      # Gamma 前 3 主成分（README ② / B01）
    GAMMA_SVD_B: int = 256     # 捕获层谱分析规模（token 数）
    GAMMA_SVD_D: int = 896     # 捕获层谱分析规模（维度，README ⑤）
    OOM_THRESHOLD_GB: float = 24.0  # lm_head 协方差内存远超单卡显存判据（README ②）
    REAL_LM_HEAD_DIM: int = 151936  # Qwen2.5-0.5B vocab_size（真实 tokenizer 配置）
    EPS: float = 1e-12         # 除零保护小量


# ---- 第二层：配置工厂 ConfigFactory（实例化 G03Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """G03 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 G03Config。"""

    def build(self) -> G03Config:
        """构建 G03Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(G03Config, "G03")


# ---------------- 纯函数工具（与原脚本逐项一致） ----------------
def should_capture(d: int, max_dim: int) -> bool:
    """与 plugin.py 第 115-117 行一致：d > max_dim 则跳过。"""
    assert isinstance(d, int) and d > 0, f"层维度 d={d} 非法"
    return d <= max_dim


def covariance_bytes(d: int, bytes_per_elem: int) -> int:
    """协方差矩阵 C in R^{d x d} 的内存占用（字节）。"""
    assert isinstance(d, int) and d > 0, f"层维度 d={d} 非法"
    return d * d * bytes_per_elem


def auto_layer_names(layer_dims: list, max_dim: int) -> list:
    """模拟 auto 层检测：返回捕获层名列表，打印跳过层（plugin.py _auto_layer_names）。"""
    assert isinstance(layer_dims, list) and len(layer_dims) > 0, "layer_dims 为空"
    captured = []
    for name, d in layer_dims:
        assert isinstance(d, int) and d > 0, f"层 {name} 维度 d={d} 非法"
        if d > max_dim:
            continue
        captured.append(name)
    return captured


def gamma_svd(H: np.ndarray, eps: float) -> float:
    """捕获层 SVD 谱 Gamma（前 3 奇异值占比，spectral_analysis.py 口径）。"""
    H = np.asarray(H, dtype=np.float64)
    assert H.ndim == 2 and H.shape[0] >= 2, f"H 需二维且 B>=2，实际 {H.shape}"
    Hc = H - H.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Hc, compute_uv=False)
    return float(S[:3].sum() / (S.sum() + eps))


# ---- 第四层：验证引擎 G03Validator（6 项验证 + 结构化日志 + 类型化异常） ----
class G03Validator(_EngineBase):
    """G03 验证引擎：顺序执行 6 项验证（结构化 JSON 日志 + 类型化异常）。

    无合成器：验证主体为纯判定函数（should_capture/covariance_bytes 等），
    唯一随机数据（SVD 小矩阵）在步骤 5 内直接生成。
    """

    def __init__(
        self,
        config: G03Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth=None, reporter=reporter)
        self._real_data = real_data

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 典型层判定
    def validate_layers(self) -> dict:
        """1) 典型层维度表：捕获/跳过判定逐层正确。"""
        cfg = self.config
        layers = [
            ("Qwen2.5-0.5B k_proj", 896),
            ("GPT-2 隐藏层", 768),
            ("LLaMA 7B 隐藏层", 4096),
            ("DeepSeek-V3 隐藏层", 7168),
            ("Qwen2.5-0.5B lm_head", 152064),
            ("Qwen2.5-0.5B 嵌入层", 152064),
            ("GPT-2 lm_head", 50257),
            ("LLaMA 7B lm_head", 32000),
            ("DeepSeek-V3 lm_head", 129280),
        ]
        parts = []
        ok1 = True
        for name, d in layers:
            cap = should_capture(d, cfg.MAX_DIM)
            expected_cap = d <= cfg.MAX_DIM
            ok = (cap == expected_cap)
            ok1 = ok1 and ok
            status = "捕获" if cap else "跳过"
            parts.append(f"{name}: d={d} -> {status}")
            if not ok:
                raise AIQValidationError(
                    f"层 {name} d={d} 判定 {cap} ≠ 期望 {expected_cap}",
                    expected=expected_cap, actual=cap, param_key="G03",
                )
        return {"detail": "[1] 典型层判定: " + "; ".join(parts), "n_layers": len(layers), "ok": ok1}

    # ------------------------------------------------------------ 2) 边界行为
    def validate_boundary(self) -> dict:
        """2) 边界行为：d=MAX_DIM 捕获、d=MAX_DIM+1 跳过。"""
        cfg = self.config
        b1 = should_capture(cfg.MAX_DIM, cfg.MAX_DIM)
        b2 = should_capture(cfg.MAX_DIM + 1, cfg.MAX_DIM)
        ok2 = b1 and (not b2)
        if not ok2:
            raise AIQValidationError(
                f"边界行为异常: d={cfg.MAX_DIM}->{b1}（应捕获）, "
                f"d={cfg.MAX_DIM+1}->{b2}（应跳过）",
                expected=[True, False], actual=[b1, b2], param_key="G03",
            )
        return {
            "detail": (f"[2] d={cfg.MAX_DIM} (==max_dim) -> 捕获; "
                       f"d={cfg.MAX_DIM+1} (max_dim+1) -> 跳过"),
            "capture_eq": b1, "skip_plus1": not b2,
        }

    # ------------------------------------------------------------ 3) 协方差内存
    def validate_covariance_mem(self) -> dict:
        """3) 协方差矩阵内存估计：隐藏层安全、lm_head 必然 OOM。"""
        cfg = self.config
        pairs = [("Qwen 隐藏层 k_proj", 896), ("LLaMA 隐藏层", 4096),
                 ("DeepSeek-V3 隐藏层", 7168), ("lm_head", 152064)]
        parts = []
        ok3 = True
        for name, d in pairs:
            gb = covariance_bytes(d, cfg.BYTES_PER_ELEM_F32) / 1e9
            safe = gb < 1.0
            if name == "lm_head":
                ok_lm = gb > cfg.OOM_THRESHOLD_GB
                ok3 = ok3 and ok_lm
                parts.append(f"{name}: d={d} -> {gb:.2f} GB（必然 OOM，需跳过）")
                if not ok_lm:
                    raise AIQValidationError(
                        f"lm_head 协方差 {gb:.1f} GB 未超单卡显存阈值",
                        expected=f">{cfg.OOM_THRESHOLD_GB} GB", actual=gb, param_key="G03",
                    )
            else:
                ok3 = ok3 and safe
                parts.append(f"{name}: d={d} -> {gb:.2f} GB（安全）")
                if not safe:
                    raise AIQValidationError(
                        f"层 {name} 协方差 {gb:.2f} GB 异常偏大",
                        expected="<1.0 GB", actual=gb, param_key="G03",
                    )
        return {"detail": "[3] 协方差内存估计: " + "; ".join(parts), "ok": ok3}

    # ------------------------------------------------------------ 4) auto 层检测
    def validate_auto_names(self) -> dict:
        """4) 模拟 _auto_layer_names：捕获层列表与期望逐项一致。"""
        cfg = self.config
        layers = [
            ("Qwen2.5-0.5B k_proj", 896),
            ("GPT-2 隐藏层", 768),
            ("LLaMA 7B 隐藏层", 4096),
            ("DeepSeek-V3 隐藏层", 7168),
            ("Qwen2.5-0.5B lm_head", 152064),
            ("Qwen2.5-0.5B 嵌入层", 152064),
            ("GPT-2 lm_head", 50257),
            ("LLaMA 7B lm_head", 32000),
            ("DeepSeek-V3 lm_head", 129280),
        ]
        capture_list = auto_layer_names(layers, cfg.MAX_DIM)
        expected_captured = [name for name, d in layers if d <= cfg.MAX_DIM]
        ok4 = capture_list == expected_captured
        if not ok4:
            raise AIQValidationError(
                f"捕获层 {capture_list} ≠ 期望 {expected_captured}",
                expected=expected_captured, actual=capture_list, param_key="G03",
            )
        return {
            "detail": (f"[4] 捕获层: {capture_list}; 目标层数 = {len(capture_list)} "
                       f"(期望 {len(expected_captured)})"),
            "captured": capture_list, "n_captured": len(capture_list),
        }

    # ------------------------------------------------------------ 5) 捕获层 SVD 可运行
    def validate_capture_svd(self) -> dict:
        """5) 捕获层 SVD 谱分析可运行性（小内存验证，Gamma 有限且∈(0,1)）。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        H = rng.standard_normal((cfg.GAMMA_SVD_B, cfg.GAMMA_SVD_D))
        gamma = gamma_svd(H, cfg.EPS)
        mem_mb = covariance_bytes(cfg.GAMMA_SVD_D, cfg.BYTES_PER_ELEM_F32) / 1e6
        ok5 = bool(np.isfinite(gamma)) and 0.0 < gamma < 1.0
        if not ok5:
            raise AIQValidationError(
                f"捕获层 Gamma={gamma:.6f} 非有限或越界",
                expected="(0,1)", actual=gamma, param_key="G03",
            )
        return {
            "detail": (f"[5] H: ({cfg.GAMMA_SVD_B}, {cfg.GAMMA_SVD_D}), "
                       f"协方差 {mem_mb:.1f} MB; Gamma(前3奇异值占比) = {gamma:.6f}"),
            "gamma": gamma, "mem_mb": mem_mb,
        }

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：hidden=896 捕获、lm_head=151936 跳过（惰性读取）。"""
        cfg = self.config
        rd = self._get_real_data()
        tag = rd.source_tag()
        hidden_real = rd.get("arch.hidden", None)
        w_bytes_real = rd.get("arch.W_bytes", None)
        if hidden_real is None:  # 真实架构缺失：审计回退
            return {
                "detail": (f"[6] [{tag}] 真实架构数据不可用，回退审计架构 "
                           f"(hidden=896 捕获 / lm_head 152064 跳过，见 [1]-[3])"),
                "source": tag, "real_mode": False,
            }
        cap_hidden = should_capture(int(hidden_real), cfg.MAX_DIM)
        cap_lm_head = should_capture(cfg.REAL_LM_HEAD_DIM, cfg.MAX_DIM)
        ok6a = cap_hidden and (not cap_lm_head)
        mem_hidden_gb = covariance_bytes(int(hidden_real), cfg.BYTES_PER_ELEM_F32) / 1e9
        mem_lm_gb = covariance_bytes(cfg.REAL_LM_HEAD_DIM, cfg.BYTES_PER_ELEM_F32) / 1e9
        if not ok6a:
            raise RealModelMismatchError(
                f"真实架构判定异常: hidden={hidden_real}->{cap_hidden}, "
                f"lm_head={cfg.REAL_LM_HEAD_DIM}->{cap_lm_head}",
                expected=[True, False], actual=[cap_hidden, cap_lm_head], param_key="G03",
            )
        extra = ""
        if w_bytes_real is not None:
            extra = f"; 真实权重 W = {w_bytes_real / 1e9:.3f} GB (fp32 实测)"
        return {
            "detail": (f"[6] [{tag}] 真实 hidden = {hidden_real} -> 捕获; "
                       f"真实 lm_head 输出维度 = {cfg.REAL_LM_HEAD_DIM} (vocab_size, "
                       f"> {cfg.MAX_DIM}) -> 跳过; vs 文档审计 lm_head 维度 152064，"
                       f"Δ={152064 - cfg.REAL_LM_HEAD_DIM}（以真实 tokenizer 配置为准）; "
                       f"hidden 协方差 {mem_hidden_gb * 1e3:.2f} MB（安全）; "
                       f"lm_head 协方差 {mem_lm_gb:.2f} GB（必然 OOM）{extra}"),
            "source": tag, "hidden_real": hidden_real, "mem_hidden_gb": mem_hidden_gb,
            "mem_lm_gb": mem_lm_gb,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "layers", self.validate_layers),
            (2, "boundary", self.validate_boundary),
            (3, "covariance_mem", self.validate_covariance_mem),
            (4, "auto_names", self.validate_auto_names),
            (5, "capture_svd", self.validate_capture_svd),
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
    """G03 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="G03 max_dim 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    engine = G03Validator(cfg, report, real_data=RD)

    print("=" * 74)
    print(f"G03 max_dim 跳过超大输出层阈值验证（四层工厂架构，max_dim={cfg.MAX_DIM}）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: MAX_DIM={cfg.MAX_DIM} OOM_THRESHOLD_GB={cfg.OOM_THRESHOLD_GB} "
          f"REAL_LM_HEAD_DIM={cfg.REAL_LM_HEAD_DIM}")
    print("=" * 74)

    if args.profile:
        res = profile_run(engine.run, out_dir, "g03_verify")
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "g03_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "g03_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
