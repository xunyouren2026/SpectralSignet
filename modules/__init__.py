"""
aiq-geometric-forensics — AI 几何指纹诊断专业包
=====================================================================
一个可导入的 Python 包，把 59 参数验证体系的真实测量引擎 + 三大功能
（健康诊断 / 家族溯源 / 基准对比）封装为统一、专业、无硬编码的接口。

公开 API：
  diagnose(name, measurement=None)      几何健康度画像（AIQ+Gamma+曲率）
  format_health(rep)                    健康报告文本
  trace(target, reference, ...)         家族溯源判定（MAD 最近邻 + SFT 稳定性）
  format_trace(rep)                     溯源报告文本
  compare(target, reference, ...)       与基准模型逐维对比
  format_compare(rep)                   对比报告文本
  measure(model_dir, ngen=64)           真实模型测量引擎（需 torch/transformers）
  Metrics(model)                        共享数据层（真实>存档>兜底）
"""
from . import observation as observation
from . import schema as schema
from . import selfcheck as selfcheck
from .compare import compare as compare
from .compare import format_compare as format_compare
from .data import Metrics, find_baseline_models, sanitize_path
from .family import family_tree as family_tree
from .family import format_family as format_family
from .forensics import fingerprint
from .forensics import format_trace as format_trace
from .forensics import trace as trace
from .harness import measure as measure
from .health import (
    aiq_factors,
    calibrate_deff_band,
    compressibility,
    depth_profile,
    health_verdicts,
)
from .health import diagnose as diagnose
from .health import format_health as format_health
from .report import evidence_badge as evidence_badge
from .report import render_compare as render_compare
from .report import render_factor_panel as render_factor_panel
from .report import render_full_page as render_full_page
from .report import render_health as render_health
from .report import render_trace as render_trace
from .report import svg_bars as svg_bars
from .report import svg_lines as svg_lines
from .selfcheck import render_report as render_selfcheck
from .selfcheck import run_selfcheck as run_selfcheck
from .stability import format_stability as format_stability
from .stability import stability as stability

__all__ = [
    "Metrics", "find_baseline_models", "sanitize_path",
    "diagnose", "format_health", "aiq_factors", "depth_profile",
    "health_verdicts", "compressibility", "calibrate_deff_band",
    "trace", "format_trace", "fingerprint",
    "compare", "format_compare",
    "stability", "format_stability",
    "family_tree", "format_family",
    "measure",
    "evidence_badge", "render_health", "render_trace", "render_compare",
    "render_factor_panel", "render_full_page", "svg_bars", "svg_lines",
    "run_selfcheck", "render_selfcheck", "selfcheck",
    "observation", "schema",
    "params_runner",
]

__version__ = "3.3.0"

