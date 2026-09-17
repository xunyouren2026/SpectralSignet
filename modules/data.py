"""
aiq-geometric-forensics.modules.data — 共享数据层（真实测量优先，集中回退）
=====================================================================
专业包数据约定：所有真实测量存档（JSON）通过本层读取，杜绝散落硬编码。

数据来源优先级（P0 > P1 > P2）：
  P0 真实测量：调用方显式传入的 measurement dict（由 harness 产生）
  P1 基准存档：本包 baselines/ 下按模型名匹配的真实实测 JSON
  P2 审计兜底：内置基线值（仅当 P0/P1 均缺失时使用，并标注 source=fallback）

用法：
  import modules.data as D                          # 包内相对导入
  m = D.Metrics("Qwen2.5-0.5B-Instruct")            # 构造（自动探测存档）
  m.get("spectral.k_proj_gamma_mean")               # 点分路径取值
  m.report()                                        # 数据来源标注
"""
from __future__ import annotations

import json
import os
from typing import Any

from . import observation, schema

# 模块级可观测 logger（副作用写入日志层；未配置时为 no-op，零测试影响）
_log = observation.get_logger(__name__)

# ---------------------------------------------------------------- 定位
# 本包内 baselines/ 目录：相对本模块向上两级（modules/ -> skills 根）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASELINE_DIR = os.path.join(_BASE_DIR, "baselines")

# ---------------------------------------------------------------- 审计兜底值
# 仅当 P0(真实测量)/P1(基准存档) 均缺失时使用。
# 专业约定：这些是"结构性占位"，语义上表示"指标存在但数值需实测"，
# 绝不可当作某特定模型的真实值。真实的健康/溯源/对比一律走
# harness.measure() 的实时测量（P0）或 baselines/ 存档（P1）。
_FALLBACK: dict[str, Any] = {
    "spectral.k_proj_gamma_mean": 0.0,        # 占位：待真实测量
    "spectral.proj_gamma": {},                # 占位：待真实测量
    "curvature.DEFF_plat": schema.PI_HALF,    # 理论锚点 π/2（与 health 一致性取自 schema）
    "curvature.DEFF_cv": 0.0,                 # 占位：待真实曲率分布（非某模型实测）
    "curvature.K_neg_pct": 0.0,               # 占位
    "curvature.H_median": 0.0,                # 占位
    "curvature.phi_mean_deg": 0.0,            # 占位
    "curvature.lambda_ratio": 0.0,            # 占位
    "aiq.AIQ": 0.0,                           # 占位：待真实 AIQ
}


class Metrics:
    """集中度量访问层。

    行为：
      - 优先使用构造时传入的 measurement（P0）；
      - 否则从 baselines/<model>.json 自动装载（P1）；
      - 仍缺失则用 _FALLBACK（P2）并置 source="fallback"。
    """

    def __init__(self, model_name: str = "Qwen2.5-0.5B-Instruct",
                 measurement: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self._data: dict[str, Any] = {}
        self.source: str = "none"

        if measurement:                                   # P0：显式真实测量
            self._data = measurement
            self.source = "real-inline"
            return

        path = os.path.join(_BASELINE_DIR, f"{model_name}.json")
        if os.path.isfile(path):                          # P1：基准存档
            try:
                with open(path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("存档顶层非对象")
                self._data = loaded
                self.source = "baseline"
                self._search_file = path
                _log.info("P1 基准存档模型=%s", model_name)
                return
            except (ValueError, json.JSONDecodeError, OSError) as e:
                # 损坏/不可读存档不崩溃：安全降级到兜底，并如实标注（不伪造实测）
                self.source = "fallback"
                self._data = {}
                self._corrupt_file = path
                _log.warning("存档损坏→回退兜底模型=%s err=%s", model_name, e)
                return

        self._data = {}                                   # P2：审计兜底
        self.source = "fallback"
        _log.warning("无真实存档→审计兜底模型=%s（占位非实测，报告会标注）", model_name)

    # ----------------------------------------------------------- 取值
    def _lookup(self, key: str) -> Any:
        """点分路径（如 'spectral.k_proj_gamma_mean'）递归取值；找不到返回 None。"""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        """带默认值的安全取值。"""
        v = self._lookup(key)
        if v is not None:
            return v
        if self.source == "fallback":                     # 兜底层仍缺 → 用全局回退
            return _FALLBACK.get(key, default)
        return default

    def get_float(self, key: str, default: float) -> float:
        v = self.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def get_list(self, key: str) -> list:
        v = self.get(key, None)
        return list(v) if isinstance(v, (list, tuple)) else []

    # ----------------------------------------------------------- 元信息
    def available(self) -> bool:
        """是否存在可用数据（任一来源）。"""
        return self.source != "none" or bool(self._data)

    def arch(self) -> dict[str, Any]:
        """架构字段；缺失时返回空 dict。"""
        a = self.get("arch", {})
        return dict(a) if isinstance(a, dict) else {}

    def report(self) -> str:
        """数据来源标注：真实测量 / 基准存档 / 审计兜底。"""
        label = {
            "real-inline": "真实测量（inline）",
            "baseline": f"真实实测存档 baselines/{self.model_name}.json",
            "fallback": "审计兜底值（无真实存档）",
            "none": "无数据",
        }.get(self.source, self.source)
        return f"数据源: {label}"


# ---------------------------------------------------------------- 便捷函数
def find_baseline_models() -> list:
    """列出 baselines/ 下可用基准模型名（按 JSON 文件名，去扩展名）。"""
    if not os.path.isdir(_BASELINE_DIR):
        return []
    return [f[:-5] for f in sorted(os.listdir(_BASELINE_DIR))
            if f.endswith(".json")]


def sanitize_path(value: Any) -> Any:
    """剔除测量 dict 中暴露用户绝对路径的敏感字段（如 model/<path>）。

    专业要求：诊断报告不应打印或泄漏任何人的盘符/用户名路径。
    兼容 POSIX('/') 与 Windows('\\') 两种路径分隔符。
    """
    def _leaf(p: str) -> str:
        # 取最后一个 '/' 或 '\\' 之后的部分（跨平台）
        head = p.replace("\\", "/").rstrip("/")
        return head.rsplit("/", 1)[-1] if "/" in head else p

    if isinstance(value, dict):
        return {
            k: (_leaf(v) if k == "model" and isinstance(v, str) else
                sanitize_path(v))
            for k, v in value.items()
        } if ("model" in value) else {
            k: sanitize_path(v) for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_path(v) for v in value]
    return value
