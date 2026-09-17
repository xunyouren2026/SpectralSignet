# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 统一参数数据层（所有参数值零硬编码）
=====================================================================
本模块是 59 个 verify.py 读取"参数值"（默认值/阈值/实测参考/网格范围）
的唯一入口。所有值集中存储在 _params_data.json（纯数据文件），并按
以下优先级解析：

  1. 真实测量（_real_metrics.json，harness 实测）——最高优先级；
  2. 集中配置（_params_data.json，参数表默认值/参考值）——次优先；
  3. 调用方传入的 default——兜底（仅在配置缺失时使用，正常不应触发）。

用法：
  import _params as P
  P.get("T01", "PLATFORM", 1.56)      # 读 T01 的 PLATFORM 值（缺失回退 1.56）
  P.get_float("D09", "LAM_MEAS", 0.852)
  P.get_list("T01", "WB8", [1.5922, ...])
  P.get_grid("D09", "GRID_LAM", (0.05, 1.0, 40))

设计原则：
  - 代码中不写任何参数数值（除算法逻辑常量 SEED/EPS 等）；
  - 新增/修改参数值只需编辑 _params_data.json，无需改任何 verify.py；
  - 数据文件纯 JSON，可由脚本自动生成/校验，便于审计与版本管理。
"""
import os
import sys
import json

# --- 通用配置：自动探测路径（本文件同目录 _cfg.py），零硬编码 ---
_HERE = os.path.dirname(os.path.abspath(__file__))   # 本文件所在目录（参数/）
if _HERE not in sys.path:                            # 确保同目录模块可导入
    sys.path.insert(0, _HERE)
import _cfg                                        # 通用路径配置

# 数据文件（均位于参数目录，相对定位，无硬编码）
_PARAMS_JSON = os.path.join(_HERE, "_params_data.json")   # 参数值集中配置
_REAL_JSON = os.path.join(_HERE, "_real_metrics.json")    # 真实测量存档

# 缓存：避免重复读盘
_params_cache = None    # _params_data.json 解析结果
_real_cache = None      # _real_metrics.json 解析结果


# ---------------------------------------------------------------- 加载
def _load_json(fp: str) -> dict | None:
    """读取 JSON 文件；文件缺失/损坏返回 None（不抛异常）。"""
    if not os.path.isfile(fp):                     # 文件不存在 → 返回 None
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:  # UTF-8 读取（中文注释友好）
            return json.load(f)                     # 解析为 dict
    except Exception:
        return None                                 # 解析失败 → 返回 None


def _params() -> dict:
    """加载参数集中配置（_params_data.json），带缓存。"""
    global _params_cache
    if _params_cache is None:                       # 首次调用才读盘
        _params_cache = _load_json(_PARAMS_JSON) or {}
    return _params_cache


def _real() -> dict | None:
    """加载真实测量存档（_real_metrics.json），带缓存。"""
    global _real_cache
    if _real_cache is None:                         # 首次调用才读盘
        _real_cache = _load_json(_REAL_JSON)
    return _real_cache


def _real_lookup(param: str, key: str):
    """在真实测量存档中按（参数,键）查找。

    映射约定：
      - 工程/谱集中/曲率指标 → real[section][key]（如 arch.n_layers）
      - 参数级专属键 → real["params"][param][key]（暂未使用，预留）
    返回找到的值，未找到返回 None。
    """
    m = _real()
    if m is None:                                   # 无真实数据 → 未找到
        return None
    # 尝试"参数级"位置：_real_metrics.json 的 params 段（预留扩展）
    params_node = m.get("params") if isinstance(m, dict) else None
    if isinstance(params_node, dict):
        node = params_node.get(param)
        if isinstance(node, dict) and key in node:
            return node[key]
    # 尝试"指标级"位置：真实测量常用键
    if key in m:
        return m[key]
    for section in ("arch", "engine", "spectral", "curvature", "aiq"):
        node = m.get(section)
        if isinstance(node, dict) and key in node:
            return node[key]
    return None


# ---------------------------------------------------------------- 读取接口
def get(param: str, key: str, default=None):
    """读取参数值，优先级：真实测量 > 集中配置 > default。

    参数：
      param:   参数标识（如 "T01"、"D09"、"A01"）
      key:     参数键名（如 "PLATFORM"、"LAM_MEAS"）
      default: 兜底默认值（仅在配置缺失时使用，正常不应触发）
    """
    # 1) 真实测量优先（真实数据是"正常数据"的最高权威）
    real_v = _real_lookup(param, key)
    if real_v is not None:
        return real_v
    # 2) 集中配置（_params_data.json）
    node = _params().get(param)
    if isinstance(node, dict) and key in node:
        return node[key]
    # 3) 兜底默认值
    return default


def get_float(param: str, key: str, default: float) -> float:
    """读取浮点参数值（带类型强制与容错）。"""
    v = get(param, key, default)                    # 按优先级取值
    try:
        return float(v)                             # 转 float；失败则回退默认
    except (TypeError, ValueError):
        return float(default)


def get_int(param: str, key: str, default: int) -> int:
    """读取整型参数值（带类型强制与容错）。"""
    v = get(param, key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def get_list(param: str, key: str, default: list | None = None) -> list:
    """读取列表参数值（如白盒 8 帧序列、网格数组）。"""
    v = get(param, key, default)
    if isinstance(v, list):                         # 已是列表 → 直接返回
        return v
    try:
        return list(v)                              # 可迭代 → 转列表
    except TypeError:
        return list(default) if default is not None else []


def get_grid(param: str, key: str, default: tuple) -> tuple:
    """读取网格参数值：(start, stop, count) 三元组。"""
    v = get(param, key, default)                    # 可能是 list 或 tuple
    try:
        return tuple(float(x) if i < 2 else int(x)
                     for i, x in enumerate(v))      # 前两项 float，第三项 int
    except (TypeError, ValueError):
        return tuple(default)


def source_tag() -> str:
    """返回数据来源标注（用于输出显示）。"""
    if _real() is not None:                         # 真实测量可用
        return "真实实测+集中配置"
    return "集中配置(无真实测量)"


# ---------------------------------------------------------------- 自检（调试用）
if __name__ == "__main__":
    # 打印样例：验证优先级正确
    print("source:", source_tag())
    print("T01 PLATFORM =", get("T01", "PLATFORM", 1.56))
    print("D09 LAM_MEAS =", get("D09", "LAM_MEAS", 0.852))
    print("real DEFF    =", get("T01", "DEFF_plat", None))
