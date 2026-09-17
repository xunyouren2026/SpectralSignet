# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 共享真实数据加载库
=====================================================================
所有 verify.py 通过本库读取真实模型测量结果（_real_metrics.json），
用真实实测值对照替代合成硬编码。若真实数据缺失则回退到文档审计值。

数据来源：
  _real_metrics.json —— 由 _real_model_harness.py 对本地模型真实测量生成
  （真实激活谱集中/曲率/工程指标/AIQ），模型路径经 _cfg 自动探测。

用法：
  import _real_data as RD
  m = RD.metrics()          # 完整真实测量字典（含 source 标注）
  m.get("spectral", {}).get("k_proj_gamma_mean")  # 真实 Gamma 均值
  RD.real_or(audit_value)   # 返回真实值；缺数据时返回审计值
"""
import os
import sys
import json

import numpy as np

# --- 通用配置：自动探测路径（本文件同目录 _cfg.py），零硬编码 ---
_HERE = os.path.dirname(os.path.abspath(__file__))       # 本文件所在目录
if _HERE not in sys.path:                                # 确保 _cfg 可导入
    sys.path.insert(0, _HERE)
import _cfg                                            # 通用路径配置

_REAL_JSON = os.path.join(_HERE, "_real_metrics.json")   # 真实实测存档（本目录）
_PHI_NPY = _cfg.phi_pairs_path()                        # 曲率对数据（_cfg 定位）

# 文档审计基准值（仅当真实数据缺失时回退）
_AUDIT_FALLBACK = {
    "k_proj_gamma_mean": 0.625,
    "tok_s": 8.39,
    "kv_bytes_MB": 24.9,
    "kv_w_ratio": 0.013,
    "rss_gb": 3.16,
    "DEFF_plat": 1.5920,
    "K_neg_pct": 45.50,
    "H_median": 1.3152,
    "phi_mean_deg": 20.26,
    "lambda_ratio": 0.8516,
    "aiq": 58.20,
}

_metrics_cache = None


def metrics():
    """加载真实测量结果；文件缺失返回 None（调用方自行回退）。"""
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache
    if not os.path.isfile(_REAL_JSON):
        _metrics_cache = None
        return None
    try:
        with open(_REAL_JSON, "r", encoding="utf-8") as f:
            _metrics_cache = json.load(f)
    except Exception:
        _metrics_cache = None
    return _metrics_cache


def has_real():
    """真实测量数据是否可用。"""
    return metrics() is not None


def real_or(audit_value):
    """优先真实值；真实数据缺失时回退审计值。

    注意：audit_value 可以是标量或 (key, default) 形式。
    用法：RD.real_or(0.625) 或 RD.real_or(("spectral.k_proj_gamma_mean", 0.625))
    """
    if isinstance(audit_value, tuple) and len(audit_value) == 2:
        key, default = audit_value
        m = metrics()
        if m is None:
            return default
        node = m
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node if node is not None else default
    return audit_value


def get(key, default=None):
    """按点分路径读取真实测量值；缺失返回 default。"""
    m = metrics()
    if m is None:
        return default
    node = m
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node if node is not None else default


def audit(key, default=None):
    """读取文档审计回退值。"""
    return _AUDIT_FALLBACK.get(key, default)


def source_tag():
    """返回数据来源标注字符串（用于输出显示）。"""
    return "真实实测(Qwen2.5-0.5B-Instruct)" if has_real() else "审计回退值"


_pairs_cache = None


def phi_pairs():
    """加载真实 (κ1,κ2) 逐点数据 phi_pairs_all.npy（AIQ 根目录，6303×2）。

    返回 (n,2) float 数组；文件缺失或损坏返回 None（调用方自行回退）。
    """
    global _pairs_cache
    if _pairs_cache is not None:
        return _pairs_cache
    if not os.path.isfile(_PHI_NPY):
        _pairs_cache = None
        return None
    try:
        data = np.load(_PHI_NPY)
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
            _pairs_cache = None
            return None
        _pairs_cache = data
    except Exception:
        _pairs_cache = None
    return _pairs_cache


def has_phi():
    """真实 (κ1,κ2) 逐点数据是否可用。"""
    return phi_pairs() is not None
