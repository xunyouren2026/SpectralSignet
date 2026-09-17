# -*- coding: utf-8 -*-
"""tests/params — 共享 pytest fixtures 与路径注入
=====================================================================
- sys.path 注入：插件根 + params/ + params/A01_model/，使 tests/params
  下可直接 `import verify` 与共享库（_factory/_errors/_logging/_perf/
  _real_data/_params/_cfg/_common）。
- fixtures：rng（固定种子 np.random.default_rng(0)）、cfg（A01Config
  实例）、synth（FingerprintSynthesizer）、engine（ValidatorEngine）。
- 全部使用固定种子（SEED=0），保证任何用例结果可复现。
=====================================================================
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---- 路径注入：插件根 + params + A01_model（按依赖顺序插入） ----
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # aiq-geometric-forensics/
_PARAMS = os.path.join(_ROOT, "params")
_A01 = os.path.join(_PARAMS, "A01_model")
for _p in (_ROOT, _PARAMS, _A01):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import verify  # noqa: E402  (注入路径后导入四层工厂模块)


@pytest.fixture
def rng() -> np.random.Generator:
    """固定种子随机数发生器：np.random.default_rng(0)（H01 seed=0 语义）。"""
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def cfg() -> "verify.A01Config":
    """A01Config 实例（直接构造，确定性；字段默认值与集中配置一致）。"""
    return verify.A01Config()


@pytest.fixture(scope="session")
def synth(cfg):
    """FingerprintSynthesizer 实例（seed=cfg.SEED=0，可复现）。"""
    return verify.FingerprintSynthesizer(cfg)


@pytest.fixture(scope="session")
def engine(cfg, synth) -> "verify.ValidatorEngine":
    """ValidatorEngine 实例（真实数据惰性导入路径）。"""
    return verify.ValidatorEngine(cfg, synth)
