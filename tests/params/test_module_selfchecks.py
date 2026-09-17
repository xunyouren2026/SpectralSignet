# -*- coding: utf-8 -*-
"""模块级 __main__ 自检块执行测试（覆盖率补充）
=====================================================================
各共享模块尾部带有 `if __name__ == "__main__":` 自检块（断言+打印），
以 `runpy.run_path(..., run_name="__main__")` 在同一进程内执行，
使自检块内的代码行纳入覆盖率统计（pytest-cov 跟踪同进程全部执行）。

注意：
  - 自检块均为无副作用断言/打印（_factory 临时写删 env；_perf 使用
    tempfile 且自动清理），可安全重复执行；
  - 执行后 logger 会被重配置为"当前 stderr"，不影响后续用例
    （各日志用例自行 configure）。
=====================================================================
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_PARAMS = Path(__file__).resolve().parents[2] / "params"

_MODULES = [
    "_errors.py",
    "_logging.py",
    "_perf.py",
    "_factory.py",
    "_cfg.py",
    "_params.py",
    "_usage_demo.py",  # 演示脚本：__main__ 顺序执行 4 案例（缺数据时优雅跳过）
]


@pytest.mark.parametrize("rel", _MODULES)
def test_module_main_selfcheck(rel):
    """以 __main__ 方式执行模块自检块（覆盖 __main__ 守卫内代码行）。"""
    runpy.run_path(str(_PARAMS / rel), run_name="__main__")  # 自检块内部自行断言
