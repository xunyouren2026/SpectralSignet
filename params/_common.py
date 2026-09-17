# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 统一工程样板（_common）
=====================================================================
把 59 个 verify.py 中重复的样板代码（stdout 编码、sys.path 注入、
底部汇总、退出码约定）抽为本模块统一函数，消除每脚本 ~20-45 行重复
（59 × 30 ≈ 1770 行样板 → 单点维护）。**绝不改变任何验证逻辑**——
各 verify.py 只需把头部样板替换为 `from _common import *` + 两次调用，
中部验证项与断言保持原样。

典型用法（verify.py 头部）：
  from _common import setup_env, finish   # 统一样板
  setup_env(__file__)                     # ① 编码 + sys.path + RD/P/_cfg 导入
  ...（常量 + 验证逻辑保持原样）...
  finish(passed, idx)                     # ② 底部汇总 + 返回退出码

边界：
  - setup_env 内部完成 sys.stdout 编码、参数目录 sys.path 注入；
  - 不 import numpy（验证逻辑自行导入），避免拖慢无 numpy 场景；
  - finish 打印统一汇总并返回 0/1（与既有约定完全一致）。
"""
from __future__ import annotations

import os
import sys

# ---- 自举：确保参数根目录（params/）在 sys.path（本模块位于 params/_common.py）----
_PARAM_ROOT = os.path.dirname(os.path.abspath(__file__))   # params/ 根
if _PARAM_ROOT not in sys.path:
    sys.path.insert(0, _PARAM_ROOT)                        # 使 _real_data/_params/_cfg 可导入

# 供各 verify.py 使用的共享库（经 setup_env 注入路径后可直接 import）
# 在模块级延迟导入：调用 setup_env 之后才能成功 import _params 等
_real_data = None      # 占位（由 _use_shared 填充）
_params = None
_cfg = None


def _use_shared() -> None:
    """惰性加载共享库（_real_data/_params/_cfg），仅首次调用导入。"""
    global _real_data, _params, _cfg, RD, P, CFG
    if _real_data is None:
        import _real_data
        _real_data = _real_data
    if _params is None:
        import _params
        _params = _params
    if _cfg is None:
        import _cfg
        _cfg = _cfg
    # 刷新便捷别名（首次导入后指向真实模块）
    RD, P, CFG = _real_data, _params, _cfg


def setup_env(script_path: str) -> tuple:
    """统一样板①：stdout 编码 + 参数目录 sys.path 注入 + 共享库导入。

    参数：
      script_path: 调用脚本的 __file__（本脚本目录 = 参数子目录）
    返回：(RD, P, CFG) 共享库三件套（等价旧代码 `import _real_data as RD`
           + `import _params as P` + `import _cfg`）。
    """
    # 1) stdout UTF-8（Windows 控制台默认 GBK，中文输出会乱码）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass   # 旧版 Python/无 TTY 环境不支持则静默降级
    # 2) 定位参数目录（本脚本的上级 = verify.py 所在目录的上级？）
    #    注意：本模块位于 params/ 下；script_path 是 verify.py 的绝对路径
    here = os.path.dirname(os.path.abspath(script_path))     # 参数子目录（A01_model/）
    param_dir = os.path.dirname(here)                        # 参数目录（params/）
    if param_dir not in sys.path:
        sys.path.insert(0, param_dir)                        # 前置注入共享库路径
    # 3) 惰性导入共享库
    _use_shared()
    return _real_data, _params, _cfg


def finish(passed: bool, n_items: int) -> int:
    """统一样板②：底部汇总 + 退出码。

    参数：
      passed: 全部通过标志（与各脚本 `passed` 语义一致）
      n_items: 验证项数（与各脚本 `idx` 语义一致）
    返回：0=全部通过，1=存在失败（与既有退出码约定一致）。
    """
    print("=" * 74)
    if passed:
        print(f"==== 汇总：全部 {n_items} 项 PASS ====")
    else:
        print(f"==== 汇总：存在 FAIL（共 {n_items} 项） ====")
    print("=" * 74)
    return 0 if passed else 1


def data_source_tag() -> str:
    """数据来源标签（真实实测 / 审计回退），供验证输出使用。

    依赖 _real_data 已导入（setup_env 后可用）；失败时回退 "审计回退"。
    """
    try:
        return "[真实实测]" if _real_data.has_real() else "[审计回退]"
    except Exception:
        return "[审计回退]"


# 便捷别名：保持旧代码风格（RD/P/CFG 小写习惯）
RD = _real_data
P = _params
CFG = _cfg