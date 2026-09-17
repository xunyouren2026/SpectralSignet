# -*- coding: utf-8 -*-
"""
AIQ 几何指纹插件 — 通用配置模块（零硬编码路径）
=====================================================================
设计目标：让整个插件在**任何电脑、任何目录**下开箱即用，不包含任何
用户特定路径（如 C:\\Users\\xxx\\...）。所有路径通过以下三种方式解析：

  1. 自动探测（默认）：从当前文件位置向上逐级查找关键目录
     （_models/ 模型库、AIQ/ 插件根目录），找到即用；
  2. 环境变量覆盖：AIQ_ROOT（项目根）、AIQ_MODELS_DIR（模型库目录）
     优先级高于自动探测，适合 CI/服务器/多项目场景；
  3. 命令行参数（harness 入口）：--model-dir 显式指定，优先级最高。

用法：
  import _cfg
  root = _cfg.project_root()          # 项目根目录（含 _models/ 的目录）
  mdir = _cfg.models_dir()            # 模型库目录（自动探测或 env）
  p = _cfg.model_path("Qwen2.5-0.5B") # 定位具体模型目录（找不到返回 None）
  p = _cfg.resolve("path/to/file")    # 相对项目根的路径解析（通用化核心）
"""
import os

# ---------------------------------------------------------------- 常量
# 本项目内部相对路径（相对"项目根"而非绝对路径，保证可移植）
_MODELS_SUBDIR = "_models"       # 模型库子目录名（默认约定）
_AIQ_SUBDIR = "AIQ"              # 插件目录名（默认约定）
_ROOT_MARKERS = ("_models", "AIQ")  # 用于向上探测的"根标记"子目录


# ---------------------------------------------------------------- 环境变量工具
def _env(name: str, default: str | None = None) -> str | None:
    """读取环境变量，空串视为未设置，返回 None 表示未提供。"""
    val = os.environ.get(name, "").strip()
    return val if val else default


# ---------------------------------------------------------------- 目录探测
def _this_dir() -> str:
    """返回本模块所在目录（绝对路径）。"""
    return os.path.dirname(os.path.abspath(__file__))


def project_root() -> str:
    """定位项目根目录。

    探测策略（按优先级）：
      1. 环境变量 AIQ_ROOT（用户显式指定）；
      2. 从本文件所在目录向上逐级查找，直到找到同时含 _models/ 的目录
         （此时该目录即项目根）；
      3. 兜底：返回本文件所在目录的上一级（AIQ/参数 -> AIQ）。
    """
    # 1) 环境变量优先
    root = _env("AIQ_ROOT")
    if root and os.path.isdir(root):
        return root
    # 2) 向上探测：检查 dir 是否含任一"根标记"
    cur = _this_dir()
    while True:
        for marker in _ROOT_MARKERS:
            if os.path.isdir(os.path.join(cur, marker)):
                return cur
        parent = os.path.dirname(cur)
        if parent == cur:          # 到达文件系统根，停止
            break
        cur = parent
    # 3) 兜底：AIQ/参数 的上一级即 AIQ 插件目录
    return os.path.dirname(_this_dir())


def models_dir() -> str:
    """定位模型库目录（_models/）。

    优先级：环境变量 AIQ_MODELS_DIR > 项目根下的 _models > 自动探测的
    模型目录（父目录含 model.safetensors 的最近 _models）。
    """
    # 1) 环境变量显式指定
    env = _env("AIQ_MODELS_DIR")
    if env and os.path.isdir(env):
        return env
    # 2) 项目根下默认位置
    root = project_root()
    candidate = os.path.join(root, _MODELS_SUBDIR)
    if os.path.isdir(candidate):
        return candidate
    # 3) 从本文件向上探测最近的 _models
    cur = _this_dir()
    while True:
        cand = os.path.join(cur, _MODELS_SUBDIR)
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return candidate          # 都不存在则返回理论路径（调用方自行处理）


def model_path(name: str) -> str | None:
    """定位指定名称的模型目录；不存在返回 None（供诊断项使用）。

    参数：
      name: 模型目录名，如 "Qwen2.5-0.5B" / "Qwen2.5-0.5B-Instruct"
    """
    mdir = models_dir()
    p = os.path.join(mdir, name)
    return p if os.path.isdir(p) else None


def resolve(rel: str) -> str:
    """将相对项目根的路径解析为绝对路径（通用化核心）。

    参数：
      rel: 相对项目根的路径，如 "AIQ/phi_pairs_all.npy"、"outputs/report.md"
    返回：绝对路径（不保证存在，由调用方判断）。
    """
    return os.path.normpath(os.path.join(project_root(), rel))


# ---------------------------------------------------------------- 便捷别名
# 供 59 个 verify.py 统一调用：不感知具体电脑路径，只按相对位置取文件
DATA_REL = "AIQ"                     # 插件目录相对项目根
PHI_PAIRS_REL = os.path.join("AIQ", "phi_pairs_all.npy")  # 曲率对数据文件（旧路径）
PHI_PAIRS_LOCAL = "phi_pairs_all.npy"  # 插件内同目录副本（迁移后优先）


def phi_pairs_path() -> str:
    """曲率对数据文件（phi_pairs_all.npy）的通用定位路径。

    探测优先级：
      1. 环境变量 AIQ_PHI_PAIRS（用户显式指定）——最高优先；
      2. 本文件（params/）同目录的副本 phi_pairs_all.npy（插件内自包含）；
      3. 相对项目根 AIQ/ 的原始位置（工作区场景回退）。
    返回：第一个存在的绝对路径；都不存在则返回候选1（调用方自行处理）。
    """
    # 1) 环境变量显式指定
    env = _env("AIQ_PHI_PAIRS")
    if env and os.path.isfile(env):
        return env
    # 2) 插件内同目录副本（自包含优先——迁移到任意位置仍可用）
    local = os.path.join(_this_dir(), PHI_PAIRS_LOCAL)
    if os.path.isfile(local):
        return local
    # 3) 工作区回退：项目根 AIQ/ 下原始位置
    return resolve(PHI_PAIRS_REL)


# ---------------------------------------------------------------- 自检（调试用）
if __name__ == "__main__":
    # 打印当前解析到的路径，便于验证"零硬编码"是否生效
    print("project_root =", project_root())
    print("models_dir   =", models_dir())
    print("phi_pairs    =", phi_pairs_path())
    print("qwen_b       =", model_path("Qwen2.5-0.5B"))
    print("qwen_i       =", model_path("Qwen2.5-0.5B-Instruct"))
