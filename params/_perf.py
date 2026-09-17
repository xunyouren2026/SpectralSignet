# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 性能工具集（_perf）
=====================================================================
面向指纹批量合成/插值/精度对照/剖析的轻量工具，全部基于 numpy
（禁止 torch/tensorflow），供各参数 verify.py 的性能敏感路径复用：

  - synth_batch：向量化批量合成（np.stack + float32 压缩）；
  - to_float32：统一转 float32 辅助；
  - cached_synth：lru_cache 包装的合成器（键 = family/variant/noise/seed）；
  - LazyInterp：惰性插值包装（resolve() 时才执行 np.interp，结果缓存）；
  - float32_corr_diff：float64 参考 vs float32 结果的精度对比；
  - profile_run：cProfile 剖析钩子（.prof 转储 + 文本摘要 top20）。

用法示例：
  from _perf import synth_batch, cached_synth, LazyInterp, profile_run
  batch = synth_batch(my_synth, [("Qwen", "base", 0.01, 42), ...])
  s = cached_synth(my_synth); a = s("Qwen", "base", 0.01, 42)
  interp = LazyInterp(x_src, y_src, x_dst); y = interp.resolve()
  prof = profile_run(bench, "logs", "synth")
"""
from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, Protocol

import numpy as np


# ---------------------------------------------------------------- 类型/堆叠
def to_float32(arr: Any) -> np.ndarray:
    """将任意数组/序列转为 float32 ndarray（视图优先，避免无谓拷贝）。"""
    return np.asarray(arr, dtype=np.float32)


def synth_batch(
    synth_fn: Callable[..., np.ndarray],
    params_list: Iterable[Sequence[Any]],
    dtype: Any = np.float32,
) -> np.ndarray:
    """向量化批量合成：对每组参数调用 synth_fn，结果堆叠为 (N, ...) 数组。

    参数：
      synth_fn:    单条合成函数，接受一组参数并返回 ndarray（各条形状须一致）
      params_list: 参数元组列表，如 [("Qwen", "base", 0.01, 42), ...]
      dtype:       输出 dtype（默认 float32，兼顾精度与内存）
    返回：堆叠后的 (N, ...) 数组；params_list 为空时返回空 (0,) 数组。
    异常：SynthesisError（各条输出形状不一致时）。
    """
    from _errors import SynthesisError

    arrays = [np.asarray(synth_fn(*p), dtype=dtype) for p in params_list]
    if not arrays:
        return np.zeros((0,), dtype=dtype)
    try:
        stacked = np.stack(arrays, axis=0)
    except ValueError as e:
        raise SynthesisError(
            f"synth_batch 输出形状不一致（第 {len(arrays)} 条）: {e}",
            actual=[a.shape for a in arrays[:3]],
        ) from e
    return np.asarray(stacked, dtype=np.float32)


# ---------------------------------------------------------------- 缓存合成
class CachedSynth(Protocol):
    """缓存合成器的结构化接口：兼可调用性与 lru_cache 标准属性。

    运行时由 functools._lru_cache_wrapper 满足；协议化仅为类型检查。
    （cache_info 返回类型标注为 Any：mypy 2.3.1 typeshed 未导出
    functools.CacheInfo，运行时仍为 CacheInfo，含 .hits/.currsize。）
    """

    def __call__(self, family: str, variant: str, noise: float, seed: int) -> Any: ...

    def cache_info(self) -> Any: ...

    def cache_clear(self) -> None: ...


def cached_synth(synth_fn: Callable[..., Any]) -> CachedSynth:
    """lru_cache 包装合成函数：键 (family, variant, noise, seed)。

    要求：四个参数均 hashable（str/float/int 均可）；返回结果被 lru_cache
    持有，重复键命中时不再调用 synth_fn。包装结果附带 cache_clear()/
    cache_info()（lru_cache 标准属性）。

    用法：
      cached = cached_synth(my_synth)
      a = cached("Qwen", "base", 0.01, 42)   # 首次计算
      b = cached("Qwen", "base", 0.01, 42)   # 命中缓存，synth_fn 不再执行
    """
    @functools.lru_cache(maxsize=256)
    def wrapper(family: str, variant: str, noise: float, seed: int) -> Any:
        return synth_fn(family, variant, noise, seed)

    wrapper.__name__ = getattr(synth_fn, "__name__", "synth") + "_cached"
    return wrapper


# ---------------------------------------------------------------- 惰性插值
class LazyInterp:
    """惰性插值包装：保存源剖面与目标格点，resolve() 时才执行 np.interp。

    适用场景：构造阶段成本低、求值阶段成本高的插值任务；多次 resolve()
    只计算一次（结果缓存），适用于网格扫描循环中反复读取同一条剖面。

    用法：
      interp = LazyInterp(src_x, src_y, dst_x)   # 不立即计算
      y = interp.resolve()                        # 首次执行 np.interp 并缓存
      y2 = interp()                               # 等价 resolve()，命中缓存
    """

    def __init__(self, src_x: Any, src_y: Any, dst_x: Any) -> None:
        """保存源剖面 (src_x, src_y) 与目标格点 dst_x（全部转 float64）。"""
        self.src_x = np.asarray(src_x, dtype=np.float64)
        self.src_y = np.asarray(src_y, dtype=np.float64)
        self.dst_x = np.asarray(dst_x, dtype=np.float64)
        self._resolved: np.ndarray | None = None
        self._done: bool = False

    def resolve(self) -> np.ndarray:
        """执行（或复用缓存的）np.interp(dst_x, src_x, src_y)。"""
        if not self._done:
            self._resolved = np.interp(self.dst_x, self.src_x, self.src_y)
            self._done = True
        assert self._resolved is not None  # _done=True 时必有结果
        return self._resolved

    def __call__(self) -> np.ndarray:
        """等价 resolve()：支持 `y = interp()` 简写。"""
        return self.resolve()

    @property
    def resolved(self) -> bool:
        """是否已执行插值（缓存是否就绪）。"""
        return self._done


# ---------------------------------------------------------------- 精度对比
def float32_corr_diff(a64: Any, a32: Any) -> dict[str, float | int]:
    """float64 参考与 float32 结果的精度对比。

    参数：
      a64: float64 参考数组（高精度基准）
      a32: float32 压缩/计算结果（待评估）
    返回：dict：
      - max_abs_diff：最大绝对偏差
      - mean_abs_diff：平均绝对偏差
      - corr：Pearson 相关系数（a64 与 a32 拉平后；空数组/单元素为 1.0）
      - n：元素个数
    """
    from _errors import SynthesisError

    ref = np.asarray(a64, dtype=np.float64).ravel()
    val = np.asarray(a32, dtype=np.float64).ravel()
    if ref.size == 0 or val.size == 0:
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "corr": 1.0, "n": 0}
    if ref.size != val.size:
        raise SynthesisError(
            "float32_corr_diff 输入长度不一致",
            expected=ref.size,
            actual=val.size,
        )
    diff = np.abs(ref - val)
    corr = float(np.corrcoef(ref, val)[0, 1]) if ref.size > 1 else 1.0
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "corr": corr,
        "n": int(ref.size),
    }


# ---------------------------------------------------------------- 剖析钩子
def profile_run(
    fn: Callable[[], Any],
    out_dir: str,
    label: str,
) -> dict[str, Any]:
    """cProfile 剖析钩子：运行 fn 并输出剖析文件与文本摘要。

    输出（均在 out_dir/logs/profile/ 下）：
      - <label>_<ts>.prof：pstats 二进制转储（可再用 pstats/可视化工具加载）
      - <label>_<ts>.txt：文本摘要（按 cumtime 排序 top 20）
    参数：
      fn:      待剖析的可调用对象（无参；其返回值原样返回）
      out_dir: 输出根目录（将自动创建 logs/profile/ 子目录）
      label:   输出文件名标签（自动附加时间戳）
    返回：dict（result=fn 返回值, prof=.prof 路径, txt=.txt 路径）。
    """
    import cProfile
    import pstats

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof_dir = os.path.join(out_dir, "logs", "profile")
    os.makedirs(prof_dir, exist_ok=True)
    prof_path = os.path.join(prof_dir, f"{label}_{ts}.prof")
    txt_path = os.path.join(prof_dir, f"{label}_{ts}.txt")

    prof = cProfile.Profile()
    prof.enable()
    try:
        result = fn()
    finally:
        prof.disable()
    prof.dump_stats(prof_path)

    # 文本摘要：按 cumtime 排序，取 top 20（经 stream 直接写文件）
    with open(txt_path, "w", encoding="utf-8") as f:
        stats = pstats.Stats(prof_path, stream=f)
        stats.sort_stats("cumtime")
        stats.print_stats(20)   # top 20（含函数头与调用者部分）

    return {"result": result, "prof": prof_path, "txt": txt_path}


if __name__ == "__main__":
    # 自检：批量合成 / 缓存 / 惰性插值 / 精度对比 / 剖析输出
    def _fake_synth(family: str, variant: str, noise: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        base = np.linspace(0.0, 1.0, 16)
        return base + noise * rng.standard_normal(16) + (0.1 if "Instruct" in variant else 0.0)

    # 1) 批量合成（float32 堆叠）
    params = [("Qwen", "base", 0.01, 1), ("Qwen", "Instruct", 0.02, 2), ("L", "base", 0.01, 3)]
    batch = synth_batch(_fake_synth, params)
    assert batch.shape == (3, 16) and batch.dtype == np.float32
    print("synth_batch:", batch.shape, batch.dtype)

    # 2) 缓存合成（二次命中）
    cached = cached_synth(_fake_synth)
    a1 = cached("Qwen", "base", 0.01, 1)
    a2 = cached("Qwen", "base", 0.01, 1)
    assert a1 is a2 and cached.cache_info().hits == 1
    print("cached_synth:", a1.shape, "hits =", cached.cache_info().hits)

    # 3) 惰性插值
    src_x = np.linspace(0, 1, 10)
    src_y = np.sin(src_x * np.pi)
    dst_x = np.linspace(0, 1, 5)
    interp = LazyInterp(src_x, src_y, dst_x)
    assert not interp.resolved
    y = interp.resolve()
    assert interp.resolved and y is interp.resolve()   # 二次调用命中缓存
    print("LazyInterp:", y)

    # 4) 精度对比
    ref64 = np.linspace(0.0, 1.0, 1000)
    val32 = to_float32(ref64)
    stats = float32_corr_diff(ref64, val32)
    assert stats["corr"] >= 0.999999 and stats["max_abs_diff"] < 1e-6
    print("float32_corr_diff:", stats)

    # 5) 剖析输出（临时目录，不污染插件工作区）
    import shutil
    import tempfile

    def _bench() -> int:
        for _ in range(200):
            _fake_synth("Qwen", "base", 0.01, 7)
        return 200

    tmp = tempfile.mkdtemp(prefix="aiq_perf_selfcheck_")
    try:
        prof = profile_run(_bench, tmp, "selfcheck")
        assert os.path.isfile(prof["prof"]) and os.path.isfile(prof["txt"])
        assert "logs" in prof["prof"].split(os.sep) and "profile" in prof["prof"].split(os.sep)
        print("profile_run ->", prof["prof"])
        print("profile_run ->", prof["txt"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("_perf 自检 PASS")
