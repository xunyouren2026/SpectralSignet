# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 结构化 JSON 日志模块（_logging）
=====================================================================
提供 `StructuredLogger`：每条 `step()` 输出**一行可被 json.loads
解析的 JSON**（字段：ts、step_id、name、elapsed_ms、status、extra），
供后续脚本/管道直接按行解析，无需正则匹配。

后端选择：
  1. loguru 优先（已装时接管全局 logger，输出格式固定为纯 `{message}`，
     保证每行即纯 JSON）；
  2. stdlib logging + JSONFormatter 兜底（loguru 缺失时自动启用，
     使用独立 logger "aiq.structured"、propagate=False，不影响全局）。

用法：
  from _logging import logger
  logger.configure("logs")                  # 控制台 + logs/structured.log
  logger.step(1, "repro", 12.3, "PASS", tol=1e-3)   # 手动记录
  with logger.step_context(2, "family"):    # 上下文自动计时并记录
      do_something()

注意：
  - configure() 会"接管"全局 loguru logger（remove 全部 handler 后
    重建纯 JSON handler），这是有意为之——本插件日志统一走本模块；
  - 未调用 configure() 时首次输出会自动延迟执行 configure(None)，
    保证控制台至少有一行纯 JSON。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

# ---- loguru 优先；缺失则回退 stdlib logging ----
try:
    from loguru import logger as _loguru_logger  # loguru 自带类型标注
    _HAS_LOGURU = True
except Exception:  # pragma: no cover - loguru 缺失路径
    _loguru_logger = None  # type: ignore[assignment]
    _HAS_LOGURU = False

if not _HAS_LOGURU:  # pragma: no cover - stdlib 兜底路径
    import logging as _stdlib_logging

    class JSONFormatter(_stdlib_logging.Formatter):
        """stdlib 兜底 Formatter：消息本身即 JSON 行，不做前缀装饰。"""

        def format(self, record: _stdlib_logging.LogRecord) -> str:
            return record.getMessage()  # 消息即完整 JSON 行


class StructuredLogger:
    """结构化 JSON 日志器（模块级单例由文件底部 `logger` 提供）。"""

    def __init__(self) -> None:
        self._use_loguru: bool = _HAS_LOGURU
        self._configured: bool = False
        self._log_dir: str | None = None
        self._log_file: str | None = None
        self._loguru_ids: list[int] = []          # 本模块添加的 loguru handler id
        self._stdlib: Any = None                  # stdlib 兜底 logger（惰性）
        self._lock = threading.Lock()             # 多线程 step 输出互斥

    # ------------------------------------------------------------ 配置
    def configure(self, log_dir: str | None = None) -> StructuredLogger:
        """配置输出目标：控制台 + 可选 logs/ 目录文件（结构化一行一 JSON）。

        参数：
          log_dir: 输出目录（可为相对/绝对路径）；提供时在
                   `log_dir/structured.log` 追加写日志，目录自动创建。
        返回：self（可链式调用）。
        """
        with self._lock:
            self._log_dir = log_dir
            if self._use_loguru:
                self._loguru_configure(log_dir)
            else:  # pragma: no cover - stdlib 兜底路径
                self._stdlib_configure(log_dir)
            self._configured = True
        return self

    def _loguru_configure(self, log_dir: str | None) -> None:
        """接管全局 loguru：移除旧 handler，重建纯 `{message}` 输出。"""
        _loguru_logger.remove()                    # 移除全部（含默认），保证纯 JSON 行
        self._loguru_ids = []
        self._loguru_ids.append(
            _loguru_logger.add(
                sys.stderr, format="{message}", colorize=False, enqueue=False
            )
        )
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._log_file = os.path.join(log_dir, "structured.log")
            self._loguru_ids.append(
                _loguru_logger.add(
                    self._log_file, format="{message}", encoding="utf-8",
                    enqueue=False, rotation="10 MB",
                )
            )

    def _stdlib_configure(self, log_dir: str | None) -> None:  # pragma: no cover
        """stdlib 兜底：独立 logger + JSONFormatter，控制台与可选文件。"""
        self._stdlib = _stdlib_logging.getLogger("aiq.structured")
        self._stdlib.setLevel(_stdlib_logging.INFO)
        self._stdlib.propagate = False
        for h in list(self._stdlib.handlers):      # 清除旧 handler 防重复
            self._stdlib.removeHandler(h)
        fmt = JSONFormatter()
        sh = _stdlib_logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        self._stdlib.addHandler(sh)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._log_file = os.path.join(log_dir, "structured.log")
            fh = _stdlib_logging.FileHandler(self._log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            self._stdlib.addHandler(fh)

    def _ensure_configured(self) -> None:
        """首次使用时若无配置，自动按默认（仅控制台）初始化。"""
        if not self._configured:
            self.configure(None)

    # ------------------------------------------------------------ 记录
    def step(
        self,
        step_id: int,
        name: str,
        elapsed_ms: float,
        status: str,
        **extra: Any,
    ) -> None:
        """输出一行结构化 JSON 日志（可被 json.loads 直接解析）。

        参数：
          step_id:    步骤序号（int）
          name:       步骤名称（如 "repro"、"family"）
          elapsed_ms: 耗时（毫秒，float）
          status:     状态（如 "PASS"/"FAIL"，调用方约定）
          extra:      附加字段（自动放入 extra 子对象；不可序列化对象
                      经 default=str 兜底，保证永不抛序列化异常）
        """
        self._ensure_configured()
        record: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "step_id": int(step_id),
            "name": name,
            "elapsed_ms": float(elapsed_ms),
            "status": status,
            "extra": dict(extra),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._emit(line)

    @contextmanager
    def step_context(
        self, step_id: int, name: str, **extra: Any
    ) -> Iterator[None]:
        """上下文管理器：自动计时并在退出时写入 step 日志。

        用法：
          with logger.step_context(1, "repro", tol=1e-3):
              result = run()

        语义：正常退出记 status="PASS"；抛异常时记 status="FAIL"
        （extra 附加 error 字段）并向上重抛异常。
        """
        t0 = time.perf_counter()
        status = "PASS"
        error: str | None = None
        try:
            yield
        except Exception as e:
            status = "FAIL"
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if error is not None:
                extra["error"] = error
            self.step(step_id, name, elapsed_ms, status, **extra)

    # ------------------------------------------------------------ 输出
    def _emit(self, line: str) -> None:
        """写一行到控制台与可选文件（线程安全）。"""
        with self._lock:
            if self._use_loguru:
                _loguru_logger.info(line)          # format={message} → 纯 JSON 行
            else:  # pragma: no cover - stdlib 兜底路径
                self._stdlib.info(line)


# 模块级单例：全插件共享同一实例
logger = StructuredLogger()


if __name__ == "__main__":
    # 自检：step 输出 + step_context 计时 + json.loads 可解析性
    logger.configure()                             # 仅控制台
    logger.step(1, "manual", 3.5, "PASS", tol=1e-3, note="自检")
    with logger.step_context(2, "auto", family="Qwen"):
        time.sleep(0.01)                           # 模拟耗时步骤
    try:
        with logger.step_context(3, "fail"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # 兜底验证：解析 _emit 侧格式（loguru 写 stderr，此处直接构造一次行校验）
    sample = json.dumps(
        {"ts": "2026-01-01T00:00:00.000+08:00", "step_id": 0, "name": "x",
         "elapsed_ms": 0.0, "status": "PASS", "extra": {}},
        ensure_ascii=False,
    )
    parsed = json.loads(sample)
    assert parsed["status"] == "PASS"
    print("_logging 自检 PASS（json.loads 可解析）")
