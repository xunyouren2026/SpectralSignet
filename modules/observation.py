"""
aiq-geometric-forensics.modules.observation — 可观测 / 结构化日志层
=====================================================================
补足大厂生产 checklist 中的「日志/可观测」维度。本层是**纯旁路观测**：
不改变任何计算返回值，只记录副作用事件，因此对 103 项回归断言零影响。

约定：
  * 模块副作用（测量开始/完成、回退兜底、缺依赖守卫、命令成败）必须
    通过 logger 记录，而不是静默 print。
  * 人类可读终端日志（含时间戳/级别/模块）用于交互；JSON 落盘用于
    CI / 离线可观测（机器可读、可按 logger/level 检索）。

用法：
  import modules.observation as obs
  log = obs.get_logger(__name__)              # → logger "aiq.xxx"
  log.info("测量完成 model=%s tok_s=%.2f", name, v)
  obs.setup_logging(level=logging.INFO, file="run.jsonl", json=True)  # CLI 入口调一次
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

_ROOT = "aiq"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """返回本包命名空间下的模块级 logger（如 'aiq.data'）。"""
    if name.startswith(_ROOT):
        return logging.getLogger(name)
    short = name.rsplit(".", 1)[-1] if "." in name else name
    return logging.getLogger(f"{_ROOT}.{short}")


def _human_handler(stream: Any = None) -> logging.Handler:
    h = logging.StreamHandler(stream)
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S"))
    return h


def _json_handler(file_path: str) -> logging.Handler:
    class _JsonHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                            time.localtime(record.created)),
                        "level": record.levelname,
                        "logger": record.name,
                        "pid": record.process,
                        "msg": record.getMessage(),
                    }, ensure_ascii=False) + "\n")
            except Exception:  # noqa: BLE001  记录失败不中断主流程
                self.handleError(record)
    return _JsonHandler()


def setup_logging(level: int = logging.INFO,
                  file: str | None = None,
                  json: bool = False,
                  stream: Any = None,
                  force: bool = True) -> logging.Logger:
    """进程级日志配置，幂等。

    参数：
      level: 根 logger 级别（默认 INFO）
      file: 若提供，追加写入该路径；json=True 写 JSONL，否则写人类可读文本
      json: 是否输出机器可读 JSONL
      stream: 终端流（默认 sys.stderr）
      force：False 时已配置则跳过（避免二次注册重复 handler）
    """
    global _configured
    if _configured and not force:
        return logging.getLogger(_ROOT)
    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    root.handlers.clear()                       # 幂等：替换旧 handler，防重复输出

    if file:
        root.addHandler(_json_handler(file) if json
                        else _file_handler(file))
    if stream is not None or not file:
        root.addHandler(_human_handler(stream))
    root.propagate = False                      # 不冒泡到 root，避免双写
    _configured = True
    return root


def _file_handler(path: str) -> logging.Handler:
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S"))
    return h


def trace_id() -> str:
    """生成一次 CLI 运行的可观测追踪 id（毫秒级时间戳）。"""
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


__all__ = ["get_logger", "setup_logging", "trace_id"]
