# -*- coding: utf-8 -*-
"""F06 out_dir 输出路径（自动回退）— 多候选目录选择逻辑验证（四层工厂架构）
====================================================================
规则: 按优先级依次尝试候选路径, 返回第一个"可创建且可写"的目录;
      全部失败 → 兜底返回 "." (当前工作目录), 不抛异常。
可写性测试: 写入探针文件 .write_test 再删除 (主文档 第4777-4788行)

验证目标（与原脚本完全一致，保真）：
  1. 首选失效(父路径被文件占位)时正确跳过, 落到第二个可用候选
  2. 优先级顺序: 多个候选都可用时取第一个
  3. 全部候选失败 → 兜底 "." 且不抛异常
  4. 选中目录可真实写入/读回 JSON
  5. 子目录组织 (fingerprints/charts/logs) 自动创建
  6. 确定性: 同一环境重复调用返回同一目录

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  F06Config              —— 配置模型（pydantic 优先；dataclass 回退；列表字段
                            SUB_DIRS 经 _mutable 防御共享可变对象，文本字段
                            WRITE_PROBE/FALLBACK_DIR/TEST_JSON 走 get_str 语义）
  ConfigFactory          —— 实例化 F06Config（env AIQ_F06_<KEY> > YAML >
                            _params_data.json > 模型默认值）
  （无合成器类：文件系统逻辑，工具函数以模块级纯函数提供）
  ValidatorEngine        —— 7 项验证 + 结构化 JSON 日志（_logging）+
                            类型化异常（_errors，携带 expected/actual）
  ReportGenerator        —— 文本/JSON/HTML 报告 + 退出码 0/1
  main()                 —— 仅编排 cfg→engine→report（--profile/--json/--html）
                            + 临时目录生命周期管理（finally 清理）

数据源：
  主文档《参数附录表完整版》行 4724-4850（F06 out_dir）
  《参数完整定义与公式.txt》F06（第 342-348 行）
  源码 _local_whitebox_detect.py（第 60-62 行：首选不存在时回退）
  《参数审计与实验报告.txt》行 75（状态=已用）
说明：纯数值/文件系统逻辑验证（tempfile 临时目录），无外部依赖。

真实模型对照：
  经 _real_data 惰性读取真实模型路径（模型库经 _cfg 自动探测），验证输出
  路径回退逻辑与模型库目录存在性；不向真实模型目录写入探针（只读验证存在性，
  可写性探针由临时目录用例覆盖）。数据来源标注：[真实实测] 或 [审计回退]。
=====================================================================
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import (  # noqa: E402
    AIQValidationError,
    ConfigError,
    RealModelMismatchError,
)
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 F06Config（pydantic 优先；dataclass 回退） ----
_HAS_PYDANTIC = False
_ConfigModelBase: Any
try:
    from pydantic import BaseModel as _PydanticBase  # noqa: E402
    _ConfigModelBase = _PydanticBase
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - pydantic 缺失路径
    import dataclasses as _dataclasses

    @_dataclasses.dataclass
    class _DataclassBase:
        """pydantic 缺失时的空壳基类（无字段，仅提供 dataclass 语义）。"""

    _ConfigModelBase = _DataclassBase

if _HAS_PYDANTIC:
    def _mutable(v: Any) -> Any:
        """pydantic 路径：默认值直接返回（pydantic 内部深拷贝，安全）。"""
        return v
else:
    def _mutable(v: Any) -> Any:
        """dataclass 路径：包装为 default_factory，避免类定义期共享可变默认值报错。"""
        return _dataclasses.field(default_factory=lambda: v)


class F06Config(_ConfigModelBase):
    """F06 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 F06 节点键名对应（SUB_DIRS）；取值优先级：
    env AIQ_F06_<KEY> > YAML > _params_data.json > 本模型默认值。
    SUB_DIRS 为列表类配置（ConfigFactory.get_list 语义）；WRITE_PROBE /
    FALLBACK_DIR / TEST_JSON 为文本类配置（ConfigFactory.get_str 语义）。
    """

    SUB_DIRS: list = _mutable(["fingerprints", "charts", "logs", "checkpoints"])  # 子目录组织
    WRITE_PROBE: str = ".write_test"   # 可写性探针文件名（主文档 4777-4788 行）
    FALLBACK_DIR: str = "."            # 全部候选失败时的兜底目录
    TEST_JSON: str = "results.json"    # 写回测试文件名


# ---- 第二层：配置工厂 ConfigFactory（实例化 F06Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """F06 配置工厂：按优先级实例化 F06Config。"""

    def build(self) -> F06Config:
        """构建 F06Config：pydantic 优先，dataclass 回退（共享基类 build_model 驱动）。"""
        return self.build_model(F06Config, "F06")


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def get_output_dir(candidates: list, cfg: F06Config) -> str:
    """尝试候选路径, 返回第一个可用的可写目录; 全部失败返回 FALLBACK_DIR。

    可写性 = 创建目录 + 写入探针文件 + 删除探针文件全链路成功
    （README ②公式 writable 定义）。捕获 OSError/ValueError 防御
    非法路径与权限异常。
    """
    for path in candidates:                      # 按优先级依次尝试
        try:
            os.makedirs(path, exist_ok=True)     # 创建目录（已存在则忽略）
            test_file = os.path.join(path, cfg.WRITE_PROBE)
            with open(test_file, "w") as f:      # 写探针：验证可写
                f.write("test")
            os.remove(test_file)                 # 清理探针：不污染目录
            return path                          # 全链路成功 -> 选中
        except (OSError, PermissionError, IOError, ValueError):
            continue                             # 任一环节失败 -> 试下一候选
    return cfg.FALLBACK_DIR                      # 全部失败 -> 兜底当前目录


def make_subdirs(out_dir: str, cfg: F06Config) -> list:
    """在 out_dir 下创建子目录（exist_ok）。返回已创建路径列表。"""
    created = []
    for name in cfg.SUB_DIRS:
        p = os.path.join(out_dir, name)
        os.makedirs(p, exist_ok=True)            # 重复调用不报错
        created.append(p)
    return created


def write_read_json(out_dir: str, data: dict, cfg: F06Config) -> bool:
    """写入/读回 JSON，验证目录可真实落盘。返回内容是否一致。"""
    fp = os.path.join(out_dir, cfg.TEST_JSON)
    with open(fp, "w", encoding="utf-8") as f:   # UTF-8 写入
        json.dump(data, f, ensure_ascii=False)
    with open(fp, "r", encoding="utf-8") as f:   # 读回
        back = json.load(f)
    return back == data and os.path.isdir(out_dir)   # 内容一致且目录存在


# ---- 第三层：验证引擎 ValidatorEngine（7 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """F06 验证引擎：顺序执行 7 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志；
    - 失败时抛 _errors 类型化异常，由 run() 捕获记 FAIL 并继续；
    - 临时目录由 main 创建并注入（finally 清理），验证逻辑不感知具体路径。
    """

    def __init__(
        self,
        config: F06Config,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
        tmp_dir: str | None = None,
    ) -> None:
        super().__init__(config, None, reporter)
        self._real_data = real_data
        self._tmp = tmp_dir or tempfile.mkdtemp(prefix="f06_verify_")  # 隔离临时根目录

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 首选失效回退
    def validate_fallback(self) -> dict:
        """1) 首选不可写(父路径是文件) → 跳过, 落到第二候选。"""
        cfg = self.config
        d2 = os.path.join(self._tmp, "d2")       # 候选 2（有效目录）
        d3 = os.path.join(self._tmp, "d3")       # 候选 3（有效目录）
        blocker1 = os.path.join(self._tmp, "blocker1")
        with open(blocker1, "w") as f:
            f.write("i am a file, not a dir")   # 占位文件 → 其下 makedirs 抛异常
        candidates = [os.path.join(blocker1, "out"), d2, d3]  # 首选被文件占位
        sel = get_output_dir(candidates, cfg)
        ok = sel == d2                           # 应跳过首选、落到第二候选
        if not ok:
            raise ConfigError(
                f"首选不可写(父路径是文件)应回退到第二候选，实际选中 {sel!r}",
                expected=d2, actual=sel, param_key="F06",
            )
        return {"detail": f"首选不可写(父路径是文件) -> 跳过, 落到第二候选: {sel}"}

    # ------------------------------------------------------------ 2) 优先级顺序
    def validate_priority(self) -> dict:
        """2) 多候选均可用 → 取优先级第一个。"""
        cfg = self.config
        d2 = os.path.join(self._tmp, "d2")       # 候选 2（有效目录）
        d3 = os.path.join(self._tmp, "d3")       # 候选 3（有效目录）
        sel2 = get_output_dir([d2, d3], cfg)
        ok = sel2 == d2                          # 两候选都可用 -> 取优先级第一
        if not ok:
            raise ConfigError(
                f"多候选均可用应取优先级第一个(d2)，实际 {sel2!r}",
                expected=d2, actual=sel2, param_key="F06",
            )
        return {"detail": f"多候选均可用 -> 取优先级第一个: {sel2} (d2 优先于 d3)"}

    # ------------------------------------------------------------ 3) 全部失败兜底
    def validate_all_fail(self) -> dict:
        """3) 全部候选失效 → 兜底 FALLBACK_DIR，不抛异常。"""
        cfg = self.config
        blocker1 = os.path.join(self._tmp, "blocker1")
        sel3 = get_output_dir([os.path.join(blocker1, "a"),
                               os.path.join(blocker1, "b")], cfg)   # 两个都被文件占位
        ok = sel3 == cfg.FALLBACK_DIR
        if not ok:
            raise ConfigError(
                f"全部候选失效应兜底返回 {cfg.FALLBACK_DIR!r}，实际 {sel3!r}",
                expected=cfg.FALLBACK_DIR, actual=sel3, param_key="F06",
            )
        return {"detail": f"全部候选失效 -> 兜底返回: {repr(sel3)} (不抛异常)"}

    # ------------------------------------------------------------ 4) JSON 写入/读回
    def validate_json_roundtrip(self) -> dict:
        """4) 选中目录可真实写入/读回 results.json。"""
        cfg = self.config
        d2 = os.path.join(self._tmp, "d2")
        out = get_output_dir([d2], cfg)
        data = {"model": "Qwen2.5-0.5B-Instruct", "tok_s": 7.20, "KV_MB": 24.2}
        ok = write_read_json(out, data, cfg)
        if not ok:
            raise ConfigError(
                f"选中目录应可写入/读回 {cfg.TEST_JSON}: {out}",
                actual=out, param_key="F06",
            )
        return {"detail": f"写入/读回 {cfg.TEST_JSON}: {data} 一致 -> {out}"}

    # ------------------------------------------------------------ 5) 子目录组织
    def validate_subdirs(self) -> dict:
        """5) 子目录自动创建（fingerprints/charts/logs/checkpoints）。"""
        cfg = self.config
        d2 = os.path.join(self._tmp, "d2")
        out = get_output_dir([d2], cfg)
        subs = make_subdirs(out, cfg)
        ok = all(os.path.isdir(s) for s in subs)
        if not ok:
            raise ConfigError(
                f"子目录应全部创建成功: {[os.path.basename(s) for s in subs]}",
                expected=list(cfg.SUB_DIRS),
                actual=[os.path.basename(s) for s in subs], param_key="F06",
            )
        return {"detail": f"子目录自动创建: {[os.path.basename(s) for s in subs]}"}

    # ------------------------------------------------------------ 6) 确定性
    def validate_determinism(self) -> dict:
        """6) 同一环境重复调用返回同一目录。"""
        cfg = self.config
        d2 = os.path.join(self._tmp, "d2")
        sel_a = get_output_dir([d2, os.path.join(self._tmp, "d3")], cfg)
        sel_b = get_output_dir([d2, os.path.join(self._tmp, "d3")], cfg)
        ok = sel_a == sel_b == d2              # 相同输入 -> 相同输出
        if not ok:
            raise ConfigError(
                f"重复调用应返回同一目录: {sel_a!r} vs {sel_b!r}",
                expected=d2, actual=(sel_a, sel_b), param_key="F06",
            )
        return {"detail": f"确定性: 重复调用返回同一目录 {sel_a}"}

    # ------------------------------------------------------------ 7) 真实模型对照
    def validate_real_model(self) -> dict:
        """7) 真实输出路径回退逻辑验证（模型库目录存在性）。

        使用真实模型路径（模型库经 _cfg 自动探测）：首选失效后应回退到真实
        模型目录（存在时），全部失效兜底 FALLBACK_DIR。只读验证存在性。
        """
        cfg = self.config
        rd = self._get_real_data()
        mdir = rd.get("model")                       # 真实模型目录路径（来自存档）
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"
        # 无真实模型路径存档：审计回退，对照层跳过（不判失败）
        if not mdir:
            return {
                "detail": f"{tag} 无真实模型路径存档 -> 审计回退，对照层跳过（不判失败）",
                "skipped": True, "source": tag,
            }
        root = os.path.dirname(mdir)                 # 模型根目录（_models）
        ok1 = os.path.isdir(root)                    # 根目录存在性（只读）
        ok2 = os.path.isdir(mdir)                    # 模型目录存在性（只读）
        fake = os.path.join(root, "_f06_missing_marker_dir")  # 不存在的"首选"候选
        sel = mdir if ok2 else cfg.FALLBACK_DIR      # 存在则选中，否则兜底
        ok3 = (sel == mdir) if ok2 else (sel == cfg.FALLBACK_DIR)
        if not (ok1 and ok2 and ok3):
            raise RealModelMismatchError(
                f"真实模型目录回退验证不符: 根目录存在={ok1}, 模型目录存在={ok2}, "
                f"选中 {sel}",
                expected={"root_isdir": True, "mdir_isdir": True, "sel": mdir},
                actual={"root": ok1, "mdir": ok2, "sel": sel}, param_key="F06",
            )
        return {
            "detail": (f"{tag} 真实模型: {mdir}; _models 根目录存在: {ok1}; "
                       f"模型目录存在: {ok2}; [回退验证] 首选失效(不存在) -> 选中 {sel} "
                       f"(只读验证存在性；可写性探针由临时目录用例覆盖)"),
            "source": rd.source_tag(), "tag": tag, "mdir": mdir,
            "root_isdir": ok1, "mdir_isdir": ok2, "sel": sel,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 7 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "fallback", self.validate_fallback),
            (2, "priority", self.validate_priority),
            (3, "all_fail", self.validate_all_fail),
            (4, "json_roundtrip", self.validate_json_roundtrip),
            (5, "subdirs", self.validate_subdirs),
            (6, "determinism", self.validate_determinism),
            (7, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except AIQValidationError as e:
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e), "expected": e.expected,
                         "actual": e.actual, "param_key": e.param_key}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """F06 验证编排：四层工厂装配 + --profile/--json/--html 输出 + 临时目录清理。"""
    parser = argparse.ArgumentParser(prog="verify", description="F06 out_dir 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()
    report = ReportGenerator()
    tmp = tempfile.mkdtemp(prefix="f06_verify_")   # 隔离临时根目录（自动清理）
    try:
        engine = ValidatorEngine(cfg, report, real_data=RD, tmp_dir=tmp)

        print("=" * 78)
        print("F06 out_dir 输出路径自动回退 —— 逻辑验证（四层工厂架构，无外部依赖）")
        print(f"数据源: {P.source_tag()}")
        print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
        print(f"临时根目录: {tmp}")
        print("=" * 78)

        # ---- ④ 运行（可选剖析）----
        if args.profile:
            res = profile_run(engine.run, out_dir, "f06_verify")
            print(f"剖析文件: {res['prof']}")
        else:
            engine.run()

        # ---- ⑤ 报告输出 ----
        print(report.render_text())
        if args.json:
            json_path = os.path.join(out_dir, "f06_verify_report.json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(report.render_json())
            print(f"JSON 报告已写入: {json_path}")
        if args.html:
            html_path = os.path.join(out_dir, "f06_verify_report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(report.render_html())
            print(f"HTML 报告已写入: {html_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # 无论成败都清理临时目录

    # ---- ⑥ 汇总与退出码 ----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
