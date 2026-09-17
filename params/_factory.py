# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 工厂基类层（_factory）
=====================================================================
为所有参数验证流程提供共享的工厂/引擎/报告基础设施：

  - ConfigFactory：统一参数取值（环境变量 > YAML > _params_data.json
    > 兜底 default），并提供类型化访问、pydantic/dataclass 模型构造；
  - ParamConfig：基于 ConfigFactory 的泛型属性访问层（__getattr__）；
  - FingerprintSynthesizer：指纹合成器基类（占位，含种子管理）；
  - ValidatorEngine：验证引擎基类（step 遍历 + 结构化日志 + 判定）；
  - ReportGenerator：报告生成器基类（文本/JSON/HTML + 退出码）。

参数取值优先级（load_param）：
  1. 环境变量 `AIQ_<PARAM>_<KEY>`（自动大写，如 AIQ_T01_PLATFORM）；
  2. YAML 配置（本目录 config.yaml，若存在且 PyYAML 可用）；
  3. _params_data.json（惰性 import 既有 _params 模块，复用其
     "真实测量 > 集中配置" 加载逻辑）；
  4. 调用方传入的 default 兜底。

环境：Python 3.10+；pydantic 可选（缺失时 build_model 自动回退
dataclass 运行时校验）；禁止 torch/tensorflow。
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from _logging import logger

# ---- 自举：确保参数根目录在 sys.path（使 _params 可惰性导入）----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _errors import ConfigError  # noqa: E402  (顶部自举后导入)


# ---------------------------------------------------------------- 类型工具
def _looks_undefined(value: Any) -> bool:
    """判断是否为 pydantic 的 Required/Undefined 哨兵值。"""
    if value is ...:
        return True
    if value is None:
        return False
    return "PydanticUndefined" in type(value).__name__


def _type_name(annotation: Any) -> str | None:
    """从类型注解推断取值类型名（"int"/"float"/"str"/"list"/"grid"）。

    支持类型对象与字符串注解（dataclass 配 from __future__ import annotations）。
    """
    if annotation is None:
        return None
    # 字符串注解（dataclass + future annotations）
    if isinstance(annotation, str):
        s = annotation.strip().lower().replace("typing.", "").replace("builtins.", "")
        if s in ("int", "integer"):
            return "int"
        if s in ("float", "double", "real"):
            return "float"
        if s in ("bool", "boolean"):
            return "int"
        if s in ("str", "string"):
            return "str"
        if s.startswith("list") or s.startswith("sequence"):
            return "list"
        if s.startswith("tuple") or s.startswith("grid"):
            return "grid"
        return None
    # 类型对象
    if annotation is int or annotation is bool:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    if annotation is list:
        return "list"
    if annotation is tuple:
        return "grid"
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "list"
    if origin is tuple:
        return "grid"
    return None


# ---------------------------------------------------------------- ConfigFactory
class ConfigFactory:
    """统一参数工厂：按优先级解析参数值，并提供类型化访问与模型构造。

    用法：
      cf = ConfigFactory()
      cf.get_float("T01", "PLATFORM", 1.56)     # 类型化读取
      cf.load_param("T01", "PLATFORM", 1.56, "float")
      cf.build_model(MyPydanticModel, "T01")    # pydantic 优先，dataclass 回退
    """

    def __init__(self, config_path: str | None = None) -> None:
        """初始化。

        参数：
          config_path: YAML 配置文件路径；默认取本目录 config.yaml（不存在则跳过）。
        """
        self._here = _HERE
        self._config_yaml = config_path or os.path.join(_HERE, "config.yaml")
        self._yaml_cache: dict | None = None
        self._yaml_loaded: bool = False
        self._params_mod: Any = None      # 惰性 import 的 _params 模块
        self._pydantic: Any = None        # None=未探测, False=不可用, 模块=可用

    # -------------------------------------------------------- 核心取值
    def load_param(
        self,
        param_id: str,
        key: str,
        default: Any = None,
        value_type: str | None = None,
    ) -> Any:
        """按优先级解析参数值：环境变量 > YAML > _params_data.json > default。

        参数：
          param_id:   参数标识（如 "T01"、"D09"）
          key:        参数键名（如 "PLATFORM"，自动大写用于环境变量）
          default:    兜底默认值
          value_type: 期望类型（"int"/"float"/"str"/"list"/"grid"），
                      用于环境变量字符串的强制转换；None 表示不转换
        """
        # 1) 环境变量 AIQ_<PARAM>_<KEY>（大写）
        env_name = f"AIQ_{param_id}_{key}".upper()
        env_val = os.environ.get(env_name)
        if env_val is not None and env_val.strip() != "":
            return self._coerce(env_val, value_type, default)
        # 2) YAML 配置
        yaml_val = self._yaml_get(param_id, key)
        if yaml_val is not None:
            return yaml_val
        # 3) _params_data.json（复用 _params 的加载逻辑）
        json_val = self._params_get(param_id, key)
        if json_val is not None:
            return json_val
        # 4) 兜底默认值
        return default

    def get_int(self, param_id: str, key: str, default: int = 0) -> int:
        """整型参数读取（带类型强制与容错）。"""
        v = self.load_param(param_id, key, default, "int")
        try:
            return int(v)
        except (TypeError, ValueError):
            return int(default)

    def get_float(self, param_id: str, key: str, default: float = 0.0) -> float:
        """浮点参数读取（带类型强制与容错）。"""
        v = self.load_param(param_id, key, default, "float")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def get_str(self, param_id: str, key: str, default: str = "") -> str:
        """字符串参数读取。"""
        v = self.load_param(param_id, key, default, "str")
        try:
            return str(v)
        except Exception:
            return str(default)

    def get_list(
        self, param_id: str, key: str, default: list | None = None
    ) -> list:
        """列表参数读取（支持 "1,2,3" / "[1,2,3]" / 原生 list）。"""
        v = self.load_param(param_id, key, default, "list")
        if isinstance(v, list):
            return v
        try:
            return list(v)
        except TypeError:
            return list(default) if default is not None else []

    def get_grid(
        self, param_id: str, key: str, default: tuple = (0.0, 1.0, 100)
    ) -> tuple:
        """网格参数读取：(start, stop, count) 三元组（前两项 float，第三项 int）。"""
        v = self.load_param(param_id, key, default, "grid")
        try:
            return tuple(float(x) if i < 2 else int(x) for i, x in enumerate(v))
        except (TypeError, ValueError):
            return tuple(default)

    # -------------------------------------------------------- 模型构造
    def build_model(self, cls: type, param_id: str, **defaults: Any) -> Any:
        """用配置构造模型实例：pydantic 优先，dataclass 运行时校验回退。

        参数：
          cls:      pydantic BaseModel 子类或 dataclass
          param_id: 参数标识（字段名大写后作为 key 参与 load_param）
          defaults: 字段级默认值覆盖（优先级最高的兜底）
        返回：cls 实例。
        异常：ConfigError（类型不支持）。
        """
        # 1) pydantic 优先
        if self._pydantic_available():
            if hasattr(cls, "model_fields") or hasattr(cls, "__fields__"):
                return self._build_pydantic(cls, param_id, defaults)
        # 2) dataclass 回退
        if dataclasses.is_dataclass(cls):
            return self._build_dataclass(cls, param_id, defaults)
        raise ConfigError(
            f"build_model 不支持模型类型 {cls!r}（需 pydantic BaseModel 或 dataclass）",
            actual=cls,
        )

    def _build_pydantic(self, cls: type, param_id: str, defaults: dict) -> Any:
        """pydantic 构造：按字段名大写取参数，缺失用模型默认值。"""
        fields = getattr(cls, "model_fields", None) or getattr(cls, "__fields__", {})
        values: dict[str, Any] = {}
        for name, field in fields.items():
            default_val = defaults.get(name)
            if default_val is None:
                dv = getattr(field, "default", None)
                df = getattr(field, "default_factory", None)
                if df is not None:
                    try:
                        default_val = df()
                    except Exception:
                        default_val = None
                elif not _looks_undefined(dv):
                    default_val = dv
            vt = _type_name(getattr(field, "annotation", None))
            values[name] = self.load_param(param_id, name.upper(), default_val, vt)
        return cls(**values)

    def _build_dataclass(self, cls: type, param_id: str, defaults: dict) -> Any:
        """dataclass 回退构造：pydantic 缺失时保持运行时校验。"""
        values: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            default_val = defaults.get(f.name)
            if default_val is None:
                if f.default is not dataclasses.MISSING:
                    default_val = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    default_val = f.default_factory()
            vt = _type_name(f.type)
            values[f.name] = self.load_param(param_id, f.name.upper(), default_val, vt)
        return cls(**values)

    # -------------------------------------------------------- 数据源
    def _coerce(self, value: Any, value_type: str | None, default: Any) -> Any:
        """按 value_type 转换（主要用于环境变量字符串）。转换失败原样返回。"""
        if value is None:
            return default
        try:
            if value_type in ("int", int):
                return int(value)
            if value_type in ("float", float):
                return float(value)
            if value_type in ("str", str):
                return str(value)
            if value_type in ("list", list):
                if isinstance(value, str):
                    s = value.strip()
                    if s.startswith("[") and s.endswith("]"):
                        return json.loads(s)
                    return [p.strip() for p in s.split(",")]
                return list(value)
            if value_type in ("grid", tuple):
                v = json.loads(value) if isinstance(value, str) else list(value)
                return tuple(float(x) if i < 2 else int(x) for i, x in enumerate(v))
        except (TypeError, ValueError):
            pass
        return value

    def _yaml_get(self, param_id: str, key: str) -> Any:
        """YAML 配置取值：config.yaml 中 data[param_id][key]。"""
        data = self._yaml_load()
        if data is None:
            return None
        node = data.get(param_id)
        if isinstance(node, dict) and key in node:
            return node[key]
        return None

    def _yaml_load(self) -> dict | None:
        """惰性加载 config.yaml；缺失/不可解析/无 PyYAML 时返回 None。"""
        if not self._yaml_loaded:
            self._yaml_loaded = True
            self._yaml_cache = None
            if os.path.isfile(self._config_yaml):
                try:
                    import yaml  # 可选依赖：未装则跳过 YAML 源
                    with open(self._config_yaml, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    self._yaml_cache = data if isinstance(data, dict) else None
                except Exception:
                    self._yaml_cache = None
        return self._yaml_cache

    def _params_get(self, param_id: str, key: str) -> Any:
        """复用既有 _params 加载逻辑（真实测量 > 集中配置）；失败返回 None。"""
        try:
            if self._params_mod is None:
                import _params
                self._params_mod = _params
            return self._params_mod.get(param_id, key, None)
        except Exception:
            return None

    def _pydantic_available(self) -> bool:
        """探测 pydantic 是否可用（惰性，带缓存）。"""
        if self._pydantic is None:
            try:
                import pydantic
                self._pydantic = pydantic
            except Exception:
                self._pydantic = False
        return bool(self._pydantic)


# ---------------------------------------------------------------- ParamConfig
class ParamConfig:
    """泛型参数访问层：以属性/方法两种方式从 ConfigFactory 解析字段。

    用法：
      pc = ParamConfig(ConfigFactory(), "T01")
      pc.PLATFORM            # 等价 cf.load_param("T01", "PLATFORM", None)
      pc.get("PLATFORM", 1.56, "float")
    """

    def __init__(self, factory: ConfigFactory, param_id: str) -> None:
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_param_id", param_id)

    def get(self, key: str, default: Any = None, value_type: str | None = None) -> Any:
        """显式读取：key 原样传给 load_param（不强制大写）。"""
        return self._factory.load_param(self._param_id, key, default, value_type)

    def __getattr__(self, name: str) -> Any:
        """属性型访问：`pc.NAME` → load_param(param_id, NAME.upper(), None)。

        私有名（以下划线开头）不进入取值流程，直接抛 AttributeError。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        factory = object.__getattribute__(self, "_factory")
        param_id = object.__getattribute__(self, "_param_id")
        return factory.load_param(param_id, name.upper(), None)


# ---------------------------------------------------------------- FingerprintSynthesizer
class FingerprintSynthesizer:
    """指纹合成器基类（占位）。

    子类约定：
      - 在 __init__ 中把自身配置存入 self._cfg（可为 dict / 模型对象）；
      - 实现 synthesize(*args, **kwargs) 返回合成指纹（np.ndarray 等）；
      - 需要随机性时先调用 seed_rng() 保证可复现。
    """

    def __init__(self, cfg: Any = None, seed: int = 0) -> None:
        self._cfg = cfg
        self._seed = int(seed)

    def seed_rng(self, seed: int | None = None) -> int:
        """重置 numpy 随机种子；seed 缺省时用构造时的 self._seed。

        返回：实际使用的种子值。
        """
        if seed is None:
            seed = self._seed
        np.random.seed(int(seed))
        return int(seed)

    def synthesize(self, *args: Any, **kwargs: Any) -> Any:
        """合成指纹（子类实现）。"""
        raise NotImplementedError("FingerprintSynthesizer.synthesize 需由子类实现")


# ---------------------------------------------------------------- ValidatorEngine
class ValidatorEngine(ABC):
    """验证引擎基类：遍历 step 函数、输出结构化日志、判定与汇总。

    子类约定：
      - 实现 run() 顺序执行各 step（可用 with logger.step_context(...) 计时）；
      - 每步判定用 self.check(passed, msg) 获得 bool 并（可选）记录到 reporter；
      - run() 返回退出码（0=全通过，1=存在失败），与既有约定一致。
    """

    def __init__(self, config: Any, synth: Any, reporter: Any = None) -> None:
        self.config = config      # 通常是 ConfigFactory 或 ParamConfig
        self.synth = synth        # FingerprintSynthesizer（或 None）
        self.reporter = reporter  # ReportGenerator（或 None）
        self._check_seq = 0       # check() 自动编号

    @abstractmethod
    def run(self) -> int:
        """抽象方法：遍历执行各 step 验证函数，返回退出码。"""
        raise NotImplementedError

    def _log_step(
        self,
        step_id: int,
        name: str,
        elapsed_ms: float,
        status: str,
        **extra: Any,
    ) -> None:
        """输出结构化 JSON 日志（转发至 _logging 单例）。"""
        logger.step(
            step_id=int(step_id),
            name=name,
            elapsed_ms=float(elapsed_ms),
            status=status,
            **extra,
        )

    def check(self, passed: bool, msg: str) -> bool:
        """校验判定：返回 bool（供短路 continue）。

        附带行为：若存在 reporter，自动追加一条
        "check{序号}" 记录（status 取 PASS/FAIL），便于报告追踪。
        """
        ok = bool(passed)
        if self.reporter is not None:
            self._check_seq += 1
            self.reporter.add(
                f"check{self._check_seq}",
                msg,
                "PASS" if ok else "FAIL",
                {"passed": ok},
            )
        return ok


# ---------------------------------------------------------------- ReportGenerator
class ReportGenerator:
    """报告生成器基类：汇总各步骤结果并提供多格式渲染与退出码。

    用法：
      rep = ReportGenerator()
      rep.add(1, "repro", "PASS", "tol ok")
      rep.add(2, "family", "FAIL", "sep < thr")
      print(rep.render_text())
      code = rep.exit_code          # 0 全通过 / 1 存在失败
    """

    _OK_STATUSES = ("PASS", "OK", "PASSED", "SUCCESS", "TRUE", "1")

    def __init__(self) -> None:
        self._items: list[dict] = []

    def add(self, step_id: Any, name: str, status: str, details: Any = None) -> ReportGenerator:
        """追加一条步骤记录；返回 self 支持链式调用。"""
        self._items.append(
            {
                "step_id": step_id,
                "name": name,
                "status": str(status),
                "details": details,
            }
        )
        return self

    # -------------------------------------------------------- 汇总属性
    @property
    def n_items(self) -> int:
        """已记录步骤数。"""
        return len(self._items)

    @property
    def passed(self) -> bool:
        """全部通过标志（空报告视为通过）。"""
        return all(
            str(i["status"]).upper() in self._OK_STATUSES for i in self._items
        )

    @property
    def exit_code(self) -> int:
        """退出码：0=全部通过，1=存在失败（与既有约定一致）。"""
        return 0 if self.passed else 1

    @staticmethod
    def _is_ok(status: str) -> bool:
        return str(status).upper() in ReportGenerator._OK_STATUSES

    # -------------------------------------------------------- 渲染
    def render_text(self) -> str:
        """文本渲染：逐条 PASS/FAIL 标记 + 底部汇总。"""
        lines = ["=" * 74]
        for it in self._items:
            mark = "PASS" if self._is_ok(it["status"]) else "FAIL"
            lines.append(
                f"[{mark:4s}] {it['step_id']} {it['name']}: {it['status']}"
            )
            if it["details"]:
                lines.append(f"        {it['details']}")
        lines.append("=" * 74)
        tail = "全部通过" if self.passed else "存在失败"
        lines.append(f"汇总：{self.n_items} 项，{tail}（exit_code={self.exit_code}）")
        lines.append("=" * 74)
        return "\n".join(lines)

    def render_json(self) -> str:
        """JSON 渲染：含 passed/n_items/exit_code/items 的完整报告。"""
        return json.dumps(
            {
                "passed": self.passed,
                "n_items": self.n_items,
                "exit_code": self.exit_code,
                "items": self._items,
            },
            ensure_ascii=False,
            indent=2,
        )

    def render_html(self) -> str:
        """HTML 渲染（可选能力）：表格 + 状态着色 + 通过统计。"""
        n_pass = sum(1 for i in self._items if self._is_ok(i["status"]))
        rows = "".join(
            f"<tr class='{('ok' if self._is_ok(i['status']) else 'fail')}'>"
            f"<td>{i['step_id']}</td><td>{i['name']}</td><td>{i['status']}</td>"
            f"<td>{i['details']}</td></tr>"
            for i in self._items
        )
        return (
            "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
            "<title>AIQ 验证报告</title>"
            "<style>table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccc;padding:6px;text-align:left}"
            ".ok{color:#0a0}.fail{color:#c00}</style></head><body>"
            f"<h1>AIQ 验证报告</h1><p>通过 {n_pass} / {self.n_items}</p>"
            "<table><tr><th>step_id</th><th>name</th><th>status</th>"
            f"<th>details</th></tr>{rows}</table></body></html>"
        )


if __name__ == "__main__":
    # 自检：取值优先级 / 类型化访问 / 模型构造 / 引擎与报告
    cf = ConfigFactory()

    # 1) 基础读取（_params_data.json 有 T01.PLATFORM=1.56）
    v = cf.get_float("T01", "PLATFORM", 1.56)
    print("T01.PLATFORM =", v)
    assert v == 1.56

    # 2) 环境变量优先级（AIQ_T01_PLATFORM）
    os.environ["AIQ_T01_PLATFORM"] = "9.9"
    assert cf.get_float("T01", "PLATFORM", 1.56) == 9.9
    del os.environ["AIQ_T01_PLATFORM"]
    assert cf.get_float("T01", "PLATFORM", 1.56) == 1.56

    # 3) ParamConfig 泛型访问
    pc = ParamConfig(cf, "T01")
    print("ParamConfig.PLATFORM =", pc.PLATFORM)
    assert pc.PLATFORM == 1.56

    # 4) dataclass 构造回退
    @dataclasses.dataclass
    class DemoDataclass:
        platform: float = 1.56
        n_bins: int = 100

    m = cf.build_model(DemoDataclass, "T01")
    assert isinstance(m, DemoDataclass) and m.platform == 1.56 and m.n_bins == 100
    print("dataclass model:", m)

    # 5) pydantic 构造
    try:
        import pydantic

        class DemoPydantic(pydantic.BaseModel):
            platform: float = 1.56
            n_bins: int = 100

        pm = cf.build_model(DemoPydantic, "T01")
        assert isinstance(pm, DemoPydantic) and pm.platform == 1.56
        print("pydantic model:", pm)
    except Exception as e:  # pragma: no cover - pydantic 缺失路径
        print("pydantic 不可用，跳过:", e)

    # 6) 报告生成器
    rep = ReportGenerator()
    rep.add(1, "repro", "PASS", "tol ok")
    rep.add(2, "family", "FAIL", "sep < thr")
    print(rep.render_text())
    assert rep.n_items == 2 and not rep.passed and rep.exit_code == 1
    json_repr = rep.render_json()
    assert json.loads(json_repr)["n_items"] == 2
    assert "<html" in rep.render_html()

    # 7) 引擎基类（子类实现 run + check 记录）
    class DemoEngine(ValidatorEngine):
        def run(self) -> int:
            with logger.step_context(1, "demo"):
                assert self.check(True, "ok")
                self.check(False, "bad")     # 失败判定仅记录，不抛异常
            return self.reporter.exit_code if self.reporter else 0

    eng = DemoEngine(cf, None, reporter=rep)
    code = eng.run()
    print("engine run() exit_code =", code)
    assert code == 1

    print("_factory 自检 PASS")
