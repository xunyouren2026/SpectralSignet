# -*- coding: utf-8 -*-
"""共享基建（_factory / _errors / _logging / _perf）— pytest 测试套件
=====================================================================
覆盖：
  - ConfigFactory：取值优先级（env > json > default）、类型化强制、
    build_model（pydantic 优先 / dataclass 回退）
  - _errors：AIQValidationError 子类的 expected/actual/param_key 字段
  - _logging：结构化 JSON 行可 json.loads，含 step_id/elapsed_ms/status
  - _perf：cached_synth 缓存命中、LazyInterp 惰性（mock spy）、
    float32_corr_diff 精度、synth_batch 形状校验
  - verify 的 pydantic 缺失 dataclass 回退（覆盖四层工厂 dataclass 路径）
=====================================================================
"""
from __future__ import annotations

import dataclasses
import importlib
import io
import json
import os
import sys
from unittest import mock

import numpy as np
import pytest

import verify
from _errors import AIQValidationError, ConfigError, SynthesisError
from _factory import ConfigFactory, ReportGenerator
from _logging import logger
from _perf import LazyInterp, cached_synth, float32_corr_diff, synth_batch, to_float32


# ---------------------------------------------------------------- ConfigFactory
def test_config_factory_priority(monkeypatch):
    """优先级：环境变量 > _params_data.json > default。"""
    cf = ConfigFactory()
    # json 源：A01.CORR_MIN 集中配置 = 0.99
    assert cf.load_param("A01", "CORR_MIN", 0.5, "float") == 0.99
    # default 兜底：不存在的键
    assert cf.load_param("A01", "NO_SUCH_KEY", 3.14, "float") == 3.14
    # env 最高优先级（自动大写）
    monkeypatch.setenv("AIQ_A01_CORR_MIN", "0.95")
    assert cf.load_param("A01", "CORR_MIN", 0.5, "float") == 0.95
    monkeypatch.delenv("AIQ_A01_CORR_MIN", raising=False)
    assert cf.load_param("A01", "CORR_MIN", 0.5, "float") == 0.99


def test_config_factory_type_coercion(monkeypatch):
    """类型化读取（int/float/str/list/grid）+ 非法值容错。"""
    cf = ConfigFactory()
    monkeypatch.setenv("AIQ_A01_GRID", "12")
    assert cf.get_int("A01", "GRID", 24) == 12
    monkeypatch.setenv("AIQ_A01_SFT_SCALE", "1.1")
    assert cf.get_float("A01", "SFT_SCALE", 1.005) == 1.1
    monkeypatch.setenv("AIQ_A01_DTYPE", "bfloat16")
    assert cf.get_str("A01", "DTYPE", "float32") == "bfloat16"
    monkeypatch.setenv("AIQ_A01_LISTKEY", "[1, 2, 3]")
    assert cf.get_list("A01", "LISTKEY", []) == [1, 2, 3]
    monkeypatch.setenv("AIQ_A01_CSVLIST", "a, b")
    assert cf.get_list("A01", "CSVLIST", []) == ["a", "b"]
    monkeypatch.setenv("AIQ_A01_GRIDKEY", "[0.0, 1.0, 20]")
    assert cf.get_grid("A01", "GRIDKEY", (0.0, 1.0, 100)) == (0.0, 1.0, 20)
    # 非法 int → 回退默认
    monkeypatch.setenv("AIQ_A01_GRID", "abc")
    assert cf.get_int("A01", "GRID", 24) == 24


def test_config_factory_build_model():
    """build_model 构造 pydantic A01Config（env > json > 模型默认值）。"""
    cf = ConfigFactory()
    m = cf.build_model(verify.A01Config, "A01")
    assert m.N_LAYERS_QWEN == 24 and m.CORR_MIN == 0.99
    assert m.DTYPE == "float32" and m.GRID == 24 and m.SEED == 0


def test_config_factory_dataclass_build():
    """build_model 的 dataclass 回退路径（_build_dataclass 运行时校验）。"""
    cf = ConfigFactory()

    @dataclasses.dataclass
    class Demo:
        corr_min: float = 0.99
        n_layers: int = 24

    m = cf.build_model(Demo, "A01")
    assert isinstance(m, Demo)
    assert m.corr_min == 0.99 and m.n_layers == 24


def test_config_factory_unsupported_model():
    """build_model 遇到不支持的类型抛 ConfigError。"""
    cf = ConfigFactory()
    with pytest.raises(ConfigError):
        cf.build_model(int, "A01")  # 非 pydantic / 非 dataclass


# ---------------------------------------------------------------- 取值工具
def test_common_finish(capsys):
    """_common.finish：PASS 返回 0 / FAIL 返回 1。"""
    from _common import finish

    assert finish(True, 6) == 0
    assert finish(False, 6) == 1
    out = capsys.readouterr().out
    assert "全部 6 项 PASS" in out and "存在 FAIL" in out


# ---------------------------------------------------------------- _errors
def test_errors_field_roundtrip():
    """AIQValidationError 子类携带 expected/actual/param_key。"""
    e = ConfigError("配置错误", expected=1, actual=2, param_key="A01")
    assert isinstance(e, AIQValidationError)
    assert e.expected == 1 and e.actual == 2 and e.param_key == "A01"
    assert e.args == ("配置错误", 1, 2, "A01")
    assert "expected=1" in str(e) and "actual=2" in str(e)
    s = SynthesisError("合成失败", actual=(2,))
    assert s.expected is None and s.actual == (2,)
    c = ConfigError("缺少配置段")
    assert c.expected is None and c.actual is None and c.param_key is None
    assert c.args == ("缺少配置段", None, None, None)


# ---------------------------------------------------------------- _logging
def test_logging_json_line():
    """_logging 输出行可 json.loads，且含 step_id/elapsed_ms/status。"""
    buf = io.StringIO()
    old_err = sys.stderr
    try:
        sys.stderr = buf
        logger.configure()
        logger.step(3, "repro", 12.34, "PASS", tol=1e-3, note="test")
    finally:
        sys.stderr = old_err
    line = buf.getvalue().strip()
    assert line, "应有结构化日志行"
    rec = json.loads(line)  # 每行必须是合法 JSON
    assert rec["step_id"] == 3
    assert rec["name"] == "repro"
    assert rec["elapsed_ms"] == 12.34
    assert rec["status"] == "PASS"
    assert rec["extra"]["tol"] == 1e-3 and rec["extra"]["note"] == "test"


def test_logging_step_context():
    """step_context：正常记 PASS、异常记 FAIL（extra.error）并重抛。"""
    buf = io.StringIO()
    old_err = sys.stderr
    try:
        sys.stderr = buf
        logger.configure()
        with logger.step_context(1, "demo_ok"):
            pass
        with pytest.raises(RuntimeError):
            with logger.step_context(2, "demo_fail"):
                raise RuntimeError("boom")
    finally:
        sys.stderr = old_err
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    assert len(lines) == 2
    r1, r2 = json.loads(lines[0]), json.loads(lines[1])
    assert r1["status"] == "PASS" and r1["step_id"] == 1
    assert r2["status"] == "FAIL" and r2["step_id"] == 2
    assert "error" in r2["extra"] and "RuntimeError" in r2["extra"]["error"]


# ---------------------------------------------------------------- _perf
def test_cached_synth_hit():
    """cached_synth 二次调用命中 lru_cache（synth_fn 仅执行一次）。"""
    calls = []

    def fake(family, variant, noise, seed):
        calls.append((family, variant, noise, seed))
        return np.array([1.0, 2.0, 3.0])

    cached = cached_synth(fake)
    a = cached("qwen", "base", 0.0, 0)
    b = cached("qwen", "base", 0.0, 0)
    assert a is b
    assert len(calls) == 1
    assert cached.cache_info().hits == 1
    # 不同键不命中
    cached("gpt2", "base", 0.0, 0)
    assert len(calls) == 2
    # 标准属性存在
    assert hasattr(cached, "cache_clear")
    assert cached.__name__.endswith("_cached")


def test_synth_cached_reuse(cfg, synth):
    """合成器 cached 包装：同一键二次命中（lru_cache 语义）。"""
    a = synth.cached("qwen", "base", 0.0, cfg.SEED)
    b = synth.cached("qwen", "base", 0.0, cfg.SEED)
    assert a is b
    assert synth.cached.cache_info().hits >= 1


def test_lazy_interp_lazy_and_cached():
    """LazyInterp：resolve 前不执行插值；二次 resolve 命中缓存（mock spy）。"""
    src_x = np.linspace(0, 1, 12)
    src_y = np.sin(src_x * np.pi)
    dst_x = np.linspace(0, 1, 24)
    interp = LazyInterp(src_x, src_y, dst_x)
    assert not interp.resolved  # 构造零成本，未执行 np.interp
    with mock.patch.object(np, "interp", wraps=np.interp) as spy:
        y1 = interp.resolve()
        assert spy.call_count == 1
        y2 = interp.resolve()  # 命中缓存
        assert spy.call_count == 1
        assert y1 is y2
        y3 = interp()  # __call__ 等价 resolve
        assert spy.call_count == 1
    assert interp.resolved
    assert y1.shape == (24,)
    assert np.allclose(y1, np.sin(dst_x * np.pi), atol=0.02)


def test_float32_corr_diff_precision():
    """float64 vs float32 corr 差 < 1e-5；空数组兜底 corr=1。"""
    rng = np.random.default_rng(0)
    ref64 = np.linspace(0.0, 1.0, 5000) + rng.normal(0, 1e-4, 5000)
    val32 = to_float32(ref64)
    stats = float32_corr_diff(ref64, val32)
    assert abs(1.0 - stats["corr"]) < 1e-5
    assert stats["max_abs_diff"] < 1e-5
    assert stats["n"] == 5000
    # 空数组兜底
    empty = float32_corr_diff(np.array([]), np.array([]))
    assert empty["corr"] == 1.0 and empty["n"] == 0
    # 长度不一致抛 SynthesisError
    with pytest.raises(SynthesisError):
        float32_corr_diff(np.zeros(3), np.zeros(4))


def test_synth_batch_stack_and_mismatch():
    """synth_batch 堆叠 float32；形状不一致抛 SynthesisError。"""
    ok_batch = synth_batch(
        lambda f, v, n, s: np.array([n, n], dtype=np.float32),
        [("qwen1", "base", 0.1, 1), ("qwen2", "base", 0.2, 2)],
    )
    assert ok_batch.shape == (2, 2) and ok_batch.dtype == np.float32
    # 空列表 → (0,) 空数组
    empty = synth_batch(lambda *p: np.zeros(4), [])
    assert empty.shape == (0,) and empty.dtype == np.float32

    def fake_shape(family, variant, noise, seed):
        return np.zeros(len(family))  # qwen→4, gpt-2→5（长度不同 → 形状不一致）

    with pytest.raises(SynthesisError):
        synth_batch(fake_shape, [("qwen", "base", 0.0, 0), ("gpt-2", "base", 0.0, 0)])


# ---------------------------------------------------------------- 报告器
def test_report_generator():
    """ReportGenerator：追加/汇总/多格式渲染/退出码。"""
    rep = ReportGenerator()
    assert rep.n_items == 0 and rep.passed and rep.exit_code == 0
    rep.add(1, "repro", "PASS", "ok").add(2, "family", "FAIL", "sep<thr")
    assert rep.n_items == 2 and not rep.passed and rep.exit_code == 1
    text = rep.render_text()
    assert "[PASS]" in text and "[FAIL]" in text and "存在失败" in text
    data = json.loads(rep.render_json())
    assert data["n_items"] == 2 and data["exit_code"] == 1
    assert "<html" in rep.render_html() and "AIQ 验证报告" in rep.render_html()


# ---------------------------------------------------------------- _factory 深覆盖
def test_factory_type_tools():
    """_looks_undefined / _type_name 全部分支。"""
    from _factory import _looks_undefined, _type_name

    assert _looks_undefined(...) is True
    assert _looks_undefined(None) is False
    assert _looks_undefined(5) is False
    assert _looks_undefined(type("PydanticUndefined", (), {})()) is True
    # 类型对象
    assert _type_name(int) == "int" and _type_name(bool) == "int"
    assert _type_name(float) == "float" and _type_name(str) == "str"
    assert _type_name(list) == "list" and _type_name(tuple) == "grid"
    assert _type_name(None) is None
    # 字符串注解（dataclass + future annotations）
    assert _type_name("int") == "int" and _type_name("boolean") == "int"
    assert _type_name("float") == "float" and _type_name("string") == "str"
    assert _type_name("typing.List[float]") == "list"
    assert _type_name("sequence[str]") == "list"
    assert _type_name("tuple[float, float, int]") == "grid"
    assert _type_name("grid") == "grid"
    assert _type_name("someobj") is None
    # 泛型对象
    assert _type_name(list[int]) == "list"
    assert _type_name(tuple[int, ...]) == "grid"
    # 其他类型对象 → None
    assert _type_name(dict) is None


def test_factory_yaml_priority(tmp_path):
    """YAML 配置优先级（> json）与损坏/缺失 YAML 回退。"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("A01:\n  CORR_MIN: 0.88\n  GRID: 48\n", encoding="utf-8")
    cf = ConfigFactory(config_path=str(yaml_file))
    assert cf.load_param("A01", "CORR_MIN", 0.5, "float") == 0.88
    assert cf.load_param("A01", "GRID", 24, "int") == 48
    # YAML 无此键 → 回落 json / default
    assert cf.load_param("A01", "SEED", 7, "int") == 7  # json 无 SEED → default
    # 损坏 YAML → 静默回退 json
    bad = tmp_path / "bad.yaml"
    bad.write_text("{{{{ not yaml", encoding="utf-8")
    cf2 = ConfigFactory(config_path=str(bad))
    assert cf2.load_param("A01", "CORR_MIN", 0.5, "float") == 0.99
    # 文件不存在 → 回退
    cf3 = ConfigFactory(config_path=str(tmp_path / "nope.yaml"))
    assert cf3.load_param("A01", "CORR_MIN", 0.5, "float") == 0.99


def test_factory_get_fallbacks(monkeypatch):
    """get_int/get_float/get_str/get_list/get_grid 的异常兜底 + _coerce 直调。"""
    cf = ConfigFactory()
    monkeypatch.setenv("AIQ_A01_SFT_SCALE", "abc")
    assert cf.get_float("A01", "SFT_SCALE", 1.005) == 1.005
    monkeypatch.setenv("AIQ_A01_GRID", "abc")
    assert cf.get_int("A01", "GRID", 24) == 24
    # list 非法迭代 → 兜底 []
    assert cf.get_list("A01", "NO_SUCH", None) == []
    # grid 非法元素 → 兜底原样 default
    assert cf.get_grid("A01", "NO_SUCH", (0, 1, "x")) == (0, 1, "x")
    # _coerce 直调分支
    assert cf._coerce(None, "int", 5) == 5
    assert cf._coerce([1, 2], "list", []) == [1, 2]
    assert cf._coerce("abc", "grid", (0, 1, 10)) == "abc"  # JSON 解析失败原样返回
    assert cf._coerce("3", "unknown_type", 0) == "3"        # 未知类型原样返回


def test_factory_build_pydantic_default_factory():
    """_build_pydantic 的 default_factory 取值路径。"""
    import pydantic

    cf = ConfigFactory()

    class Demo(pydantic.BaseModel):
        name: str = "x"
        tags: list = pydantic.Field(default_factory=list)

    m = cf.build_model(Demo, "A01")
    assert m.name == "x" and m.tags == []


def test_factory_build_dataclass_default_factory():
    """_build_dataclass 的 default_factory 取值路径。"""
    cf = ConfigFactory()

    @dataclasses.dataclass
    class Demo2:
        tags: list = dataclasses.field(default_factory=list)
        n: int = 3

    m = cf.build_model(Demo2, "A01")
    assert m.tags == [] and m.n == 3


def test_factory_params_get_failure():
    """_params_get：_params 导入失败时返回 None。"""
    cf = ConfigFactory()
    with mock.patch.dict(sys.modules, {"_params": None}):
        assert cf._params_get("A01", "CORR_MIN") is None


def test_factory_pydantic_unavailable():
    """_pydantic_available：pydantic 缺失时返回 False（带缓存）。"""
    cf = ConfigFactory()
    with mock.patch.dict(sys.modules, {"pydantic": None}):
        assert cf._pydantic_available() is False
    # 恢复后重新探测可用
    cf2 = ConfigFactory()
    assert cf2._pydantic_available() is True


def test_param_config_access():
    """ParamConfig 属性/方法访问 + 私有名抛 AttributeError。"""
    from _factory import ParamConfig

    pc = ParamConfig(ConfigFactory(), "A01")
    assert pc.get("CORR_MIN", 0.5, "float") == 0.99
    assert pc.CORR_MIN == 0.99
    with pytest.raises(AttributeError):
        _ = pc._private


def test_factory_synthesizer_base():
    """基类合成器：seed_rng 语义 + synthesize 抛 NotImplementedError。"""
    from _factory import FingerprintSynthesizer as BaseSynth

    bs = BaseSynth(cfg=None, seed=7)
    assert bs.seed_rng() == 7
    assert bs.seed_rng(42) == 42
    with pytest.raises(NotImplementedError):
        bs.synthesize("x")


def test_factory_validator_engine_base(cfg):
    """引擎基类：_log_step / check 记录 / 抽象 run 抛 NotImplementedError。"""
    from _factory import ValidatorEngine

    rep = ReportGenerator()

    class DemoE(ValidatorEngine):
        def run(self):
            return super().run()

    eng = DemoE(cfg, None, reporter=rep)
    eng._log_step(1, "parse", 0.5, "PASS", tol=1.0)
    assert eng.check(True, "ok") is True
    assert eng.check(False, "bad") is False
    assert rep.n_items == 2  # check 自动记录两条
    with pytest.raises(NotImplementedError):
        eng.run()


# ---------------------------------------------------------------- _logging 深覆盖
def test_logging_configure_with_log_dir(tmp_path):
    """configure(log_dir)：结构化日志写入 log_dir/structured.log。"""
    logger.configure(str(tmp_path))
    logger.step(1, "file_log", 1.5, "PASS")
    log_file = tmp_path / "structured.log"
    assert log_file.is_file()
    rec = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert rec["step_id"] == 1 and rec["name"] == "file_log"
    assert rec["status"] == "PASS"


# ---------------------------------------------------------------- _params 深覆盖
def test_params_real_lookup_branches():
    """_params 真实测量查找分支（mock _real）。"""
    import _params as pmod

    with mock.patch.object(pmod, "_real", return_value=None):
        assert pmod._real_lookup("A01", "CORR_MIN") is None
        assert pmod.source_tag() == "集中配置(无真实测量)"
    with mock.patch.object(pmod, "_real", return_value={"params": {"A01": {"CORR_MIN": 0.5}}}):
        assert pmod.get("A01", "CORR_MIN", None) == 0.5
    with mock.patch.object(pmod, "_real", return_value={"CORR_MIN": 0.7}):
        assert pmod.get("A01", "CORR_MIN", None) == 0.7
    # 真实数据优先于集中配置
    assert pmod.get("A01", "CORR_MIN", None) == 0.99  # 恢复后回 json


def test_params_typed_getters_fallbacks():
    """get_float/get_int/get_list/get_grid 的类型强制与兜底。"""
    import _params as pmod

    with mock.patch.object(pmod, "get", return_value="abc"):
        assert pmod.get_float("A01", "X", 1.5) == 1.5
        assert pmod.get_int("A01", "X", 3) == 3
    with mock.patch.object(pmod, "get", return_value=(1, 2, 3)):
        assert pmod.get_list("A01", "X", []) == [1, 2, 3]
    with mock.patch.object(pmod, "get", return_value=5):
        assert pmod.get_list("A01", "X", None) == []
    with mock.patch.object(pmod, "get", return_value=[0.1, 1.0, 10]):
        assert pmod.get_grid("A01", "X", (0.0, 1.0, 100)) == (0.1, 1.0, 10)
    with mock.patch.object(pmod, "get", return_value="bad"):
        assert pmod.get_grid("A01", "X", (0.0, 1.0, 100)) == (0.0, 1.0, 100)


def test_params_load_json_fallbacks():
    """_load_json：文件缺失 / 损坏 → None。"""
    import _params as pmod

    with mock.patch("_params.os.path.isfile", return_value=False):
        assert pmod._load_json("whatever.json") is None
    with mock.patch("_params.os.path.isfile", return_value=True), \
            mock.patch("_params.open", side_effect=OSError("boom")):
        assert pmod._load_json("whatever.json") is None


# ---------------------------------------------------------------- _real_data 深覆盖
def test_real_data_metrics_fallbacks():
    """metrics()：文件缺失 / 解析失败 → None（mock 缓存+磁盘）。"""
    import _real_data as rd_mod

    with mock.patch.object(rd_mod, "_metrics_cache", None), \
            mock.patch("_real_data.os.path.isfile", return_value=False):
        assert rd_mod.metrics() is None
    with mock.patch.object(rd_mod, "_metrics_cache", None), \
            mock.patch("_real_data.os.path.isfile", return_value=True), \
            mock.patch("_real_data.open", side_effect=OSError("boom")):
        assert rd_mod.metrics() is None


def test_real_data_real_or():
    """real_or：元组路径命中 / 缺失回退 / 真实缺失时回退审计值。"""
    import _real_data as rd_mod

    assert rd_mod.real_or(("spectral.k_proj_gamma_mean", 0.625)) == pytest.approx(0.4695, abs=1e-3)
    assert rd_mod.real_or(("spectral.NO_SUCH", 0.5)) == 0.5
    assert rd_mod.real_or(0.625) == 0.625  # 标量直接返回
    # get：路径中途缺键 → default（非终点缺键）
    assert rd_mod.get("spectral.NO_SUCH.deep", 1.0) == 1.0
    with mock.patch.object(rd_mod, "_metrics_cache", None), \
            mock.patch("_real_data.os.path.isfile", return_value=False):
        assert rd_mod.real_or(("spectral.k_proj_gamma_mean", 0.625)) == 0.625
        assert rd_mod.get("arch.n_layers", None) is None
        assert rd_mod.has_real() is False


def test_real_data_phi_pairs():
    """phi_pairs：正常加载 / 缺失 / 形状非法 / 加载异常 → 兜底。"""
    import _real_data as rd_mod

    pairs = rd_mod.phi_pairs()
    if pairs is not None:  # 数据存在时验证形状
        assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert rd_mod.has_phi() == (pairs is not None)
    with mock.patch.object(rd_mod, "_pairs_cache", None), \
            mock.patch("_real_data.os.path.isfile", return_value=False):
        assert rd_mod.phi_pairs() is None
    with mock.patch.object(rd_mod, "_pairs_cache", None), \
            mock.patch("_real_data.os.path.isfile", return_value=True), \
            mock.patch("_real_data.np.load", return_value=np.zeros((3, 3))):
        assert rd_mod.phi_pairs() is None  # ndim!=2 → 非法
    with mock.patch.object(rd_mod, "_pairs_cache", None), \
            mock.patch("_real_data.os.path.isfile", return_value=True), \
            mock.patch("_real_data.np.load", side_effect=Exception("corrupt")):
        assert rd_mod.phi_pairs() is None  # 加载异常 → 兜底


# ---------------------------------------------------------------- _cfg 深覆盖
def test_cfg_env_overrides(tmp_path, monkeypatch):
    """_cfg 环境变量覆盖：AIQ_ROOT / AIQ_MODELS_DIR / AIQ_PHI_PAIRS。"""
    import _cfg

    d = tmp_path / "proj"
    d.mkdir()
    models = d / "_models"
    models.mkdir()
    monkeypatch.setenv("AIQ_ROOT", str(d))
    assert _cfg.project_root() == str(d)
    monkeypatch.setenv("AIQ_MODELS_DIR", str(models))
    assert _cfg.models_dir() == str(models)
    fp = tmp_path / "pairs.npy"
    fp.write_bytes(b"x")
    monkeypatch.setenv("AIQ_PHI_PAIRS", str(fp))
    assert _cfg.phi_pairs_path() == str(fp)
    # resolve 相对根路径
    assert _cfg.resolve("AIQ/x") == str(d / "AIQ" / "x")
    # model_path 存在/缺失
    (models / "M1").mkdir()
    assert _cfg.model_path("M1") == str(models / "M1")
    assert _cfg.model_path("NOPE") is None


def test_cfg_fallbacks(tmp_path, monkeypatch):
    """_cfg 兜底分支：models_dir 探测循环 / phi_pairs 本地副本缺失。"""
    import _cfg

    # models_dir：项目根无 _models → 向上探测最近的 _models
    empty = tmp_path / "empty"
    empty.mkdir()
    with mock.patch("_cfg.project_root", return_value=str(empty)):
        found = _cfg.models_dir()
        assert isinstance(found, str) and found  # 探测到真实 _models 或理论路径
    # phi_pairs：env 未设 + 本地副本缺失 → 回退 resolve（AIQ/phi_pairs_all.npy）
    monkeypatch.delenv("AIQ_PHI_PAIRS", raising=False)
    with mock.patch("_cfg.os.path.isfile", return_value=False):
        assert _cfg.phi_pairs_path().endswith("phi_pairs_all.npy")


# ---------------------------------------------------------------- 模块自举分支
def test_shared_libs_path_bootstrap():
    """共享库自举 sys.path.insert 分支（params 目录不在 sys.path 时触发）。"""
    import importlib.util
    import _real_data as rd_mod

    params_dir = os.path.dirname(os.path.abspath(rd_mod.__file__))
    saved = list(sys.path)
    try:
        for fname in ("_common.py", "_factory.py", "_params.py", "_real_data.py"):
            # 每次执行前移除 params 目录，使模块顶部自举 insert 分支触发
            sys.path[:] = [
                p for p in sys.path
                if os.path.normcase(p) != os.path.normcase(params_dir)
            ]
            name = "_bootstrap_" + fname[:-3]
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(params_dir, fname)
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)  # 触发顶部 sys.path.insert(0, params)
            finally:
                sys.modules.pop(name, None)
        assert params_dir in sys.path  # 自举已重新注入
    finally:
        sys.path[:] = saved


# ---------------------------------------------------------------- _common 深覆盖
def test_common_setup_env_stdout_fallback():
    """setup_env：stdout 不支持 reconfigure 时静默降级（except 分支）。"""
    import _common as common_mod

    real_stdout = sys.stdout
    saved = list(sys.path)
    try:
        sys.stdout = io.StringIO()  # 无 reconfigure → AttributeError → 静默降级
        common_mod.setup_env(common_mod.__file__)
    finally:
        sys.stdout = real_stdout
        sys.path[:] = saved


def test_common_data_source_tag_fallback():
    """data_source_tag：_real_data 不可用时回退 '[审计回退]'。"""
    import _common as common_mod

    with mock.patch.object(common_mod, "_real_data", None):
        assert common_mod.data_source_tag() == "[审计回退]"


# ---------------------------------------------------------------- verify dataclass 回退
def test_verify_pydantic_fallback_dataclass():
    """pydantic 缺失时 verify.A01Config 回退 dataclass（覆盖四层工厂 dataclass 路径）。"""
    with mock.patch.dict(sys.modules, {"pydantic": None}):
        verify_mod = importlib.reload(verify)
    assert verify_mod._HAS_PYDANTIC is False
    cfg = verify_mod.ConfigFactory().build()
    assert cfg.N_LAYERS_QWEN == 24 and cfg.CORR_MIN == 0.99
    synth = verify_mod.FingerprintSynthesizer(cfg)
    prof = synth.synth_beta_profile("qwen", "base", 0.0)
    assert prof.shape == (24,)
    eng = verify_mod.ValidatorEngine(cfg, synth)
    assert eng.validate_parse()["qwen"]["family"] == "qwen"
    # 恢复 pydantic 版本（后续用例继续使用 pydantic 路径）
    with mock.patch.dict(sys.modules, {"pydantic": None}):
        pass  # 仅清空环境干扰占位
    verify_mod = importlib.reload(verify)
    assert verify_mod._HAS_PYDANTIC is True
    assert verify_mod.A01Config(N_LAYERS_QWEN=24).N_LAYERS_QWEN == 24
