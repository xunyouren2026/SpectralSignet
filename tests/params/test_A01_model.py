# -*- coding: utf-8 -*-
"""A01 model 四层工厂架构 — pytest 测试套件
=====================================================================
覆盖（与原 verify.py 的 6 项验证一一对应）：
  1. model_id 解析（家族/变体/n_layers 映射）
  2. 可复现性（同种子两次合成逐元素相等）
  3. SFT 不变性（base↔instruct corr > CORR_MIN，中层β最高）
  4. 家族分离（分离比 > SEP_RATIO_MIN）
  5. 溯源（21 样本 acc=100%）
  6. 真实模型对照（真实层数一致性；RD 离线场景 mock 降级）

另含：边界测试（n_layers=0 / noise 负值·极大值 / 空 model_id）、
pytest-benchmark 性能基准（synth/interp < 0.1ms）、环境变量覆盖。
=====================================================================
"""
from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest

import verify
from _errors import (
    ConfigError,
    FamilySeparationError,
    RealModelMismatchError,
    ReproducibilityError,
    SFTInvarianceError,
    SynthesisError,
    TraceabilityError,
)
from _factory import ReportGenerator

SEED = 0  # H01 固定种子（与 A01Config.SEED 一致）


# ---------------------------------------------------------------- 1) 解析
def test_parse_model_id(cfg):
    """qwen/gpt2/unknown 家族与 variant 解析、n_layers 映射。"""
    # Qwen 系：instruct 变体
    q = verify.parse_model_id("Qwen2.5-0.5B-Instruct", cfg)
    assert q["family"] == "qwen"
    assert q["variant"] == "instruct"
    assert q["n_layers"] == cfg.N_LAYERS_QWEN
    assert q["dtype"] == cfg.DTYPE
    assert q["seed"] == cfg.SEED
    assert q["local_files_only"] is True
    # GPT-2 系：plain 变体 + 自有层数（12）
    g = verify.parse_model_id("gpt2-124m", cfg)
    assert g["family"] == "gpt2"
    assert g["variant"] == "plain"
    assert g["n_layers"] == cfg.N_LAYERS_GPT2
    # 大小写不敏感 + 前缀 "gpt-2" 分支 + base 变体
    q2 = verify.parse_model_id("QwEn2.5-0.5B-base", cfg)
    assert q2["family"] == "qwen" and q2["variant"] == "base"
    g2 = verify.parse_model_id("GPT-2-355M", cfg)
    assert g2["family"] == "gpt2" and g2["n_layers"] == cfg.N_LAYERS_GPT2
    # 引擎第 1 步（parse 验证项）结果一致
    eng = verify.ValidatorEngine(cfg, verify.FingerprintSynthesizer(cfg))
    res = eng.validate_parse()
    assert res["qwen"]["n_layers"] == cfg.N_LAYERS_QWEN
    assert res["gpt2"]["n_layers"] == cfg.N_LAYERS_GPT2


# ---------------------------------------------------------------- 2) 可复现
def test_reproducibility(cfg, synth):
    """同种子两次合成逐元素相等（H01 语义）。"""
    p1 = synth.synth_beta_profile("qwen", "base", 0.0)                      # 默认 rng（固定种子）
    p2 = synth.synth_beta_profile("qwen", "base", 0.0)
    assert np.array_equal(p1, p2)
    # 独立 rng（同种子）同样逐元素相等——真正验证"同种子→同指纹"
    p3 = synth.synth_beta_profile("qwen", "base", 0.0, np.random.default_rng(SEED))
    assert np.array_equal(p1, p3)
    # 引擎第 2 步
    eng = verify.ValidatorEngine(cfg, synth)
    assert eng.validate_reproducibility()["seed"] == cfg.SEED


# ---------------------------------------------------------------- 3) SFT 不变性
def test_sft_invariance(cfg, synth):
    """base↔instruct corr > CORR_MIN，且中层β最高（anchor 依据）。"""
    rng = np.random.default_rng(SEED)
    g_base = synth.synth_beta_profile("qwen", "base", 0.0003, rng)
    g_instruct = synth.sft_perturb(g_base, rng)
    corr = float(np.corrcoef(g_base, g_instruct)[0, 1])
    assert corr > cfg.CORR_MIN
    mid, sh, dp = g_base[8:16].mean(), g_base[0:8].mean(), g_base[16:24].mean()
    assert mid > sh and mid > dp
    # 引擎第 3 步
    eng = verify.ValidatorEngine(cfg, synth)
    res = eng.validate_sft_invariance()
    assert res["corr"] > cfg.CORR_MIN
    assert res["mid_beta"] > res["sh_beta"] and res["mid_beta"] > res["dp_beta"]


# ---------------------------------------------------------------- 4) 家族分离
def test_family_separation(cfg, synth):
    """分离比 > SEP_RATIO_MIN（cross >> within）。"""
    rng = np.random.default_rng(SEED)
    g_base = synth.synth_beta_profile("qwen", "base", 0.0003, rng)
    g_instruct = synth.sft_perturb(g_base, rng)
    g_gpt2 = synth.interp_to_grid(synth.cached("gpt2", "base", 0.0, 7), cfg.GRID)
    within = verify.l2_dist(g_base, g_instruct)
    cross = verify.l2_dist(g_base, g_gpt2)
    ratio = cross / within
    assert within > 0.0
    assert ratio > cfg.SEP_RATIO_MIN
    # 引擎第 4 步（依赖前置步骤共享状态）
    eng = verify.ValidatorEngine(cfg, synth)
    eng.validate_sft_invariance()
    res = eng.validate_family_separation()
    assert res["ratio"] > cfg.SEP_RATIO_MIN


# ---------------------------------------------------------------- 5) 溯源
def test_traceability(cfg, synth):
    """21 样本家族溯源 acc=100%。"""
    eng = verify.ValidatorEngine(cfg, synth)
    eng.validate_sft_invariance()
    eng.validate_family_separation()
    res = eng.validate_traceability()
    assert res["acc_pct"] == 100.0
    assert res["n_ok"] == res["n_total"] == cfg.N_TOTAL


# ---------------------------------------------------------------- 6) 真实模型对照
def test_real_model_compare(cfg, synth):
    """真实层数一致性：真实 n_layers == cfg == parse（在线真实数据）。"""
    eng = verify.ValidatorEngine(cfg, synth)
    res = eng.validate_real_model()
    assert res["n_layers_real"] == cfg.N_LAYERS_QWEN == 24
    assert res["n_layers_real"] == res["n_layers_cfg"]
    assert res["n_kv"] == 2 and res["hd"] == 64 and res["hidden"] == 896
    assert res["tok_s_real"] is not None
    assert res["tag"] in ("[真实实测]", "[审计回退]")


def test_real_model_offline_degrades(cfg, synth):
    """RD 离线场景（mock 掉真实数据）：仅第 6 项抛 RealModelMismatchError，
    1-5 项照常通过；恢复真实数据后 run() 全过。"""
    import _real_data as rd_mod

    rep = ReportGenerator()
    eng = verify.ValidatorEngine(cfg, synth, reporter=rep, real_data=rd_mod)
    with mock.patch.object(rd_mod, "metrics", return_value=None):  # 模拟离线：无真实测量
        # 1-5 项不依赖真实数据 → 全部通过
        assert eng.validate_parse()["qwen"]["family"] == "qwen"
        assert eng.validate_reproducibility()["seed"] == cfg.SEED
        r3 = eng.validate_sft_invariance()
        assert r3["corr"] > cfg.CORR_MIN
        r4 = eng.validate_family_separation()
        assert r4["ratio"] > cfg.SEP_RATIO_MIN
        r5 = eng.validate_traceability()
        assert r5["acc_pct"] == 100.0
        # 第 6 项：真实层数不可得 → RealModelMismatchError
        with pytest.raises(RealModelMismatchError):
            eng.validate_real_model()
    # 恢复真实数据：run() 6 项全过，exit_code=0
    assert eng.run() == 0
    assert rep.passed and rep.n_items == 6


def test_real_model_offline_run_fail_only_step6(cfg, synth):
    """run() 集成：离线时仅第 6 项 FAIL，1-5 项 PASS，exit_code=1。"""
    import _real_data as rd_mod

    rep = ReportGenerator()
    eng = verify.ValidatorEngine(cfg, synth, reporter=rep, real_data=rd_mod)
    with mock.patch.object(rd_mod, "metrics", return_value=None):
        code = eng.run()
    assert code == 1  # 存在失败
    data = json.loads(rep.render_json())
    assert data["n_items"] == 6
    statuses = [it["status"] for it in data["items"]]
    assert statuses[:5] == ["PASS"] * 5
    assert statuses[5] == "FAIL"
    assert data["exit_code"] == 1


# ---------------------------------------------------------------- 边界测试
def test_synth_zero_layers_raises(cfg):
    """n_layers=0 抛 SynthesisError（防御：无法构造剖面）。"""
    bad_cfg = verify.A01Config(N_LAYERS_QWEN=0, N_LAYERS_GPT2=0)
    bad_synth = verify.FingerprintSynthesizer(bad_cfg)
    with pytest.raises(SynthesisError):
        bad_synth.synth_beta_profile("gpt2", "base", 0.0)
    with pytest.raises(SynthesisError):
        bad_synth.synth_beta_profile("qwen", "base", 0.0)


def test_synth_noise_extremes(cfg, synth):
    """noise 负值/极大值不崩溃；结果仍 clip 在 [0,1]（β 为能量占比）。"""
    for noise in (-1.0, -0.5, 0.0, 1.0, 1e6, 1e12):
        prof = synth.synth_beta_profile("qwen", "base", noise, np.random.default_rng(1))
        assert prof.shape == (cfg.N_LAYERS_QWEN,)
        assert float(prof.min()) >= 0.0 and float(prof.max()) <= 1.0
        assert np.isfinite(prof).all()
    # 极大噪声下（全 clip 到边界）仍不抛异常
    prof = synth.synth_beta_profile("gpt2", "base", 1e12, np.random.default_rng(2))
    assert prof.shape == (cfg.N_LAYERS_GPT2,)
    assert float(prof.min()) >= 0.0 and float(prof.max()) <= 1.0


def test_model_id_empty_unknown(cfg):
    """空字符串/未知家族 → unknown family，不崩溃。"""
    for mid in ("", " ", "llama-3-8b", "llama-3-8b-instruct", "Mistral-7B", "DeepSeek-V3"):
        r = verify.parse_model_id(mid, cfg)
        assert r["family"] == "unknown"
        assert r["variant"] in ("plain", "base", "instruct")  # 变体解析规则照常生效
        assert r["n_layers"] == cfg.N_LAYERS_QWEN  # unknown 走 Qwen 模板层数
    # 空串：无 instruct/base 子串 → plain
    assert verify.parse_model_id("", cfg)["variant"] == "plain"


# ---------------------------------------------------------------- 防御/错误路径
def test_synth_sft_perturb_empty(cfg, synth):
    """sft_perturb 空剖面抛 SynthesisError（防御分支）。"""
    with pytest.raises(SynthesisError):
        synth.sft_perturb(np.array([]), np.random.default_rng(0))


def test_synth_interp_bad_inputs(cfg, synth):
    """interp_to_grid 空剖面 / 非法格点抛 SynthesisError（防御分支）。"""
    prof = synth.synth_beta_profile("gpt2", "base", 0.0)
    with pytest.raises(SynthesisError):
        synth.interp_to_grid(np.array([]))
    for bad in (0, -5):
        with pytest.raises(SynthesisError):
            synth.interp_to_grid(prof, bad)


def test_verdict_both_branches(cfg, synth):
    """verdict 近邻判定：到 Qwen 更近 → Qwen家族；到 GPT-2 更近 → GPT-2家族。"""
    q = synth.synth_beta_profile("qwen", "base", 0.0)
    g = synth.interp_to_grid(synth.synth_beta_profile("gpt2", "base", 0.0), cfg.GRID)
    name, d_q, d_g = verify.verdict(q, q, g)
    assert name == "Qwen家族" and d_q == 0.0 and d_g > 0.0
    name2, d2_q, d2_g = verify.verdict(g, q, g)
    assert name2 == "GPT-2家族" and d2_g == 0.0 and d2_q > 0.0


def test_validate_parse_failure_raises(cfg, synth):
    """第 1 步失败路径：解析结果不符 → ConfigError（携带 expected/actual）。"""
    eng = verify.ValidatorEngine(cfg, synth)
    bad = {"family": "unknown", "variant": "plain", "dtype": cfg.DTYPE,
           "seed": cfg.SEED, "n_layers": 12}
    with mock.patch.object(verify, "parse_model_id", return_value=bad):
        with pytest.raises(ConfigError) as ei:
            eng.validate_parse()
    assert ei.value.param_key == "A01" and ei.value.expected["n_layers_qwen"] == 24


def test_reproducibility_failure_raises(cfg, synth):
    """第 2 步失败路径：同种子两次结果不同 → ReproducibilityError。"""
    eng = verify.ValidatorEngine(cfg, synth)
    p1 = synth.synth_beta_profile("qwen", "base", 0.0)
    # 缓存路径（cached）与直算路径结果不一致 → 触发失败分支
    with mock.patch.object(eng.synth, "_cached", return_value=p1 + 1e-3):
        with pytest.raises(ReproducibilityError):
            eng.validate_reproducibility()


def test_sft_invariance_failure_raises(cfg, synth):
    """第 3 步失败路径：SFT 微扰后相关性跌破阈值 → SFTInvarianceError。"""
    eng = verify.ValidatorEngine(cfg, synth)
    with mock.patch.object(eng.synth, "sft_perturb",
                           return_value=np.full(cfg.N_LAYERS_QWEN, 0.99)):
        with pytest.raises(SFTInvarianceError):
            eng.validate_sft_invariance()


def test_family_separation_failure_raises(cfg, synth):
    """第 4 步失败路径：跨家族距离塌缩 → FamilySeparationError。"""
    eng = verify.ValidatorEngine(cfg, synth)
    eng.validate_sft_invariance()
    with mock.patch.object(eng.synth, "interp_to_grid",
                           return_value=np.array(eng.g_base)):
        with pytest.raises(FamilySeparationError):
            eng.validate_family_separation()


def test_traceability_failure_raises(cfg, synth):
    """第 5 步失败路径：参考簇被篡改 → 溯源准确率 < 100% → TraceabilityError。"""
    eng = verify.ValidatorEngine(cfg, synth)
    eng.validate_sft_invariance()
    eng.validate_family_separation()
    eng.ref_q = np.array(eng.ref_g)  # Qwen 参考簇移至 GPT-2 处 → Qwen 样本全误判
    with pytest.raises(TraceabilityError):
        eng.validate_traceability()


# ---------------------------------------------------------------- benchmark
def _bench_mean(benchmark) -> float:
    """pytest-benchmark 统计均值（兼容 4.x stats 直返 / 5.x Metadata.stats；
    --benchmark-disable 时未测量 → 返回 0，断言自然通过）。"""
    stats = benchmark.stats
    if stats is None:  # --benchmark-disable：不做测量
        return 0.0
    return float(stats.mean if hasattr(stats, "mean") else stats.stats.mean)


@pytest.mark.benchmark
def bench_synth(benchmark, cfg, synth):
    """单次 synth_beta_profile 平均耗时 < 0.1ms。"""
    result = benchmark(synth.synth_beta_profile, "qwen", "base", 0.0)
    assert result.shape == (cfg.N_LAYERS_QWEN,)
    assert _bench_mean(benchmark) < 0.0001  # 0.1ms


@pytest.mark.benchmark
def bench_interp(benchmark, cfg, synth):
    """单次 interp_to_grid 平均耗时 < 0.1ms。"""
    prof = synth.synth_beta_profile("gpt2", "base", 0.0)
    result = benchmark(synth.interp_to_grid, prof, cfg.GRID)
    assert result.shape == (cfg.GRID,)
    assert _bench_mean(benchmark) < 0.0001  # 0.1ms


# ---------------------------------------------------------------- 环境变量覆盖
def test_env_override_corr_min(monkeypatch):
    """环境变量 AIQ_A01_CORR_MIN > 集中配置（0.99）。"""
    monkeypatch.setenv("AIQ_A01_CORR_MIN", "0.95")
    cfg = verify.ConfigFactory().build()
    assert cfg.CORR_MIN == 0.95
    # 其余字段不受影响
    assert cfg.N_LAYERS_QWEN == 24 and cfg.SEP_RATIO_MIN == 2.0
    monkeypatch.delenv("AIQ_A01_CORR_MIN", raising=False)
    cfg2 = verify.ConfigFactory().build()
    assert cfg2.CORR_MIN == 0.99


# ---------------------------------------------------------------- 全流程编排
def test_main_pipeline(tmp_path, capsys):
    """main() 全流程：JSON/HTML 报告落盘 + 退出码 0（含 --profile 分支）。"""
    code = verify.main(["--json", "--html", "--out-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "a01_verify_report.json").is_file()
    assert (tmp_path / "a01_verify_report.html").is_file()
    out = capsys.readouterr().out
    assert "全部通过" in out
    data = json.loads((tmp_path / "a01_verify_report.json").read_text(encoding="utf-8"))
    assert data["passed"] is True and data["n_items"] == 6


def test_main_profile(tmp_path):
    """main() --profile 分支：cProfile 剖析文件落盘。"""
    code = verify.main(["--profile", "--out-dir", str(tmp_path)])
    assert code == 0
    prof_dir = tmp_path / "logs" / "profile"
    assert prof_dir.is_dir()
    assert any(p.suffix == ".prof" for p in prof_dir.iterdir())
    assert any(p.suffix == ".txt" for p in prof_dir.iterdir())
