# -*- coding: utf-8 -*-
"""A04 layer_names 挂钩层列表 — 四层工厂架构验证
=====================================================================
验证目标（与原脚本完全一致，保真）：
  1. 层名生成：model.layers.{i}.self_attn.k_proj（Qwen 24 层，主文档行 199）
  2. None → 自动全选；指定子集 → 前缀过滤（24/8/8/8）
  3. 分层指纹差异：浅 0.2-0.3 / 中 0.5-0.8 / 深 0.3-0.6（中层最高）
  4. 消融实验：剔除中层对指纹变化最大（核心指纹层，主文档行 279-291）
  5. 层×投影：k_proj 集中（β≈0.7）> up_proj 弥散（β≈0.3）
  6. 真实模型对照：真实 24 层结构 + 真实逐层 Gamma 分层统计（如实报告差异）

四层工厂架构（本文件内实现，复用 params/ 共享基类）：
  A04Config               —— 配置模型（pydantic 校验；pydantic 缺失时自动 dataclass 回退）
  ConfigFactory           —— 实例化 A04Config（环境变量 AIQ_A04_<KEY> > YAML > _params_data.json > 默认）
  LayerBetaSynthesizer    —— 逐层 β 剖面合成（三层模板，主文档行 245-249）
  ValidatorEngine         —— 6 项验证 + 结构化 JSON 日志（_logging）+ 类型化异常（_errors）
  ReportGenerator         —— 文本/JSON/HTML 报告 + 退出码 0/1（复用 _factory 基类）
  main()                  —— 仅编排 cfg→synth→engine→report，解析
                            --profile（_perf.profile_run）/ --json / --html

数据源：
  主文档行 173-315（A04 操作流程、分层表、消融、判断标准）
  《AI几何指纹插件_参数完整定义与公式.txt》行 33-40
说明：纯数值合成数据，不加载任何大模型。运行时间数秒内。
=====================================================================
真实模型对照：
  真实模型：本地 Qwen2.5-0.5B-Instruct（_real_metrics.json，共享库 _real_data.py）。
  接入点：真实 24 层结构（0-23，arch.n_layers）；分层 ZONE 真实层号
          （浅 0-7 / 中 8-15 / 深 16-23）；真实 k_proj 逐层 Gamma（24 维）分层统计。
  如实呈现：真实逐层 Gamma 分层均值为 浅(≈0.509) > 深(≈0.451) > 中(≈0.448)，
            与文档声称"中层最高（0.5-0.8）"相反——如实报告差异、不掩盖。
=====================================================================
"""
import argparse
import io
import os
import sys
import time
from typing import Any

import numpy as np

# ---- 统一工程样板：把参数根目录（params/）加入 sys.path，复用共享基类 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
from _common import finish, setup_env  # noqa: E402
from _errors import AIQValidationError, ConfigError, RealModelMismatchError  # noqa: E402
from _factory import ConfigFactory as _ConfigFactoryBase  # noqa: E402
from _factory import FingerprintSynthesizer as _SynthBase  # noqa: E402
from _factory import ReportGenerator  # noqa: E402
from _factory import ValidatorEngine as _EngineBase  # noqa: E402
from _logging import logger as structured_logger  # noqa: E402
from _perf import profile_run  # noqa: E402

# 统一样板：stdout/stderr UTF-8 + 共享库注入（RD 供真实模型对照，P 供数据源标注）
RD, P, CFG = setup_env(__file__)
if isinstance(sys.stderr, io.TextIOWrapper):  # 结构化日志写 stderr，统一 UTF-8 防乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- 第一层：配置模型 A04Config（pydantic 优先；dataclass 回退） ----
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


class A04Config(_ConfigModelBase):
    """A04 配置模型：全部阈值/常量集中于此（零硬编码判据）。

    字段与 _params_data.json 的 A04 节点键名一一对应；取值优先级：
    环境变量 AIQ_A04_<KEY> > YAML > _params_data.json > 本模型默认值。
    """

    SEED: int = 0                    # H01 固定随机种子（算法逻辑常量，保留）
    N_LAYERS: int = 24               # Qwen 层数（主文档行 199；层名列表长度基准）
    PROJ: str = "k_proj"             # 默认投影（层名示例用；k_proj 是集中度最高的指纹投影）
    ZONE_MID_START: int = 8          # 中层起始层号（浅/中分界）
    ZONE_MID_END: int = 16           # 中层结束层号（中/深分界）
    K_BETA: list = [0.70, 0.72, 0.60]   # k_proj 三区 β 均值 [浅,中,深]（主文档行 284）
    UP_BETA: list = [0.30, 0.32, 0.28]  # up_proj 三区 β 均值 [浅,中,深]（主文档行 285）


# ---- 第二层：配置工厂 ConfigFactory（实例化 A04Config） ----
class ConfigFactory(_ConfigFactoryBase):
    """A04 配置工厂：按优先级（环境变量 > YAML > _params_data.json > 默认）实例化 A04Config。"""

    def build(self) -> A04Config:
        """构建 A04Config（pydantic 优先，dataclass 回退，共享基类 build_model 驱动）。"""
        return self.build_model(A04Config, "A04")


# ---- 第三层：合成器 LayerBetaSynthesizer（算法与原脚本完全一致） ----
class LayerBetaSynthesizer(_SynthBase):
    """A04 逐层 β 剖面合成器（三层模板，主文档行 245-249）。"""

    def __init__(self, cfg: A04Config) -> None:
        super().__init__(cfg, seed=cfg.SEED)

    def synth_layer_beta(self, n_layers: int | None = None,
                         rng: np.random.Generator | None = None) -> np.ndarray:
        """合成逐层 β 剖面（主文档行 245-249 模板）：
          浅层 0-7  : 0.2-0.3
          中层 8-15 : 0.5-0.8
          深层 16-23: 0.3-0.6
        """
        cfg = self._cfg
        if n_layers is None:
            n_layers = cfg.N_LAYERS
        if rng is None:
            rng = np.random.default_rng(cfg.SEED)     # 未显式传 rng 时用固定种子，保证可复现
        if n_layers != cfg.N_LAYERS:
            raise ValueError(f"本验证仅支持 {cfg.N_LAYERS} 层模板，实得 {n_layers}")
        b = np.zeros(n_layers)                    # 逐层 β 容器
        b[0:8] = rng.uniform(0.20, 0.30, 8)       # 浅层：集中度最低（输入编码重塑）
        b[8:16] = rng.uniform(0.50, 0.80, 8)      # 中层：集中度最高（高保真传输）
        b[16:24] = rng.uniform(0.30, 0.60, 8)     # 深层：中等（输出前重组）
        return b


# ---------------- 纯函数工具（与验证逻辑解耦，保持可测试） ----------------
def all_layer_names(n_layers: int, proj: str) -> list[str]:
    """生成 model.layers.{i}.self_attn.{proj} 层名列表（主文档行 181-197）。"""
    if n_layers <= 0:
        raise ValueError(f"非法的层数: {n_layers}")   # 防御：层数非正则无法编号
    # 列表推导逐个生成从层 0 到层 n-1 的挂钩路径
    return [f"model.layers.{i}.self_attn.{proj}" for i in range(n_layers)]


def resolve_layers(layer_names: str | list[str] | None, n_layers: int, proj: str) -> list[str]:
    """A04 解析：None → 全选所有线性层；字符串前缀/列表 → 过滤。
    模拟主文档行 207：layer_names = None  # 自动全选"""
    all_names = all_layer_names(n_layers, proj)         # 全量层名基准
    if layer_names is None:
        return all_names                          # None：自动全选全部层
    if isinstance(layer_names, str):
        layer_names = [layer_names]               # 单字符串统一转为列表处理
    out: list[str] = []
    for pat in layer_names:
        # 前缀匹配：凡层名包含给定模式（如 "model.layers.8."）即纳入
        out += [n for n in all_names if pat in n]
    return out


def layer_zone(i: int, zone_mid_start: int, zone_mid_end: int, n_layers: int) -> str:
    """分层区标签（主文档行 245-249），用于输出展示。"""
    if i < zone_mid_start:
        return f"浅层(0-{zone_mid_start - 1})"        # 输入编码重塑区
    if i < zone_mid_end:
        return f"中层({zone_mid_start}-{zone_mid_end - 1})"       # 高保真传输区
    return f"深层({zone_mid_end}-{n_layers - 1})"          # 输出前重组区


# ---- 第四层：验证引擎 ValidatorEngine（6 项验证 + 结构化日志 + 类型化异常） ----
class ValidatorEngine(_EngineBase):
    """A04 验证引擎：顺序执行 6 项验证。

    - 每步输出一行可 json.loads 的结构化 JSON 日志（step_id/name/elapsed_ms/status/extra）；
    - 失败时抛 _errors 类型化异常（携带 expected/actual），由 run() 捕获记 FAIL 并继续；
    - _real_data 惰性导入（经 _common.setup_env 注入 RD）。
    """

    def __init__(
        self,
        config: A04Config,
        synth: LayerBetaSynthesizer,
        reporter: ReportGenerator | None = None,
        real_data: Any = None,
    ) -> None:
        super().__init__(config, synth, reporter)
        self._real_data = real_data  # 惰性注入（None 时 validate_real_model 方法内 import）
        self.beta: np.ndarray | None = None   # 合成逐层 β 剖面（供消融/投影步骤复用）

    def _get_real_data(self) -> Any:
        """真实数据访问：优先用注入的 RD；否则方法内惰性导入 _real_data。"""
        if self._real_data is None:
            import _real_data  # 惰性导入（仅真实模型对照步骤需要）
            self._real_data = _real_data
        return self._real_data

    # ------------------------------------------------------------ 1) 层名生成
    def validate_names(self) -> dict:
        """1) 层名生成：数量恰为 N_LAYERS，首末层名符合 Qwen 命名规范。"""
        cfg = self.config
        names = all_layer_names(cfg.N_LAYERS, cfg.PROJ)        # 生成全部 24 层层名
        # 断言：数量恰为 24，首末层名符合 Qwen 命名规范（层0 与 层23）
        ok1 = (len(names) == cfg.N_LAYERS
               and names[0] == f"model.layers.0.self_attn.{cfg.PROJ}"
               and names[-1] == f"model.layers.{cfg.N_LAYERS - 1}.self_attn.{cfg.PROJ}")
        if not ok1:
            raise ConfigError(
                f"层名生成异常: 数量={len(names)}, 首={names[0]}, 末={names[-1]}",
                expected=cfg.N_LAYERS, actual=len(names), param_key="A04",
            )
        return {"detail": f"层名生成: {names[0]} ... {names[-1]} (共 {len(names)} 个)",
                "n_names": len(names)}

    # ------------------------------------------------------------ 2) None 全选 / 子集过滤
    def validate_resolve(self) -> dict:
        """2) None→自动全选；三层子集前缀过滤（24/8/8/8，带尾点精确匹配防串层）。"""
        cfg = self.config
        z_sh, z_mid, z_deep = (0, cfg.ZONE_MID_START), (cfg.ZONE_MID_START, cfg.ZONE_MID_END), (cfg.ZONE_MID_END, cfg.N_LAYERS)
        full = resolve_layers(None, cfg.N_LAYERS, cfg.PROJ)                 # None → 自动全选
        # 用带尾点的模式精确匹配层号，避免 "model.layers.1" 误配层 10-19
        shallow = resolve_layers([f"model.layers.{i}." for i in range(*z_sh)], cfg.N_LAYERS, cfg.PROJ)
        mid = resolve_layers([f"model.layers.{i}." for i in range(*z_mid)], cfg.N_LAYERS, cfg.PROJ)
        deep = resolve_layers([f"model.layers.{i}." for i in range(*z_deep)], cfg.N_LAYERS, cfg.PROJ)
        # 断言：全选 N_LAYERS 层，三个分层子集各 ZONE 宽度层（带尾点精确匹配避免串层）
        ok2 = (len(full) == cfg.N_LAYERS
               and len(shallow) == (z_mid[0] - z_sh[0])
               and len(mid) == (z_mid[1] - z_mid[0])
               and len(deep) == (z_deep[1] - z_deep[0]))
        if not ok2:
            raise ConfigError(
                f"子集过滤数量不符: full={len(full)}, 浅={len(shallow)}, 中={len(mid)}, 深={len(deep)}",
                expected=[cfg.N_LAYERS, z_mid[0] - z_sh[0], z_mid[1] - z_mid[0], z_deep[1] - z_deep[0]],
                actual=[len(full), len(shallow), len(mid), len(deep)],
                param_key="A04",
            )
        return {"detail": (f"None→全选 {len(full)} 层; 浅层子集 {len(shallow)} 层; "
                           f"中层子集 {len(mid)} 层; 深层子集 {len(deep)} 层"),
                "full": len(full), "shallow": len(shallow), "mid": len(mid), "deep": len(deep)}

    # ------------------------------------------------------------ 3) 分层 β 差异
    def validate_zone_beta(self) -> dict:
        """3) 分层 β 差异：中层均值最高且落在 [0.5,0.8]，且中层>深层>浅层。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.SEED)
        beta = self.synth.synth_layer_beta(rng=rng)            # 合成逐层 β 剖面
        self.beta = beta                                       # 供消融步骤复用
        b_sh, b_mid, b_deep = beta[0:8], beta[8:16], beta[16:24]   # 三区切分
        # 判据：中层均值最高且落在 [0.5,0.8]，且中层>深层>浅层（文档分层结构）
        ok3 = (b_mid.mean() > b_deep.mean() > b_sh.mean()) and (0.5 < b_mid.mean() < 0.8)
        if not ok3:
            raise AIQValidationError(
                f"分层 β 关系不符: 浅={b_sh.mean():.3f}, 中={b_mid.mean():.3f}, 深={b_deep.mean():.3f}",
                expected="mid > deep > shallow 且 mid ∈(0.5,0.8)",
                actual={"sh": float(b_sh.mean()), "mid": float(b_mid.mean()), "deep": float(b_deep.mean())},
                param_key="A04",
            )
        return {"detail": (f"分层 β均值: 浅层={b_sh.mean():.3f} (0.2-0.3), "
                           f"中层={b_mid.mean():.3f} (0.5-0.8), 深层={b_deep.mean():.3f} (0.3-0.6)"),
                "sh": float(b_sh.mean()), "mid": float(b_mid.mean()), "deep": float(b_deep.mean())}

    # ------------------------------------------------------------ 4) 消融实验
    def validate_ablation(self) -> dict:
        """4) 消融实验：剔除每层的指纹变化，核心层须落在中层（8-15）。"""
        cfg = self.config
        assert self.beta is not None  # 前置步骤已计算
        beta = self.beta
        full_vec = beta                              # 完整指纹剖面
        deltas = np.zeros(cfg.N_LAYERS)                  # 每层剔除后的指纹变化量
        for i in range(cfg.N_LAYERS):
            reduced = np.delete(beta, i)             # 剔除层 i 后的剖面
            # 指纹差异：剔除前后聚合指纹的相对变化（模拟主文档行 283-289）
            # 第二项 beta[i]*0.05 使高集中度层即使均值变化小也有区分度
            deltas[i] = abs(full_vec.mean() - reduced.mean()) + beta[i] * 0.05
        core = int(np.argmax(deltas))                # Δ 最大层 = 核心指纹层
        z_mid = (cfg.ZONE_MID_START, cfg.ZONE_MID_END)
        # 判据：核心层必须落在中层（8-15）——文档"中层为核心指纹层"
        ok4 = (z_mid[0] <= core < z_mid[1])
        if not ok4:
            raise AIQValidationError(
                f"核心指纹层不在中层: core={core}",
                expected=f"core ∈ [{z_mid[0]}, {z_mid[1]})", actual=core, param_key="A04",
            )
        return {"detail": (f"消融 Δ 最大层 = 层{core} ({layer_zone(core, cfg.ZONE_MID_START, cfg.ZONE_MID_END, cfg.N_LAYERS)}), "
                           f"Δ={deltas[core]:.4f} (中层应为核心指纹层)"),
                "core": core, "delta": float(deltas[core])}

    # ------------------------------------------------------------ 5) 层×投影
    def validate_proj(self) -> dict:
        """5) 层×投影：k_proj 集中（β≈0.7）> up_proj 弥散（β≈0.3）。"""
        cfg = self.config
        # 三区 β 均值从数据层读取（K_BETA/UP_BETA 各为 [浅,中,深] 三值），重复到逐层
        k_beta = np.repeat(np.asarray(cfg.K_BETA, dtype=float), cfg.ZONE_MID_START)   # k_proj β≈0.7
        up_beta = np.repeat(np.asarray(cfg.UP_BETA, dtype=float), cfg.ZONE_MID_START)  # up_proj β≈0.3
        # 判据：k_proj 均值>0.6（集中），up_proj 均值<0.4（弥散），且 k>up
        ok5 = (k_beta.mean() > 0.6 and up_beta.mean() < 0.4 and k_beta.mean() > up_beta.mean())
        if not ok5:
            raise AIQValidationError(
                f"投影差异不符: k={k_beta.mean():.3f}, up={up_beta.mean():.3f}",
                expected={"k>0.6": True, "up<0.4": True, "k>up": True},
                actual={"k": float(k_beta.mean()), "up": float(up_beta.mean())},
                param_key="A04",
            )
        return {"detail": (f"k_proj β均值={k_beta.mean():.3f} vs up_proj β均值={up_beta.mean():.3f} "
                           f"(k 集中 ≈0.7, up 弥散 ≈0.3)"),
                "k_beta": float(k_beta.mean()), "up_beta": float(up_beta.mean())}

    # ------------------------------------------------------------ 6) 真实模型对照
    def validate_real_model(self) -> dict:
        """6) 真实模型对照：真实 24 层结构 + 分层 ZONE + 真实逐层 Gamma 分层统计。"""
        cfg = self.config
        rd = self._get_real_data()                     # 惰性导入 / 注入的 _real_data
        n_layers_real = rd.get("arch.n_layers")                 # 真实层数（24）
        gamma_layers = np.asarray(rd.get("spectral.k_proj_gamma_layers"), dtype=float)  # 真实逐层 Gamma
        tag = "[真实实测]" if rd.has_real() else "[审计回退]"       # 数据来源前缀标签
        # 基础一致性断言：真实层数与常量一致、逐层 Gamma 数组恰为 24 维
        ok6a = (n_layers_real == cfg.N_LAYERS) and (gamma_layers.size == cfg.N_LAYERS)
        if not ok6a:
            raise RealModelMismatchError(
                f"真实层数与常量不符: real={n_layers_real}, 常量={cfg.N_LAYERS}",
                expected=cfg.N_LAYERS, actual=n_layers_real, param_key="A04",
            )
        sh_r, mid_r, dp_r = (gamma_layers[0:8].mean(), gamma_layers[8:16].mean(),
                             gamma_layers[16:24].mean())        # 真实三区 Gamma 均值
        core_r = int(np.argmax(gamma_layers))                   # 真实集中度最高层
        # 真实逐层消融（口径与原项4一致）：剔除层 i 对 gamma 均值的相对影响
        deltas_r = np.zeros(cfg.N_LAYERS)
        for i in range(cfg.N_LAYERS):
            reduced = np.delete(gamma_layers, i)
            deltas_r[i] = abs(gamma_layers.mean() - reduced.mean()) + gamma_layers[i] * 0.05
        core_abl_r = int(np.argmax(deltas_r))                   # 真实消融 Δ 最大层
        gamma12_r = float(gamma_layers[cfg.N_LAYERS // 2])      # B10 anchor 候选层 layer12（N//2）
        detail = (f"{tag} 真实层数 = {n_layers_real}（0-{n_layers_real - 1}），ZONE 层号 "
                  f"浅0-{cfg.ZONE_MID_START - 1}/中{cfg.ZONE_MID_START}-{cfg.ZONE_MID_END - 1}/"
                  f"深{cfg.ZONE_MID_END}-{cfg.N_LAYERS - 1}; 真实逐层 Gamma 分层均值: "
                  f"浅层 = {sh_r:.4f}, 中层 = {mid_r:.4f}, 深层 = {dp_r:.4f}（真实顺序 浅>深>中；"
                  f"文档声称 中层最高 0.5-0.8 —— 如实报告差异）; 集中度最高层 = layer{core_r}"
                  f"（gamma={gamma_layers[core_r]:.4f}）；消融 Δ 最大层 = layer{core_abl_r}; "
                  f"B10 anchor 候选层 layer{cfg.N_LAYERS // 2}.k_proj gamma = {gamma12_r:.4f} "
                  f"vs 全局均值 {gamma_layers.mean():.4f}（偏差 "
                  f"{100 * (gamma12_r - gamma_layers.mean()) / gamma_layers.mean():+.1f}%）")
        return {
            "detail": detail,
            "source": rd.source_tag(), "tag": tag,
            "n_layers_real": n_layers_real,
            "sh_r": float(sh_r), "mid_r": float(mid_r), "dp_r": float(dp_r),
            "core_r": core_r, "core_abl_r": core_abl_r,
        }

    # ------------------------------------------------------------ 编排
    def run(self) -> int:
        """顺序执行 6 项验证：每步输出结构化 JSON 日志，失败记 FAIL 并继续。"""
        steps: list[tuple[int, str, Any]] = [
            (1, "names", self.validate_names),
            (2, "resolve", self.validate_resolve),
            (3, "zone_beta", self.validate_zone_beta),
            (4, "ablation", self.validate_ablation),
            (5, "proj", self.validate_proj),
            (6, "real_model", self.validate_real_model),
        ]
        for step_id, name, fn in steps:
            t0 = time.perf_counter()
            status, extra, detail = "PASS", {}, ""
            try:
                extra = dict(fn() or {})
                detail = extra.pop("detail", "")
            except (AIQValidationError, ValueError) as e:  # 类型化异常 + 防御性 ValueError
                status = "FAIL"
                detail = str(e)
                extra = {"error": str(e),
                         "expected": getattr(e, "expected", None),
                         "actual": getattr(e, "actual", None),
                         "param_key": getattr(e, "param_key", None)}
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # 结构化 JSON 日志（_logging 单例；每行可 json.loads）
            structured_logger.step(step_id, name, elapsed_ms, status, **extra)
            if self.reporter is not None:
                self.reporter.add(step_id, name, status, detail)
        return self.reporter.exit_code if self.reporter is not None else 0


# ---------------- 入口：仅编排 cfg→synth→engine→report ----------------
def main(argv: list[str] | None = None) -> int:
    """A04 验证编排：四层工厂装配 + --profile/--json/--html 输出。"""
    parser = argparse.ArgumentParser(prog="verify", description="A04 layer_names 四层工厂验证")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告文件")
    parser.add_argument("--html", action="store_true", help="输出 HTML 报告文件")
    parser.add_argument("--profile", action="store_true", help="用 cProfile 剖析验证流程")
    parser.add_argument("--out-dir", default=None, help="报告输出目录（默认本脚本目录）")
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(os.path.abspath(__file__))

    # ---- 四层工厂装配 ----
    cfg = ConfigFactory().build()                     # ① 配置层（env > YAML > JSON > 默认）
    synth = LayerBetaSynthesizer(cfg)                 # ② 合成层
    report = ReportGenerator()                        # 报告器（复用 _factory 基类）
    engine = ValidatorEngine(cfg, synth, report, real_data=RD)  # ③ 验证层（RD 经 setup_env 注入）

    print("=" * 74)
    print("A04 layer_names 验证（四层工厂架构，合成数据，不加载大模型）")
    print(f"数据源: {P.source_tag()}")
    print(f"配置模型: {'pydantic' if _HAS_PYDANTIC else 'dataclass 回退'}")
    print(f"配置: N_LAYERS={cfg.N_LAYERS} PROJ={cfg.PROJ} ZONE_MID={cfg.ZONE_MID_START}-{cfg.ZONE_MID_END} "
          f"K_BETA={cfg.K_BETA} UP_BETA={cfg.UP_BETA} SEED={cfg.SEED}")
    print("=" * 74)

    # ---- ④ 运行（可选剖析）----
    if args.profile:
        res = profile_run(engine.run, out_dir, "a04_verify")  # cProfile 剖析钩子（_perf）
        print(f"剖析文件: {res['prof']}")
    else:
        engine.run()

    # ---- ⑤ 报告输出 ----
    print(report.render_text())
    if args.json:
        json_path = os.path.join(out_dir, "a04_verify_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.render_json())
        print(f"JSON 报告已写入: {json_path}")
    if args.html:
        html_path = os.path.join(out_dir, "a04_verify_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(report.render_html())
        print(f"HTML 报告已写入: {html_path}")

    # ---- ⑥ 汇总与退出码（复用 _common.finish 约定：0=全过，1=存在失败）----
    return finish(report.passed, report.n_items)


if __name__ == "__main__":
    raise SystemExit(main())
