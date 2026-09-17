# -*- coding: utf-8 -*-
"""
AIQ 参数验证 — 统一异常层次（_errors）
=====================================================================
本模块定义整个 aiq-geometric-forensics 插件共享的异常体系，供
_factory / _perf / 各参数 verify.py 统一使用。所有异常均以
`AIQValidationError` 为根，携带 expected / actual / param_key 三个
结构化字段，便于上层报告生成器与日志系统自动提取"期望值 vs 实测值"。

用法：
  raise ReproducibilityError(
      "再现性偏差超阈值", expected=1e-3, actual=2.1e-3, param_key="D10")
  raise ConfigError("缺少配置段")          # expected/actual 可选

设计约定：
  - 根异常 `__init__` 将 message/expected/actual/param_key 同时存入
    属性与 `args`（保持 Exception 标准行为，pickle/打印兼容）；
  - 子类一律不重写 `__init__`，天然继承全字段签名；
  - ConfigError / SynthesisError 的 expected/actual 通常不参与数值
    对照，默认 None 即可（"可选"语义）。
"""
from __future__ import annotations


class AIQValidationError(Exception):
    """AIQ 验证体系根异常：携带期望值/实测值/参数键结构化信息。"""

    def __init__(
        self,
        message: str,
        expected=None,
        actual=None,
        param_key: str | None = None,
    ) -> None:
        """构造根异常。

        参数：
          message:   错误描述（必填）
          expected:  期望值（阈值/参考值），用于报告对照；默认 None
          actual:    实测值，用于报告对照；默认 None
          param_key: 关联的参数标识，如 "D10" / "T01.PLATFORM"；默认 None
        """
        self.message = message
        self.expected = expected
        self.actual = actual
        self.param_key = param_key
        # 写入 args：保持 Exception 标准行为（str(e)/pickle 兼容）
        super().__init__(message, expected, actual, param_key)

    def __str__(self) -> str:
        """友好描述：含参数键与期望/实测对照（None 字段省略）。"""
        parts = [self.message]
        if self.param_key is not None:
            parts.append(f"param_key={self.param_key!r}")
        if self.expected is not None:
            parts.append(f"expected={self.expected!r}")
        if self.actual is not None:
            parts.append(f"actual={self.actual!r}")
        return " ".join(parts)


class ReproducibilityError(AIQValidationError):
    """再现性（可复现性）验证失败：同种子/同参数多次结果应一致。"""


class FamilySeparationError(AIQValidationError):
    """家族分离验证失败：不同模型家族指纹应可区分。"""


class SFTInvarianceError(AIQValidationError):
    """SFT 不变性验证失败：微调前后应保持稳定的几何指纹特征。"""


class TraceabilityError(AIQValidationError):
    """溯源（家族归属判定）验证失败：样本无法正确归属到真实家族。"""


class RealModelMismatchError(AIQValidationError):
    """真实模型对照验证失败：合成指纹与真实测量存在显著偏差。"""


class ConfigError(AIQValidationError):
    """配置错误：参数缺失/类型非法/数据文件损坏等。

    注意：expected/actual 为可选字段——配置错误通常不涉及数值对照，
    仅在确有期望/实际值可对比时传入。
    """


class SynthesisError(AIQValidationError):
    """指纹合成错误：合成过程失败/输出非法。

    注意：expected/actual 为可选字段——合成错误通常只需 message，
    数值对照场景（如剖面长度不符）可传入 actual 辅助定位。
    """


if __name__ == "__main__":
    # 自检：验证字段存储、args 写入与子类继承
    e = ReproducibilityError(
        "再现性偏差超阈值", expected=1e-3, actual=2.1e-3, param_key="D10"
    )
    assert e.expected == 1e-3 and e.actual == 2.1e-3 and e.param_key == "D10"
    assert e.args == ("再现性偏差超阈值", 1e-3, 2.1e-3, "D10")
    assert isinstance(e, AIQValidationError)
    print("str:", e)
    print("args:", e.args)

    c = ConfigError("缺少配置段")          # expected/actual 可选
    assert c.expected is None and c.actual is None
    assert c.args == ("缺少配置段", None, None, None)
    print("str:", c)
    print("_errors 自检 PASS")
