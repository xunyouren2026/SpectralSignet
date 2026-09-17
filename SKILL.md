---
name: "aiq-geometric-forensics"
description: "AI 几何指纹诊断：对模型输出健康画像（AIQ+Gamma+曲率）、家族溯源判定、基准对比与溯源鲁棒性评估、家族族谱，支持本地真实测量与 HTML/JSON 报告导出。当用户要对模型做身份鉴定、健康体检、可压缩性评估、溯源鲁棒性/微调前后对比、家族归属或族谱分析时使用。"
---

# aiq-geometric-forensics — AI 几何指纹诊断包

## 定位

> 输入：任意模型路径 / 名称；输出：几何健康画像 + 家族溯源判定 + 与基准对比 + 可导出的 HTML/JSON 报告。
> 以**真实实测**为准（Qwen / Llama / BLOOM 三家族 5 个 0.5–1.5B 模型标定），参数实时获取，拒绝文档水分值与硬编码。

本包把 59 参数验证体系的真实测量引擎封装为三大函数式能力 + 统一直觉 CLI，可从 `modules` 直接导入。

## 何时使用

- 用户要对一个模型做**几何健康体检**（AIQ 智能商 / Gamma 剖面 / 退化-过深-短链判定）
- 用户要判断一个模型**是否属于某家族**，或验证**微调（SFT）后指纹是否稳定**
- 用户要评估模型**可压缩性**（Gamma 谱 → KV/权重剪枝建议）
- 用户要**对比待检模型与已标定基准**（家族归属 + 量化偏差）
- 用户要对本地模型做**真实前向测量**（需 torch/transformers）
- 用户要把结果导出为**自包含 HTML 报告**或结构化 JSON

## 快速使用（Python API）

```python
# 在本包根目录下运行（modules 为包内模块）
from modules import diagnose, format_health, trace, format_trace, compare, format_compare

print(format_health(diagnose("Qwen2.5-0.5B-Instruct")))          # 健康画像
print(format_trace(trace("Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct"))) # 家族溯源
print(format_compare(compare("Qwen2.5-0.5B", "bloom-560m")))      # 基准对比
```

一键跑通 4 大用例：

```bash
python aiq_forensics_demo.py
```

## 命令行（CLI）

一个入口覆盖 6 大动作，支持 JSON / HTML 报告导出：

```bash
python -m modules.cli list                       # 列出基准库
python -m modules.cli health NAME --json r.json --html r.html
python -m modules.cli trace TARGET REF --html r.html
python -m modules.cli compare TARGET REF
python -m modules.cli stability [TARGET] --html r.html   # 溯源鲁棒性（扰动下指纹存活）
python -m modules.cli family --html r.html               # 家族族谱（指纹亲缘聚类）
python -m modules.cli verify [PARAM]                     # 参数验证体系（59 参数，缺省批量全部）
python -m modules.cli verify --limit 5                   # 限数快速冒烟
python -m modules.cli report --json all.json --html all.html   # 跨家族综合报告
python -m modules.cli measure path/to/model --out measure.json --dtype bfloat16
python -m modules.cli selfcheck --json check.json          # 生产完整性门禁（发布前自检）
```

导出物为**自包含 HTML**（内联 CSS + SVG，零外部依赖，含中文字体），并带**证据分级徽标**。

## 三种数据来源（自动降级 + 证据分级）

1. **真实测量（inline，绿）**：把 `measure()` 结果的 dict 传入，最高优先。
2. **基准存档（baseline，蓝）**：`baselines/<模型名>.json`，本包自带 5 个跨家族真实实测。
3. **审计兜底（fallback，琥珀）**：内置结构占位，仅当前两者缺失时使用并显著标注。

每个报告都标注 `数据源:` 与徽标，绝不把兜底伪装成实测。

## 对本地模型做真实测量

```python
from modules import measure, diagnose, format_health
m = measure("path/to/local-model", dtype="bfloat16")   # 真实前向（需 torch + transformers）
print(format_health(diagnose("my-model", measurement=m)))
```

把新家族加入基准库可扩展指纹识别范围（见 `docs/使用指南.md` 方式 C）。

## 基准库（跨家族真实实测）

| 模型 | 家族 | 层数 | Γ 谱集中 | AIQ |
|------|------|------|----------|-----|
| Qwen2.5-0.5B | Qwen(base) | 24 | 0.5138 | 54.15 |
| Qwen2.5-0.5B-Instruct | Qwen(SFT) | 24 | 0.5122 | 54.60 |
| Qwen2.5-1.5B-Instruct | Qwen(扩参) | 28 | 0.4325 | 58.02 |
| TinyLlama-1.1B-Chat | Llama | 22 | 0.4950 | 52.67 |
| BLOOM-560M | BLOOM | 24 | 0.3700 | 55.46 |

> AIQ 为 2.3 版公式（f3=1/(1+Hmed) 修复非退化、f4=全投影谱集中独立于 f1）后实时复算值，与存档一致。
> **DEFF 平台口径**：5 模型 `DEFF_plat` 均高于理论锚点 π/2，偏差 **+0.74%~+2.32%（均值 +1.38%±0.60%）**——本包按"区间 + 误差带"表述（非单一"锁定 π/2"点值），并提示存在系统性正偏；健康判定的"家族标定带"由 `calibrate_deff_band()` 实时取自基准观测支撑集 `[1.5825, 1.6072]`。
> **因子判别力**：f2(DEFF稳定) 因 DEFF_cv≈0.19 在 5 模型近乎恒定，属"基线项"，对排序零贡献，不参与模型区分；判别详情见跨家族报告"因子判别度诊断"表。可压缩性分级阈值 0.35/0.45 为**推测级（经验），非实测**。

## 设计原则（专业完整体 8 条）

1. 统一架构：一个可导入包 `modules/`，公开入口收敛在 `modules/__init__.py`
2. 标准接口：`diagnose()/trace()/compare()/measure()` 返回结构化 dict，可序列化
3. 集中数据层：真实 > 存档 > 兜底，杜绝散落硬编码
4. **实时获取**：DEFF_cv / 层数 / 曲率对 / 架构字段全部从真实测量与 config 实时解析，无 Qwen 私有默认值
5. 零硬编码路径 + `sanitize_path()` 剔除用户绝对路径，不泄漏隐私
6. 统一 CLI + 自包含 HTML/JSON 报告导出，含证据分级徽标
7. 工程规范：类型注解 / 边界防御（无实测投影跳过）/ 退出码 / 断言带消息
8. 可验证：`tests/run_all_tests.py`(33) + `tests/verify_derivations.py`(56) + `tests/test_robustness.py`(14) + `tests/test_selfcheck.py`(73) + `tests/test_extras.py`(13) 一键回归全绿（**189 断言**）；`ruff check` 与 `mypy` 静态检查全绿；CI（`.github/workflows/ci.yml`）与 pre-commit 门禁就绪；发布前可跑 `python -m modules.cli selfcheck` 做生产完整性自检

## 质量验证

```bash
python tests/run_all_tests.py          # 期望：PASS=33 FAIL=0
python tests/verify_derivations.py     # 期望：PASS=56 FAIL=0
python tests/test_robustness.py        # 期望：PASS=14 FAIL=0
python tests/test_selfcheck.py         # 期望：PASS=73 FAIL=0（生产完整性门禁）
python tests/test_extras.py            # 期望：PASS=13 FAIL=0（stability/family 追加回归）
python -m modules.cli selfcheck        # 发布前自检：版本/存档/漂移/证据纪律全绿
```

## 结构

```
modules/         包本体（__init__/data/health/forensics/stability/family/compare/harness/report/cli/selfcheck）
params/          参数验证体系（59 参数 verify.py + README + 统一数据层，A-T 九组）
baselines/       跨家族真实实测存档（5 个模型）
tests/           零依赖验证套件（33 + 56 + 14 + 73 + 13 断言）
docs/使用指南.md  详细用法与输出解读
aiq_forensics_demo.py  6 大用例入口
CHANGELOG.md     版本变更记录
```

详细说明见 `docs/使用指南.md`；版本沿革见 `CHANGELOG.md`。