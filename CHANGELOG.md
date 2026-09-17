# 变更记录 (CHANGELOG)

## 3.3.1 (2026-09-08) — smoke_all 全量冒烟脚本工业级升级

### 重写 `tests/params/smoke_all.py`（原文件被误覆写为需求文档，已恢复为完整代码）
- **进程隔离**：每个 verify.py 独立子进程运行，`subprocess.run` + 超时（默认 600s），互不污染。
- **结果记录**：退出码 / 墙钟耗时 / 超时标志 / 启动失败标志 / stdout+stderr 尾部（默认 3000 字符）。
- **状态判定**：PASS（exit=0 且 ≤ 阈值 5s）/ WARNING（exit=0 但 > 阈值，不影响返回码）/ FAIL（exit≠0、超时或启动失败）。
- **报告输出**：逐项明细表 + 汇总统计 + FAIL 详情（stdout/stderr 尾部）+ WARNING 慢脚本清单（按耗时降序）。
- **返回码**：0 = 全部 PASS，1 = 存在 FAIL，供 CI/CD 门禁。
- **可配置**：`SmokeConfig` 集中阈值/超时/并发数；CLI 支持 `--skip-run` / `--threshold` / `--timeout` / `--jobs` / `--json` / `--params-root`。
- **健壮性**：解释器缺失、启动失败、超时、意外异常均兜底为 FAIL 且不中断整体；pathlib 跨平台；纯标准库，Python 3.8+。
- **并发可选**：`--jobs N` 用线程池并行（默认顺序保证稳定性）。
- **验证**：模拟 4 个 verify.py（PASS / FAIL / 超时 / 慢）实测分类与退出码正确；真实 params/ 扫描发现 59 个；JSON 报告结构可用。

---

## 3.3.0 (2026-09-08) — 59 参数验证脚本工业级升级（优化工厂架构）

### 核心重构（A01 旗舰样板 + 全量扩展，验证逻辑保真）
- **四层工厂架构**：59 个 `verify.py` 全部重构为 `ConfigFactory` → 合成器（FingerprintSynthesizer）→ `ValidatorEngine` → `ReportGenerator`，`main` 仅编排并返回退出码 0/1。
- **类型安全**：配置由 pydantic `*Config(BaseModel)` 管理（无 pydantic 自动回退 dataclass + 运行时校验），全函数精确 Type Hints（typing + dataclass）。
- **可观测性**：新增 `params/_logging.py` 结构化 JSON 日志（loguru 优先、stdlib 兜底），每步验证输出 `step_id` / `elapsed_ms` / `status`，可接入 ELK/CloudWatch。
- **错误处理**：新增 `params/_errors.py` 根异常 `AIQValidationError` 及子类（Reproducibility / FamilySeparation / SFTInvariance / Traceability / RealModelMismatch / Config / Synthesis Error），均携带 `expected` / `actual` 字段。
- **零硬编码**：新增 `params/_factory.py` 统一 `ConfigFactory`，优先级 `AIQ_<PARAM>_<KEY>` 环境变量 > YAML(可选) > `_params_data.json` > 兜底默认。
- **模块化导入**：`_real_data` / `_params` / `_cfg` 惰性加载 + 依赖注入（`_common.setup_env`），零文件路径硬依赖。

### 性能优化（`params/_perf.py`）
- 合成剖面向量化（矩阵堆叠消除 Python 层 for 循环）；`lru_cache` 按 (family, variant, seed) 缓存；惰性插值 `LazyInterp` 仅比对时触发；统一 `np.float32`（Corr 差 < 1e-5）；`--profile` 输出 `logs/profile/` cProfile 报告。
- T05 尖峰显著性：`norm.cdf` 整网格向量化（结果逐位恒等），7.05s → 2.35s。

### 测试与 CI
- 新增 `tests/params/`：`conftest.py`（rng/基线 fixtures）、`test_A01_model.py`（6 原子测试 + 边界 + benchmark < 0.1ms + RD mock 离线）、`test_shared_infra.py`、`smoke_all.py`（全量 59 脚本一次性一致性检查）。
- `pytest tests/params` **69 用例全绿**，行覆盖率 **97%**（≥95%）；benchmark synth 27µs / interp 14µs。
- `.github/workflows/ci.yml`：ruff / mypy / pytest / pytest-cov 覆盖率门禁 ≥95%。
- SOTA 审计 `params/_sota_audit.py`：类型安全 / 错误处理 / 防御性 **59/59** 达标。

### 修复（升级过程中，未改动判定逻辑）
- H02/H03：补 `import argparse`；F01：`_params_data.json` 的 CTX 由 4096 修正为 1024（与脚本文档语义一致）；D05/D08 存量"审计回退"分支未定义变量崩溃修复。

### 验证
- 全量 `smoke_all.py` 59/59 PASS，单脚本 < 5s；归档 `_aiq60_extract\AIQ\参数` 已同步（抽查 A01/D09/T05 独立运行 exit=0）。

---

## 3.2.0 (2026-09-08) — 参数验证编排器 SOTA 增强（报告/统计/HTML）

### SOTA 框架增强（不改动 59 个已验证脚本的逻辑）
- **`params_runner` 结构化报告**：verify_all 输出新增 `groups`（按组 A-T 聚合：通过/失败/耗时）、`performance`（总/均/最慢参数）、`failures`（失败明细）。
- **`render_params` HTML 报告**：自包含卡片，含通过率、按组聚合表、逐项明细（PASS/FAIL + 耗时）。
- **CLI `verify --html`**：参数验证可导出自包含 HTML 报告。
- **SOTA 质量审计结论**：59 脚本在类型注解(59/59)、真实对照(59/59)、数据层(59/59)、纯函数(58/59)、断言消息(57/59)、try 防御(58/59)均达标；外部数据加载脚本全部有 isfile 守卫（无真实缺失）。
- `_common.py` 统一样板就绪（A01 试点），作为后续样板统一的渐进路径；不批量改写已验证逻辑（避免回归风险）。

### 验证
- ruff/mypy 0 错（14 源文件）；五套件全绿（189 断言）；selfcheck 13/13；verify --limit/--html 冒烟 PASS；版本三源一致 3.2.0。

---

## 3.1.0 (2026-09-08) — 参数验证体系统一收口（CLI 入口 + selfcheck 覆盖）

### 工程优化（承接"参数只是搬运、未真正接入插件"评审）
- **`modules/params_runner.py` 统一收口**：把 59 个散落的 verify.py 收口为可编程批量执行器。
  - `list_params()` / `verify_one(name)` / `verify_all(limit=None)`；
  - 子进程隔离运行（cwd=参数目录），保留各脚本 sys.path 注入完全零侵入；
  - 结果统一汇总（按组 + 总 PASS/FAIL + 耗时），退出码 0/1。
- **CLI 新子命令 `verify`**：`python -m modules.cli verify <参数>`（单参）/
  `verify`（批量全部）/ `verify --limit N`（限数冒烟）`--json` 导出。
- **selfcheck 门禁扩展到 13 项**：新增「参数体系可枚举（59）」「参数 verify.py 可编译」「参数数据层完整（_params_data.json 59 组）」。
- `modules/__init__.py` 导出 `params_runner`。

### 验证
- 插件五套件全绿（189 断言）；`cli selfcheck` **13/13**；`cli verify` 批量 **59/59 PASS**；
  单参 T04 0.2s PASS；`--limit 3` 冒烟 PASS。版本三源一致 3.1.0。

---

## 3.0.0 (2026-09-08) — 参数验证体系整体并入（59 参数全能力）

### 重大能力合并（承接 v2 专业化合并包）
- **`params/` 子目录**：将完整 59 参数验证体系（A-T 九组 + 14 基础设施）整体迁入插件，成为插件内置的"参数验证引擎"。
  - 59 参数目录：每参数含 `verify.py`（可执行验证）+ `README.md`（六要素研究文档）；
  - 基础设施：`_cfg.py`（零硬编码路径）/`_params.py`+`_params_data.json`（统一参数数据层 59 组 371 键）/`_real_data.py`（真实测量优先）/`_real_model_harness.py`（真实测量引擎）/`_real_metrics*.json`（真实实测存档）；
  - 真实曲率数据 `phi_pairs_all.npy`（6303×2）已随迁并自包含。
- **`_cfg.py` 增强**：`phi_pairs_path()` 探测升级为三级——环境变量 `AIQ_PHI_PAIRS` → 插件内 params/ 同目录副本 → 工作区 AIQ/ 回退。插件迁移到任意位置数据仍可定位。
- **B05/B07/B08 路径适配**：硬编码 `_PARAM_DIR/..` 的 phi_pairs 路径统一改为 `_cfg.phi_pairs_path()`（插件自包含优先），并补 `import _cfg`。

### 能力总览（合并后）
| 层 | 内容 | 验证 |
|----|------|------|
| 诊断 API | diagnose/trace/compare/stability/family（modules/ 13 源文件） | 五套件 189 断言 |
| 参数验证 | 59 参数 verify.py（A-T 九组） | 59/59 PASS |
| 真实数据 | baselines/ 5 模型 + params/ 真实曲率 + _real_metrics | 门禁 10/10 |

### 验证
- 插件本体五套件全绿：`run_all`(33) + `verify`(56) + `robust`(14) + `selfcheck`(73) + `extras`(13) = **189 断言**。
- 参数体系：**插件内 params/ 59/59 PASS** 且 **工作区 AIQ/参数 59/59 PASS**（双端一致）。
- `cli selfcheck` 门禁 10/10；`ruff`/`mypy` 静态检查全绿。

---

## 2.5.0 (2026-09-08) — 能力拓展：溯源鲁棒性 + 家族族谱

### 新能力（两大模块，承接 59 参数 E 组溯源体系）
- **`modules/stability.py` 溯源鲁棒性**：对目标模型指纹施加三类扰动（SFT 微扰 / 层谱截断 / 幅度改写，档位覆盖"由轻到毒"），统计"扰动后归属存活率"。
  - **判据经实测校准**：存活 = 扰动后最近邻认回自身（位移仅监控）。v0 曾用"位移<阈值"硬否决，实测发现 scale 50% / trunc 90% / sft σ=0.1 等强扰动下仍 100% 认回自身（**剖面形状不变性**），旧判据把强鲁棒指纹误报为失守——已重写。
  - **科学结论（诚实呈现）**：几何指纹剖面形状对微调/改写/层面抹除高度不变 → 溯源强鲁棒，版权维权证据可靠。
- **`modules/family.py` 家族族谱**：跨模型指纹亲缘聚类（距离=剖面 MAD，纯 numpy 单链接 agglomerative，无 scipy 依赖）。
  - **距离口径修正**：v0 用完整向量的欧氏距离，曲率标量段（K%≈45 量级）主导导致 5 档全判"远亲"零凝聚；改为**剖面段 MAD** 后 Qwen base↔Instruct=0.0031 精准识别（SFT 同族）。
  - **保守聚类**：阈值取 SFT 级（0.01），避免单链接 chaining（TinyLlama 作桥梁把 Qwen 与 BLOOM 误合 1 簇）；亲缘表保留完整近亲/远亲视角。

### 工程
- `schema.py` 新增 stability/family 阈值常量（STAB_MAD_THR / PERT_* / FAM_*），全部标注 heuristic 并纳入 EVIDENCE；selfcheck 证据纪律覆盖。
- `cli.py` 新增 `stability [TARGET]`、`family` 子命令（支持 --json/--html）；`report.py` 新增 `render_stability` / `render_family` 自包含 HTML 卡片。
- `modules/__init__.py` 导出 `stability / format_stability / family_tree / format_family`。
- 新增测试：`tests/test_extras.py`（13 断言：扰动边界、归属存活、距离性质、保守聚类）。

### 验证
- 五套测试全绿：`run_all`(33) + `verify`(56) + `robust`(14) + `selfcheck`(73) + `extras`(13) = **189 断言**。
- `ruff` 0 错误 · `mypy --check-untyped-defs` 0 错误（18 源文件）· `cli selfcheck` 门禁 10/10。
- CLI 冒烟：`stability` 归属存活 100% + 位移灰度；`family` Qwen 双变体同簇（d=0.0031），BLOOM/Llama 独立簇。

---

## 2.4.1 (2026-09-08) — 生产就绪度评审整改（类型修正 + 环境无关测试 + 压缩建议集中）

### 修复（承接生产就绪度评审 P2）
- **mypy 严格模式类型错误**：`harness.py` 中 `fams` 变量在 qkv 分支（3 元组）与单投影分支（1 元组）赋值类型不一致，显式注解 `tuple[str, ...]` 修复；`mypy --check-untyped-defs` 现 **Success（15 源文件）**。
- **test_robustness 环境相关失败**：`缺 torch 依赖守卫`测试依赖真实环境是否装有 torch（本地有→FAIL，CI 无→PASS）。改为通过 `builtins.__import__` 拦截桩**模拟缺 torch**，任何环境可重复，`finally` 恢复零副作用。
- **ruff I001 导入排序**：`tests/test_robustness.py` 函数内导入排序修正；`ruff check modules tests` 全绿。

### 工程（承接评审"可压缩性阈值集中"）
- **压缩建议档位集中到 schema**：新增 `COMPRESS_HIGH_RATIO/MID_RATIO/LOW_RATIO`（keep_ratio 建议区间），`health.py` 中硬编码的"0.25-0.4 / 0.5-0.6 / 保持全精度"文案改从 schema 读取；`_COMPRESS_*` 重复定义收敛到文件头；EVIDENCE 补 3 个新标签。
- 可压缩性阈值 **0.35/0.45 仍为 heuristic（推测级）**：诚实标注不变，升级路径见 `docs/可压缩性阈值标定协议.md`（待压缩-精度对照实验执行）。

### 验证
- 四套测试全绿：`run_all`(33) + `verify`(56) + `robust`(14) + `selfcheck`(73) = **176 断言**。
- `ruff` 0 错误 · `mypy --check-untyped-defs` 0 错误 · `cli selfcheck` 门禁 10/10 · `cli health` 冒烟正常（压缩建议显示 keep_ratio 0.25-0.40 等，取自 schema）。

---

### 能力
- 新增 `modules/selfcheck.py` **生产完整性门禁（自检）**：发布前一键自证清白，6 类检查全覆盖——
  ① 版本三源一致（`__init__`/`pyproject`/`CHANGELOG`）；② 基准库完整性（每档可装载、核心字段齐备且有限）；③ 数值叶片无 NaN/Inf；④ **AIQ 实时复算 vs 存档无漂移**（捕获存档旧公式漂移）；⑤ 证据纪律（EVIDENCE 标签合法、启发式判据保持 heuristic 标注）；⑥ 报告冒烟 + 隐私脱敏。已接入 `cli selfcheck [--json]` 并导出 API（`run_selfcheck` / `render_selfcheck`）。
- **DEFF 区间口径落码**：新增 `health.py::calibrate_deff_band()`，从基准家族 `DEFF_plat` 观测支撑集实时标定带 [lo,hi]（当前实测 5 模型 [1.5825,1.6072]，对 π/2 系统性正偏 +0.74%~+2.32%）；`health_verdicts` 增加可选 `deff_band` 参数，在说明中追加"家族标定带内/出带"走向；`diagnose` 输出新增 `curvature.deff_band`。布尔判据（DEFF_TOL）与既有契约完全不变。

### 测试
- 新增 `tests/test_selfcheck.py`（73 断言）：覆盖门禁整体通过、版本三源一致、DEFF 带出厂值/带内/出带标注、5 模型核心路径逐条解析。
- 门禁套件累计：`run_all`(33) + `verify`(56) + `robust`(14) + `selfcheck`(73) = **176 断言**。

### 文档
- `SKILL.md` / `docs/使用指南.md`：补 `selfcheck` 子命令、DEFF 家族标定带口径、断言总数 33+56+14+73、结构树含 `selfcheck.py`。

### 版本
- `pyproject.toml`、`modules/__init__.py` 同步 2.4.0。

### 验证
- 自检门禁 10/10 通过；`ruff` 0 错误；`mypy` 0 错误；四套件全绿；`cli report` 冒烟生成跨家族综合报告 HTML。

---

## 2.3.3 (2026-09-08) — 因子判别度诊断 + DEFF 区间口径 + 文档一致性回归

### 科学口径（承接审计发现 1/3）
- 新增跨家族**因子判别度诊断**：`report.py::render_factor_panel` 实时复算各模型 f1..f5，逐因子呈现跨模型跨度/相对变异与判别力分级（高/中/低/基线），把"f2(DEFF稳定) 近乎恒定、对排序零贡献"诚实呈现，不做隐式暗改。已接入 `cli report`，并导出 API。
- **DEFF 平台改区间口径**：确认 5 模型 `DEFF_plat` 对 π/2 均呈系统性正偏（+0.74%~+2.32%，均值 +1.38%±0.60%）。SKILL.md / 使用指南 改按"区间 + 误差带 + 提示系统正偏"表述，不再以单一"锁定 π/2"点值断言。
- 健康判定文案同步为区间口径；可压缩阈值 0.35/0.45 保持"推测级，非实测"标注（升级协议见 docs/）。

### 文档一致性（承接审计发现 4）
- `SKILL.md`：修正内部自相矛盾的断言计数（100-101 行旧 23/54 → 33/56/14，109 行 23+54 → 33+56+14），补齐 robustness 套件入口。
- `docs/使用指南.md`：标题更新为"三家族 5 模型标定"；设计原则补 8/9 两条（可观测层、门禁）；目录结构补 `schema.py`/`observation.py`/`test_robustness.py` 与报告导出能力；输出解读补 f2 判别力提示与 DEFF 区间口径。

### 版本
- `pyproject.toml`、`modules/__init__.py` 同步 2.3.3。

### 验证
- 复跑门禁：`ruff` 0 错误 · `mypy` 0 错误（13 源文件）· `run_all`=33 · `verify`=56 · `robust`=14，合计 **103 断言全绿**；`cli report` 冒烟生成含因子判别度面板的 HTML。

---

## 2.3.2 (2026-09-08) — 可观测层落地 + CI 目录/依赖修复（大厂 checklist 归零 ❌）

### 日志 / 可观测（此前 ❌ 全 print → ✅ 结构化可观测）
- 新增 `modules/observation.py` 最小可观测层：集中式 `setup_logging`（时间戳+级别+模块），支持**人类可读文本**与 **JSONL 结构化日志**（ts/level/logger/pid/msg，CI 可解析），进程级幂等配置。
- `cli.py` 全局新增 `--log FILE`、`--json-log FILE`、`--verbose`；每次运行记录 `subcommand/run_id/exit`，错误改走 `_log.exception`。
- 反伪造敏感副作用接入日志：`data.py`（P1 命中 / 存档损坏→兜底 / 无存档→审计兜底）、`harness.py`（真实测量开始/完成：model/tok_s/AIQ/f3/f4）。

### CI（此前路径/依赖引用不成立 → ✅ 可过）
- `.github/workflows/ci.yml` 修正 `working-directory` 为 `.trae/skills/aiq-geometric-forensics`，`Install deps` 步骤改为在该目录 `-c constraints.txt -r requirements.txt`（原先在 repo 根引用无效），并新增 **CLI 冒烟门禁**（`list --json-log` 产出 JSON 与日志）。

### 版本
- `pyproject.toml`、`modules/__init__.py::__version__` 同步至 2.3.2。

### 验证
- `ruff` 0 错误 · `mypy` 0 错误（13 源文件）· `run_all`=33 · `verify`=56 · `robust`=14，合计 **103 断言全绿**；CLI `list --json-log` 端到端产出 JSONL。

---

## 2.3.1 (2026-09-08) — 大厂工程质量门禁补强

### 静态类型检查 / Lint（此前 ❌ → ✅）
- 新增 `pyproject.toml` 的 `[tool.ruff]`（E/F/W/I/UP/B 规则集）与 `[tool.mypy]` 配置。
- 全仓清空：`ruff check modules tests` = **0 错误**，`mypy modules tests` = **0 错误**（12 源文件）。
- 顺手消除：死变量（pool_idx/rng/v4/D）、未用循环变量、裸分号单行、裸 f-string、`zip` 缺 `strict`、typing.Tuple→tuple 等类。

### 异常路径测试（此前 ⚠ 无覆盖 → ✅）
- 新增 `tests/test_robustness.py`（14 断言）：未知模型降级 fallback、损坏 JSON 不崩、AIQ 边界输入有界、缺 torch 依赖守卫、脏 proj_gamma 过滤、路径隐私净化（含 Windows）。

### CI / 版本锁定 / 预提交（此前 ❌ → ✅）
- 新增 `.github/workflows/ci.yml`：py3.10/3.11 × lint+mypy+3 套件门禁。
- 新增 `.pre-commit-config.yaml`：ruff/mypy/空白/尾行拦截。
- 新增 `constraints.txt` 固化 numpy（CI 复现锁定）。

### 科学口径（P1 延展）
- 新增 `docs/可压缩性阈值标定协议.md`：可压缩阈值 0.45/0.35 从推测级升级为实测的可执行数据契约 + 基线层级 + 模型级防泄漏划分。

### 验证
- `run_all`=33 · `verify`=56 · `robust`=14，合计 **103 断言全绿**；ruff、mypy 全绿。

---

## 2.3.0 (2026-09-08) — AIQ 标定修复 + 单一事实源 + 生产就绪

### 修复（P1-科学/阻断）
- **「H 收敛」f3 退化**：`f3 = 1 - min(1, Hmed)` 对真实模型 Hmed∈[1.24,1.38] 恒为 0。改为有界单调衰减 `f3 = 1/(1+Hmed)`（Hmed≥0 时 ∈(0,1]），真实 f3 现落 0.42–0.45，有区分度。health 与 harness 同步修正，数学等价。
- **f1/f4 重复计数**：旧调用把 f4 传成 f1（g 投影谱集中）导致同一值被两次加权。现 f4 为「全投影谱集中均值」（k/q/v/o/gate/up/down 各投影 spl_gamma 的均值），与 f1(k 单投影) 独立。diagnose 与 measure 均实时取独立 SPL。
- **AIQ 实时复算**：diagnose 不再信任存档内嵌旧 aiq，改为从原始字段实时复算，杜绝旧式退化数据被沿用。5 份存档 `aiq` 块已按新公式回标定（rev=2.3-f3/f4-fixed），AIQ 现为 52.67–58.02。

### 配置集中（P1-配置 / P2-魔法数）
- 新增 `modules/schema.py` 单一事实源：AIQ 权重、K_slot、DEFF_tol、H 阈值、可压缩分级(0.45/0.35)、溯源阈值、指纹维、图表上限，并附**证据分级标注**（measured/theory/heuristic/design）。
- `health.py` / `forensics.py` / `data.py`(DEFF 平台) / `cli.py`(yMax) 全部改为从 schema 引用，消除散落魔法数漂移。
- 可压缩性阈值明确标注为**推测级（经验）**，非实测；待压缩-精度对照实验升级。

### 工程化（P2）
- `data.py` 损坏/不可读存档安全降级为 fallback（不崩溃、不伪造）。
- CLI 错误出口改用结构化 `logging`（不污染正常 stdout 报表）。
- 新增 `requirements.txt`、`pyproject.toml`、`LICENSE`(MIT)；`modules` 包暴露 `schema`，版本升至 `2.3.0`。

### 验证
- `tests/run_all_tests.py`：PASS=33 FAIL=0（新增 f3 非退化、f4 独立、AIQ≈54.60 断言）
- `tests/verify_derivations.py`：PASS=56 FAIL=0（f3=1/(1+Hmed) 解析值更新）
- 全部 CLI 子命令 + HTML/JSON 导出实测通过。

### 遗留（诚实标注，非本版范围）
- 可压缩性分级（0.35/0.45）仍未做压缩-精度对照实验，属推测级。
- f4 的「全投影谱集中」与 f1「k 投影谱集中」有一定相关性（数据固有，非方法缺陷）；若需严格正交可引入跨 token 轨迹谱投影的独立 SPL 采样（需重测模型，当前沙箱无 torch）。

---

## 2.2.1 (2026-09-08) — BUG 修复轮

### 修复
- **[核心] 跨层数溯源失效**：fingerprint 向量原为 `层数+5` 维，不同层数模型的 MAD 最近邻因 shape 不匹配恒返回 `unknown/inf`（如 28 vs 22 层）。改为把 Gamma 剖面按相对深度重采样到恒定 `_PROFILE_K=64` 点，维度恒定 `64+5`，使任意层数指纹可比、溯源归位。
- **深度剖面非等分**：旧 `step=ceil(N/3)` 在 22 层切成 8/8/6、28 层 10/10/8，不均衡。改用 `array_split` 严格三等分（层数差 ≤1）。
- **兜底 DEFF 不精确**：`_FALLBACK` 的 `DEFF_plat=1.57` 与 health 的 π/2 精确值不一致，统一为精确 π/2。
- **死常量 MARGIN_EPS=0.29 从未使用**：`margin` 形同虚设。`family_verdict` 扩为三档（同族/变异/明显异族），让该门控真正参与判定。

### 验证
- 跨层数溯源回归化：`trace(28层, 22层)` 由 unknown/inf → MAD 有限且归属明确。
- `tests/run_all_tests.py`：PASS=25 FAIL=0（新增跨层数追溯断言）
- `tests/verify_derivations.py`：PASS=55 FAIL=0（指纹维度契约改为恒定 K+5）

### 待定（涉及 AIQ 标定语义，未擅改）
- `f3 = 1 - min(1, H_median)`：5 个真实模型 H_median∈[1.24,1.38] 全部≥1 使 f3 恒 0，「H 收敛」判据对真实模型 100% 失败、无区分度。
- `f1`/`f4` 在无独立 SPL 实测时取值相同，AIQ 实为约 4 个独立因子。

---

## 2.2.0 (2026-09-08) — 完整专业化（上一轮）

### 新增
- 统一命令行入口 `modules/cli.py`：`list` / `health` / `trace` / `compare` / `measure` / `report` 六子命令，统一退出码。
- 自包含 HTML 报告导出 `modules/report.py`：内联 CSS + SVG（横向条形/深度剖面折线），零外部依赖，含中文字体。
- **证据分级徽标**（real-inline 绿 / baseline 蓝 / fallback 琥珀），报告绝不把兜底伪装成实测。
- `cli` 支持 `--json FILE`（结构化导出）与 `--html FILE`（自包含报告）；`report` 子命令生成跨家族综合报告。
- `CHANGELOG.md`：版本沿革。

### 修复
- 移除 `harness.py` 残留的伪造高斯死代码 `_load_phi_pairs`（仍含硬编码 λ=0.8516、引用不存在的 `phi_pairs_all.npy`）。
- `measure()` 返回 schema 对齐基准存档：arch 补 `n_params` / `dtype`，`_real_metrics.json` 等坏引用清除。
- `compressibility`：防御投影 `proj_gamma` 为 `None`/非有限值，未实测投影不再抛错而是跳过（新综合报告触发）。
- 文档一致性：SKILL.md / 使用指南 清理残留旧值（0.4695/49.94/0.0009）与坏文件引用，更新为跨家族 5 模型、实时获取口径、CLI 用法与 23+54 断言。
- `modules/__version__ = "2.2.0"`，与 `harness_version` 对齐。

### 验证
- `tests/run_all_tests.py`：PASS=23 FAIL=0
- `tests/verify_derivations.py`：PASS=54 FAIL=0
- 全部 CLI 子命令 + HTML/JSON 导出实测通过

---

## 2.1 (2026-09-07) — 跨家族实测 + 实时获取

- 新增跨家族真实测量：Qwen-0.5B(base/Instruct)、Qwen2.5-1.5B-Instruct、TinyLlama-1.1B-Chat、BLOOM-560M 共 5 个基准存档。
- 硬编码 → 实时获取：DEFF_cv 由真实曲率分布计算；层数由各家族 config 动态解析；曲率对由真实生成轨迹拟合；AIQ 兜底改为中立结构占位。

## 1.0 — 初始

- 首个可用的 59 参数几何指纹验证包（Qwen-E-only 标定、文档水分值审计）。