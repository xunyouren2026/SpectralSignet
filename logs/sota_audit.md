# AIQ 参数验证 SOTA 审计报告

- 生成时间：params 环境，共审计 **59** 个 verify.py

## 三向达标统计

| 指标 | 达标数 |
| --- | --- |
| 类型安全（Config 字段类型注解） | 59/59 |
| 错误处理（validate_* 用 AIQValidationError 子类/ValueError） | 59/59 |
| 防御性（外部数据校验） | 59/59 |

## 逐脚本明细

| 脚本 | 类型安全 | 错误处理 | 防御性 | 结论 |
| --- | --- | --- | --- | --- |
| A01_model | PASS | PASS | PASS | PASS |
| A02_prompt | PASS | PASS | PASS | PASS |
| A03_gen_len | PASS | PASS | PASS | PASS |
| A04_layer_names | PASS | PASS | PASS | PASS |
| A05_projs | PASS | PASS | PASS | PASS |
| A06_三实例路径 | PASS | PASS | PASS | PASS |
| B01_M | PASS | PASS | PASS | PASS |
| B02_N_max | PASS | PASS | PASS | PASS |
| B03_GRID | PASS | PASS | PASS | PASS |
| B04_TOK_PER_FRAME | PASS | PASS | PASS | PASS |
| B05_SAMPLES_PER_FRAME | PASS | PASS | PASS | PASS |
| B06_N_SAMPLE_PER_FRAME | PASS | PASS | PASS | PASS |
| B07_K_NN_CURV | PASS | PASS | PASS | PASS |
| B08_NFRAME | PASS | PASS | PASS | PASS |
| B09_n_sample | PASS | PASS | PASS | PASS |
| B10_anchor | PASS | PASS | PASS | PASS |
| C01_energy_thr | PASS | PASS | PASS | PASS |
| C02_ks | PASS | PASS | PASS | PASS |
| D01_bins | PASS | PASS | PASS | PASS |
| D02_fit_range | PASS | PASS | PASS | PASS |
| D03_theta_grid | PASS | PASS | PASS | PASS |
| D04_mu_grid | PASS | PASS | PASS | PASS |
| D05_sigma_grid | PASS | PASS | PASS | PASS |
| D06_a_grid | PASS | PASS | PASS | PASS |
| D07_sp_grid | PASS | PASS | PASS | PASS |
| D08_mc_n | PASS | PASS | PASS | PASS |
| D09_lam_grid | PASS | PASS | PASS | PASS |
| D10_target_deff | PASS | PASS | PASS | PASS |
| D11_closure_thr | PASS | PASS | PASS | PASS |
| E01_REF_PROMPTS | PASS | PASS | PASS | PASS |
| E02_REWRITES | PASS | PASS | PASS | PASS |
| E03_trunc | PASS | PASS | PASS | PASS |
| E04_GEN_PROMPT | PASS | PASS | PASS | PASS |
| E05_PROM | PASS | PASS | PASS | PASS |
| E06_N_FRAMES | PASS | PASS | PASS | PASS |
| E07_ln_sigma | PASS | PASS | PASS | PASS |
| F01_ctx | PASS | PASS | PASS | PASS |
| F02_bandwidths | PASS | PASS | PASS | PASS |
| F03_kv_ctx | PASS | PASS | PASS | PASS |
| F04_ctx_scan | PASS | PASS | PASS | PASS |
| F05_gen_len_eng | PASS | PASS | PASS | PASS |
| F06_out_dir | PASS | PASS | PASS | PASS |
| G01_gate_strength | PASS | PASS | PASS | PASS |
| G02_keep_ratio | PASS | PASS | PASS | PASS |
| G03_max_dim | PASS | PASS | PASS | PASS |
| G04_target_metric | PASS | PASS | PASS | PASS |
| G05_max_ctx | PASS | PASS | PASS | PASS |
| H01_seed | PASS | PASS | PASS | PASS |
| H02_OPENBLAS_NUM_THREADS | PASS | PASS | PASS | PASS |
| H03_OMP_NUM_THREADS | PASS | PASS | PASS | PASS |
| H04_data_file | PASS | PASS | PASS | PASS |
| H05_dtype | PASS | PASS | PASS | PASS |
| T01_DEFF平台告警 | PASS | PASS | PASS | PASS |
| T02_非稳态告警 | PASS | PASS | PASS | PASS |
| T03_K%告警槽 | PASS | PASS | PASS | PASS |
| T04_AIQ因子权重 | PASS | PASS | PASS | PASS |
| T05_尖峰显著性 | PASS | PASS | PASS | PASS |
| T06_闭环判据 | PASS | PASS | PASS | PASS |
| T07_谱门控增强 | PASS | PASS | PASS | PASS |

---
_由 params/_sota_audit.py 自动生成_