"""
aiq-geometric-forensics.modules.harness — 真实模型测量引擎
=====================================================================
专业包的一等公民：对任意本地 HuggingFace 因果语言模型提取真实激活，
计算全部真实几何指标（Gamma / 曲率身份 / 工程 / AIQ），输出可被
health / forensics / compare 直接消费的 measurement dict。

可选依赖：torch + transformers（仅在真正测量时导入）。若不可用，
上层自动回退到 baselines/ 存档或审计兜底值——保证包"无大模型也能体检"。

典型调用：
  import modules.harness as H
  m = H.measure(model_dir="Qwen2.5-0.5B-Instruct", ngen=64)
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

from . import observation

# 模块级可观测 logger（副作用写入日志层；未配置时 no-op）
_log = observation.get_logger(__name__)

# --- 环境前置：必须在 numpy/torch 导入前（OpenBLAS 保护）---
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# -- 延迟导入重型依赖：仅当显式调用 measure() 且参数 given 时加载 --
_np, _torch, _tr = None, None, None
try:
    import numpy as np
    _np = np
except Exception:                                     # pragma: no cover
    _np = None


def _require_numpy():
    if _np is None:
        raise ImportError("numpy 不可用，harness 无法运行。")
    return _np


def _load_ml():
    """延迟加载 torch/transformers（测量时才导入，避免开销）。"""
    global _torch, _tr
    if _torch is None:
        import torch
        _torch = torch
    if _tr is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tr = (AutoModelForCausalLM, AutoTokenizer)
    return _torch, _tr


# ---------------------------------------------------------------- 架构自适应
# 消除对"Qwen 字段名"的硬编码：不同家族 config 字段名不同，统一别名解析。
#   Qwen/Llama: num_hidden_layers / hidden_size / num_key_value_heads
#   BLOOM:      n_layer / n_embed / （无 KV 头→用注意力头）
def arch_from_config(cfg: Any) -> dict[str, Any]:
    def _num(names, defval):
        for n in names:
            v = getattr(cfg, n, None)
            if v is not None:
                return int(v)
        return defval
    n_layers = _num(("num_hidden_layers", "n_layer", "layers"), 12)
    hidden = _num(("hidden_size", "n_embed", "hidden_dim", "d_model"), 512)
    n_head = _num(("num_attention_heads", "n_head", "num_heads"), 1)
    n_kv = _num(("num_key_value_heads", "n_head", "num_attention_heads"), n_head)
    hd = hidden // max(1, n_head)
    return {"n_layers": n_layers, "n_head": n_head, "n_kv": n_kv,
            "hidden": hidden, "hd": hd,
            "model_type": str(getattr(cfg, "model_type", ""))}


# 投影家族规范名（跨家族统一口径：k/q/v/o/gate/up/down）
_FAMILY_ALIASES = {
    "k_proj": ("k_proj",), "q_proj": ("q_proj",), "v_proj": ("v_proj",),
    "o_proj": ("o_proj", "dense", "c_proj"),         # BLOOM 注意力输出=dense
    "gate_proj": ("gate_proj", "w2", "ffn_out"),
    "up_proj": ("up_proj", "dense_h_to_4h", "c_fc", "w13"),
    "down_proj": ("down_proj", "dense_4h_to_h"),
}
_QKV_FUSED = "query_key_value"

def _layer_idx(path: str) -> int:
    """从任意模块路径提取层号，兼容多家族命名（Qwen/Llama/BLOOM）。"""
    import re as _re
    m = _re.search(r"(?:layers|h)\.(\d+)\.", path)
    if m:
        return int(m.group(1))
    m = _re.search(r"\.(\d+)\.", path)
    return int(m.group(1)) if m else 0

def projection_roles(model, n_head: int, hd: int) -> dict[str, str]:
    """扫描模型全部 Linear 模块，分类为规范家族名；BLOOM 融合 QKV 标记为 qkv。
    返回 {module_path: family_name}，family ∈ {k_proj,q_proj,v_proj,o_proj,
    gate_proj,up_proj,down_proj} 或 "qkv"。"""
    roles: dict[str, str] = {}
    for n, m in model.named_modules():
        if "Linear" not in m.__class__.__name__:
            continue
        leaf = n.rsplit(".", 1)[-1]
        if leaf == _QKV_FUSED:
            roles[n] = "qkv"
            continue
        for fam, aliases in _FAMILY_ALIASES.items():
            if leaf in aliases:
                roles[n] = fam
                break
    return roles


def _slice_qkv(o, which: str, n_head: int, hd: int):
    """BLOOM 融合 QKV 输出 [.., 3*hidden]，按 head 交错 [q,k,v]，切出指定分量。"""
    np = _require_numpy()
    o = np.asarray(o)
    T = o.shape[-2]
    idx = {"q": 0, "k": 1, "v": 2}[which]
    try:
        o3 = o.reshape(T, n_head, 3, hd)
        return o3[:, :, idx, :].reshape(T, n_head * hd).astype(np.float32, copy=False)
    except Exception:
        return o.astype(np.float32, copy=False)       # 无法切分则整块（退而求其次）


# ---------------------------------------------------------------- 谱集中
def spl_gamma(H) -> float:
    """跨 token 谱集中度 = PCA 前 3 特征值占比（C-子空间能量占比口径）。"""
    np = _require_numpy()
    Hc = H - H.mean(0, keepdims=True)
    if Hc.shape[0] < 3 or not np.isfinite(Hc).all():
        return float("nan")
    S = np.linalg.eigvalsh(Hc.T @ Hc / (Hc.shape[0] - 1))[::-1]
    tot = float(S.sum())
    return float(S[:3].sum() / tot) if tot > 0 else float("nan")


def gamma_energy(H, m: int = 3) -> float:
    """能量口径 Gamma = 前 m 个奇异值平方和 / 总平方和。"""
    np = _require_numpy()
    Hc = H - H.mean(0, keepdims=True)
    S = np.linalg.svd(Hc, compute_uv=False)
    S2 = S ** 2
    return float(S2[:m].sum() / (S2.sum() + 1e-12))


# ---------------------------------------------------------------- 曲率身份
def curvature_metrics(data) -> dict[str, Any]:
    """从逐点 (κ1,κ2) 计算曲率身份指标（版本A口径）。"""
    np = _require_numpy()
    k1, k2 = data[:, 0], data[:, 1]
    a1, a2 = np.abs(k1), np.abs(k2)
    denom = k1 * k1 + k2 * k2
    deff = (a1 + a2) ** 2 / (denom + 1e-16)            # DEFF ∈ [1,2]
    gauss = k1 * k2
    k_neg = 100.0 * float(np.mean(gauss < 0))
    hmed = float(np.median(np.abs((k1 + k2) / 2)))
    small, large = np.minimum(a1, a2), np.maximum(a1, a2)
    phi = np.degrees(np.arctan2(small, large + 1e-16))
    lam = float(k2.std() / (k1.std() + 1e-16))
    energy = float((a1 + a2).mean())
    # DEFF 变异系数：从本次真实曲率数据实时算，而非硬编码 Qwen 的值
    deff_cv = float(deff.std() / (deff.mean() + 1e-16)) if deff.size > 1 else 0.0
    return {
        "DEFF_plat": float(deff.mean()), "K_neg_pct": k_neg,
        "H_median": hmed, "phi_mean_deg": float(phi.mean()),
        "phi_med_deg": float(np.median(phi)),
        "same_sign_ratio": float(np.mean(gauss > 0)),
        "corr_k1k2": float(np.corrcoef(k1, k2)[0, 1]),
        "corr_abs": float(np.corrcoef(a1, a2)[0, 1]),
        "lambda_ratio": lam, "energy_mean": energy,
        "DEFF_cv": deff_cv, "n_points": int(len(k1)),
    }


# ---------------------------------------------------------------- AIQ 五因子
def aiq_score(gamma_layers: list[float], deff_cv: float, h_median: float,
              spl: float, w=(0.20, 0.25, 0.20, 0.20, 0.15)) -> dict[str, Any]:
    """AIQ = 100·Σ(w_i·f_i)，五因子综合商（与 health.aiq_factors 数学等价）。

    f3 = 1/(1+Hmed)：有界单调衰减的「曲率尺度收敛指数」。
    f4 由调用方传入独立 SPL（全投影谱集中均值），不得再传 f1 造成重复计数。
    """
    np = _require_numpy()
    f1 = float(np.mean(gamma_layers))
    f2 = 1.0 - float(deff_cv)
    f3 = 1.0 / (1.0 + max(float(h_median), 0.0))
    f4 = float(spl) if gamma_layers else f1
    f5 = 1.0 - (float(np.max(gamma_layers)) - float(np.min(gamma_layers)))
    aiq = 100.0 * (w[0] * f1 + w[1] * f2 + w[2] * f3 + w[3] * f4 + w[4] * f5)
    return {"AIQ": float(aiq), "f1": f1, "f2": f2, "f3": f3, "f4": f4,
            "f5": f5, "weights": list(w)}


# ---------------------------------------------------------------- 激活捕获
PROMPT = ("The universe is expanding, and galaxies are drifting farther apart "
          "over time. In the early universe, matter was distributed almost "
          "uniformly, and small fluctuations grew under gravity.")


def _capture_acts(model, ids, kinds=("k_proj", "q_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj")) -> dict[str, Any]:
    """跨家族激活捕获：BLOOM 融合 QKV 按 head 切分出 K/Q/V；Qwen/Llama 直取。
    返回 {模块路径#家族名: 真实激活矩阵}。"""
    torch, _ = _load_ml()
    np = _require_numpy()
    arch = arch_from_config(model.config)
    roles = projection_roles(model, arch["n_head"], arch["hd"])
    acts: dict[str, Any] = {}
    hooks = []

    def _mk(path, role):
        def _fn(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o = o.detach().float().cpu().numpy()
            if o.ndim == 3:
                o = o[0]
            if role == "qkv":
                for which, fam in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
                    acts[f"{path}#{fam}"] = _slice_qkv(o, which, arch["n_head"], arch["hd"])
            else:
                acts[f"{path}#{role}"] = o.astype(np.float32, copy=False)
        return _fn

    for path, role in roles.items():
        fams = ("k_proj", "q_proj", "v_proj") if role == "qkv" else (role,)
        if not any(f in kinds for f in fams):
            continue
        try:
            mod = model.get_submodule(path)
        except Exception:
            continue
        hooks.append(mod.register_forward_hook(_mk(path, role)))
    try:
        with torch.no_grad():
            _ = model(ids)
    finally:
        for h in hooks:
            h.remove()
    return acts


def curvature_from_acts(acts: dict[str, Any], max_pairs: int = 4096):
    """从真实激活矩阵推导主曲率对 (κ1,κ2) —— 取代伪造高斯对。

    原理：每张真实激活矩阵 H（形状 [seq, d]）去中心后，其奇异值
    σ1≥σ2≥… 刻画激活点云沿各主轴的标准差。取前两主轴作为主曲率：
      κ1 = σ1, κ2 = σ2   （形状尺度下的主方向弯曲）
    跨全部投影类型×逐层聚合，得到该模型真实专属的 (κ1,κ2) 分布，
    再交给 curvature_metrics() 计算 DEFF / K<0% / Hmed / φ / λ。

    任一模型其光谱集中度不同 → 奇异值谱不同 → 曲率身份不同，
    从而真正具备家族区分度（也消除旧"伪造高斯对"中的硬编码 λ=0.8516）。
    """
    np = _require_numpy()
    pairs: list[tuple] = []
    for _, H in acts.items():
        if not isinstance(H, np.ndarray) or H.ndim != 2 or H.shape[0] < 2:
            continue
        Hc = H - H.mean(0, keepdims=True)
        s = np.linalg.svd(Hc, compute_uv=False)
        if s.size < 2:
            continue
        pairs.append((float(s[0]), float(s[1])))
    arr = np.asarray(pairs, dtype=np.float32)
    if arr.size == 0:                                     # 极端回退：无有效激活
        return None
    if len(arr) > max_pairs:                             # 聚合样本上限（可选）
        idx = np.linspace(0, len(arr) - 1, max_pairs).astype(int)
        arr = arr[idx]
    return arr


# ---------------------------------------------------------------- 轨迹曲率（真实口径）
# 忠实移植自已验证实验：对生成激活轨迹逐点做 kNN 局部二次拟合，
# 取 Hessian 本征值 (κ1,κ2) —— 这是产出 DEFF_plat≈π/2 结论的精算口径。

def capture_trajectory(model, tok, prompt, ntoken):
    """自回归生成 ntoken 个 token，逐 token 捕获 k/up/o 三类投影的最后一步激活。
    跨家族：BLOOM 融合 QKV 按 head 切出 K，dense_h_to_4h/dense 映射到 up/o。
    返回 (n_gen, layer_td)，其中 layer_td[key='family_lidx'] = [T, dim]。"""
    np = _require_numpy()
    torch, _ = _load_ml()
    arch = arch_from_config(model.config)
    roles = projection_roles(model, arch["n_head"], arch["hd"])
    # 只保留曲率用到的家族：k_proj / up_proj / o_proj
    want = {"k_proj", "up_proj", "o_proj"}
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    captures: dict[str, list] = {}
    hooks = []
    for path, role in roles.items():
        fams: tuple[str, ...]              # 该层挂钩的投影家族（可变长元组）
        if role == "qkv":
            fams = ("q_proj", "k_proj", "v_proj")
        else:
            fams = (role,)
        use = [f for f in fams if f in want]
        if not use:
            continue
        try:
            mod = model.get_submodule(path)
        except Exception:
            continue
        lidx = _layer_idx(path)
        def _h(m, i, o, path=path, role=role, lidx=lidx, use=use, arch=arch):
            arr = o.detach().float().cpu().numpy()[0]      # [T, dim]
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            if role == "qkv":
                fk, fq, fv = ("k_proj", "q_proj", "v_proj")
                for which, fam in (("q", fq), ("k", fk), ("v", fv)):
                    if fam not in want:
                        continue
                    sl = _slice_qkv(arr, which, arch["n_head"], arch["hd"])
                    captures.setdefault(f"{fam}_{lidx}", []).append(sl[-1:].copy())
            else:
                captures.setdefault(f"{role}_{lidx}", []).append(arr[-1:].copy())
        hooks.append(mod.register_forward_hook(_h))
    inp = ids
    past = None
    n = 0
    with torch.no_grad():
        for _ in range(ntoken):
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().item()
            if nxt == tok.eos_token_id:
                break
            inp = torch.tensor([[nxt]])
            n += 1
    for h in hooks:
        h.remove()
    layer_td: dict[str, Any] = {}
    for k, lst in captures.items():
        lst = lst[:n]
        if len(lst) < 2:
            continue
        try:
            layer_td[k] = np.concatenate(lst, axis=0)      # [T, dim]
        except Exception:
            continue
    return n, layer_td


def _curv_pair_pointwise(H_td, frame_mask, k_nn=15, n_sample=18, np=None):
    """对轨迹激活 H_td=[T,d]，在 frame 内的采样点逐点算 (κ1,κ2)。"""
    if np is None:
        np = _require_numpy()
    rows_idx = np.where(frame_mask)[0]
    if rows_idx.size == 0:
        return np.zeros((0, 2))
    pick = np.random.choice(rows_idx, size=min(n_sample, rows_idx.size * 3), replace=True)
    pick = np.unique(pick)[:n_sample]
    rows = H_td[pick].astype(np.float32, copy=False)
    M = rows.shape[0]
    if M < 4:
        return np.zeros((0, 2))
    T_all = H_td.shape[0]
    candidate = np.zeros(T_all, dtype=bool)
    for t in rows_idx:
        candidate[max(0, t - 5):min(T_all, t + 6)] = True
    pool = H_td[candidate]
    if pool.shape[0] < k_nn + 2:
        pool = H_td
    pairs = []
    pool_norm2 = (pool * pool).sum(axis=1)
    for m in range(M):
        q = rows[m]
        diff = pool - q[None, :]
        d2 = (diff * diff).sum(axis=1)
        order = np.argpartition(d2, k_nn)[:k_nn + 1]
        order = order[np.argsort(d2[order])]
        order = order[1:1 + k_nn]
        k = len(order)
        if k < 6:
            continue
        neigh = pool[order]
        nbar = neigh.mean(axis=0)
        X = neigh - nbar[None, :]
        try:
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S[1] < 1e-8 * max(S[0], 1e-12):
            continue
        V2 = Vt[:2].T
        u = X @ V2
        z = pool_norm2[order] - pool_norm2[order].mean()
        A = np.column_stack([np.ones(k), u[:, 0], u[:, 1],
                             0.5*u[:, 0]**2, u[:, 0]*u[:, 1], 0.5*u[:, 1]**2])
        try:
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        except Exception:
            continue
        Hes = np.array([[coef[3], coef[4]], [coef[4], coef[5]]])
        try:
            eigs = np.linalg.eigvalsh(Hes)
        except Exception:
            continue
        k1, k2 = eigs[0], eigs[1]
        if not (np.isfinite(k1) and np.isfinite(k2)):
            continue
        denom = k1 * k1 + k2 * k2
        if denom < 1e-24:
            continue
        deff = (abs(k1) + abs(k2)) ** 2 / denom
        if deff < 0.9 or deff > 2.1:                       # 几何有效性窗
            continue
        pairs.append((k1, k2))
    return np.array(pairs, dtype=np.float32)


def curvature_from_trajectory(layer_td, ntoken, n_frames=12, tok_per_frame=8,
                              k_nn=15, n_sample=18, seed=0):
    """把逐层逐帧的逐点曲率对聚合为模型真实 (κ1,κ2) 分布。
    替代旧的伪造高斯对（它合成数据并硬编码 λ=0.8516）。"""
    np = _require_numpy()
    T = ntoken
    if not layer_td:
        return None
    all_pairs = []
    for fi in range(n_frames):
        t_lo = fi * tok_per_frame
        t_hi = min(T, t_lo + tok_per_frame)
        if t_hi - t_lo < 4:
            break
        fm = np.zeros(T, dtype=bool)
        fm[t_lo:t_hi] = True
        for _, Hmat in layer_td.items():
            if Hmat.shape[0] < T:
                continue
            try:
                pr = _curv_pair_pointwise(Hmat, fm, k_nn=k_nn, n_sample=n_sample, np=np)
            except Exception:
                continue
            if len(pr) > 0:
                all_pairs.append(pr)
    if not all_pairs:
        return None
    return np.concatenate(all_pairs).astype(np.float32)    # [P, 2] 真实曲率对


# ---------------------------------------------------------------- 主入口
def measure(model_dir: str, ngen: int = 64,
            dtype: str = "float32",
            prompt: str = PROMPT) -> dict[str, Any]:
    """对指定模型目录做真实测量，返回完整 measurement dict。

    参数：
      model_dir: 本地模型目录（HF 格式）；为模型名时尝试在 baselines 检索。
      ngen: 自回归生成 token 数。
      dtype: 加载精度（"float32"/"float16"/"bfloat16"）。沙箱（≤6G 内存）
             下建议 "bfloat16" 并配 low_cpu_mem_usage，避免 fp32 OOM。
    返回：与 baselines/<模型名>.json 同构的标准 measurement dict —— 可直接写入
    baselines/ 或交由 diagnose()/trace()/compare() 消费。
    """
    np = _require_numpy()
    torch, (Atok, Acausal) = _load_ml()

    model_dir = _resolve_model_path(model_dir)
    torch.manual_seed(0)
    np.random.seed(0)
    _log.info("真实测量开始 model=%s ngen=%d dtype=%s",
              _model_display(model_dir), ngen, dtype)

    load_dtype = getattr(torch, dtype, torch.float32)
    tok = Atok.from_pretrained(model_dir, local_files_only=True,
                               trust_remote_code=True)
    model = Acausal.from_pretrained(model_dir, torch_dtype=load_dtype,
                                    local_files_only=True,
                                    trust_remote_code=True,
                                    low_cpu_mem_usage=True)
    model.eval()
    cfg = model.config
    _arch = arch_from_config(cfg)
    n_layers = _arch["n_layers"]
    n_kv = _arch["n_kv"]
    hd = _arch["hd"]
    hidden = _arch["hidden"]
    load_dtype = getattr(torch, dtype, torch.float32)
    elem_bytes = max(2 if load_dtype in (torch.float16, torch.bfloat16)
                     else (1 if "int8" in str(load_dtype) else 4), 1)
    n_params = sum(p.numel() for p in model.parameters())
    W_bytes = n_params * elem_bytes
    kv_per_tok = 2 * n_layers * n_kv * hd * elem_bytes

    # -- decode tok/s --
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    inp, past, n = ids, None, 0
    t0 = time.time()
    for _ in range(ngen):
        with torch.no_grad():
            out = model(inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1].argmax().item()
        if nxt == tok.eos_token_id:
            break
        inp = torch.tensor([[nxt]])
        n += 1
    dt = time.time() - t0
    tok_s = n / max(dt, 1e-9)
    bw = W_bytes / max(tok_s, 1e-9)

    # -- KV cache 字节 --
    ctx_trial = 1024
    base = tok(prompt, return_tensors="pt")["input_ids"]
    outkv = None
    while ctx_trial >= 64:
        reps = max(1, ctx_trial // base.shape[1])
        long_ids = base.repeat(1, reps)[:, :ctx_trial]
        try:
            with torch.no_grad():
                outkv = model(long_ids, use_cache=True)
            break
        except RuntimeError as e:
            if "memory" in str(e).lower():
                ctx_trial //= 2
                continue
            raise
    seq = base.shape[1] if outkv is None else outkv.past_key_values.get_seq_length()
    kv_bytes = kv_per_tok * int(seq)
    kv_w = kv_bytes / W_bytes

    # -- RSS --
    rss_gb = None
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1e9
    except Exception:
        pass

    # -- 真实激活 → 谱集中 --
    acts = _capture_acts(model, ids, kinds=("k_proj", "q_proj", "v_proj",
                                            "o_proj", "gate_proj", "up_proj",
                                            "down_proj"))
    k_acts = {n: h for n, h in acts.items() if n.endswith("k_proj")}
    sorted_names = sorted(k_acts, key=_layer_idx)
    gamma_layers = [spl_gamma(k_acts[n]) for n in sorted_names]
    proj_gamma = {}
    for kind in ("k_proj", "q_proj", "v_proj", "o_proj", "gate_proj",
                 "up_proj", "down_proj"):
        vals = [spl_gamma(h) for n, h in acts.items() if n.endswith(kind)]
        proj_gamma[kind] = float(np.mean(vals)) if vals else None

    # -- 曲率身份：真实轨迹逐点主曲率（替代伪造高斯对，消除硬编码 λ=0.8516） --
    ntraj = 96
    n_tr, layer_td = capture_trajectory(model, tok, prompt, ntraj)
    pairs = curvature_from_trajectory(layer_td, n_tr, n_frames=12,
                                      tok_per_frame=8, k_nn=15, n_sample=18)
    if pairs is None:
        pairs = curvature_from_acts(acts)          # 轻量 SVD 回退（仍为真实数据）
    curv: dict[str, Any]
    if pairs is None:
        curv = {"DEFF_plat": float(np.pi / 2), "K_neg_pct": 0.0, "H_median": 0.0,
                "phi_mean_deg": 0.0, "lambda_ratio": 0.0, "DEFF_cv": 0.0,
                "n_points": 0}
        curv["data_source"] = "placeholder"
    else:
        curv = curvature_metrics(pairs)
        curv["data_source"] = "real-trajectory"

    # -- AIQ（实时实取；f4=独立的全投影谱集中均值，非 f1 重复） --
    deff_cv = float(curv.get("DEFF_cv", 0.0))
    _proj_vals = [float(v) for v in proj_gamma.values()
                  if v is not None and isinstance(v, (int, float))]
    spl_all = (float(np.mean(_proj_vals)) if _proj_vals
               else float(np.mean(gamma_layers)))
    aiq = aiq_score(gamma_layers, deff_cv, curv["H_median"], spl_all)

    _log.info("真实测量完成 model=%s tok_s=%.2f AIQ=%.2f f3=%.3f f4=%.3f",
              _model_display(model_dir), float(tok_s),
              float(aiq.get("AIQ", 0.0)), float(aiq.get("f3", 0.0)),
              float(aiq.get("f4", 0.0)))
    return {
        "model": _model_display(model_dir),                 # 仅名称，不泄漏路径
        "arch": {"n_layers": n_layers, "n_kv": n_kv, "n_head": _arch["n_head"],
                 "hd": hd, "hidden": hidden, "n_params": int(n_params),
                 "W_bytes": int(W_bytes), "kv_per_tok": int(kv_per_tok),
                 "dtype": dtype},
        "engine": {"tok_s": float(tok_s), "kv_bytes": int(kv_bytes),
                   "kv_w_ratio": float(kv_w), "rss_gb": rss_gb,
                   "bandwidth_gbs": float(bw / 1e9),
                   "ngen_actual": int(n), "prompt_len": int(ids.shape[1])},
        "spectral": {"k_proj_gamma_layers": gamma_layers,
                     "k_proj_gamma_mean": float(np.mean(gamma_layers)),
                     "proj_gamma": proj_gamma},
        "curvature": curv, "aiq": aiq,
        "harness_version": "2.2",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------- 辅助
def _resolve_model_path(model_dir: str) -> str:
    """把模型名解析为本地目录；绝对路径原样返回（信任用户）。"""
    if os.path.isdir(model_dir):
        return model_dir
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(_BASE, "_models", model_dir)
    return cand


def _model_display(model_dir: str) -> str:
    """仅返回模型目录名（basename），绝不打印他人绝对路径。"""
    return os.path.basename(os.path.normpath(model_dir))


# ---------------------------------------------------------------- CLI
def _cli():
    ap = argparse.ArgumentParser(description="AIQ 真实模型测量引擎（无硬编码）")
    ap.add_argument("--model-dir", default="Qwen2.5-0.5B-Instruct",
                    help="模型目录/名称（默认 Qwen2.5-0.5B-Instruct）")
    ap.add_argument("--ngen", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16",
                    help="加载精度（默认 bfloat16，适配≤6G 内存沙箱）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认不写盘）")
    a = ap.parse_args()
    m = measure(a.model_dir, a.ngen, dtype=a.dtype)
    import json
    print("=" * 72)
    print("AIQ 真实模型测量完成")
    print(f"  模型: {m['model']}")
    print(f"  arch: {m['arch']['n_layers']}L {m['arch']['n_kv']}KV "
          f"{m['arch']['hd']}hd hidden={m['arch']['hidden']}")
    print(f"  Gamma 均值: {m['spectral']['k_proj_gamma_mean']:.4f}")
    print(f"  AIQ: {m['aiq']['AIQ']:.2f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        print(f"  已写入 {a.out}")
    print("=" * 72)


if __name__ == "__main__":
    _cli()
