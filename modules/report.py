"""
aiq-geometric-forensics.modules.report — 自包含 HTML 报告渲染
=====================================================================
把 diagnose()/trace()/compare() 产出的结构化 dict 渲染成：
  * 可打印文本以外的自包含 HTML（内联 CSS + SVG，零外部依赖）
  * 含"证据分级"徽标：真实测量(real-inline) > 真实存档(baseline) > 审计兜底(fallback)
  * 可组合成跨家族综合报告（多模型 Gamma/AIQ 条形图 + 深度剖面折线）

此模块只做渲染，不重新计算；所有数值来自上层传入的诊断 dict。
"""
from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Any

# ---------------------------------------------------------------- 证据分级
_SOURCE_LABEL = {
    "real-inline": "真实测量 (inline)",
    "baseline": "真实实测存档 (baseline)",
    "fallback": "审计兜底值 (fallback)",
    "none": "无数据",
}
_SOURCE_COLOR = {
    "real-inline": "#059669",   # 绿：最高置信
    "baseline": "#2563eb",      # 蓝：归档实测
    "fallback": "#b45309",      # 琥珀：兜底，需警惕
    "none": "#64748b",
}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def evidence_badge(source_code: str | None, label: str | None = None) -> str:
    """渲染证据分级徽标。source_code 为 'real-inline'/'baseline'/'fallback'。

    若未显式给出 source_code，则从数据源文案（label）反推：
      * 含 "inline" → real-inline；含 "基准存档"/"存档" → baseline；
      * 含 "兜底" → fallback；否则 none。
    """
    sc = source_code
    if sc is None and label:
        if "真实测量" in label or "inline" in label:
            sc = "real-inline"
        elif "存档" in label or "基准" in label:
            sc = "baseline"
        elif "兜底" in label:
            sc = "fallback"
        else:
            sc = "none"
    sc = sc if sc in _SOURCE_LABEL else "none"
    txt = label or _SOURCE_LABEL[sc]
    return (f'<span class="badge" style="background:{_SOURCE_COLOR[sc]}">'
            f"{_esc(txt)}</span>")


# ---------------------------------------------------------------- 内联 SVG
def svg_bars(items: list[tuple[str, float, str]], w: int = 900, h: int = 340,
             ymax: float | None = None, color_map: dict[str, str] | None = None,
             unit: str = "") -> str:
    """横向条形图。items=[(label, value, family)]；family 决定颜色。"""
    pad = 230
    n = len(items)
    gap_bottom = 40
    bw = (h - gap_bottom) / n
    barh = max(12, bw * 0.56)
    ymax = ymax or max(it[1] for it in items) * 1.15 or 1.0
    cm = color_map or {}

    grid = []
    for k in range(1, 5):
        x = pad + (k / 5) * (w - pad - 90)
        val = ymax * k / 5
        grid.append(
            f'<line x1="{x:.0f}" y1="12" x2="{x:.0f}" y2="{h-12}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{x:.0f}" y="{h-6}" font-size="10" fill="#94a3b8" '
            f'text-anchor="middle">{val:.3f}</text>')
    rows = []
    for i, (lbl, v, fam) in enumerate(items):
        y = 12 + i * bw + (bw - barh) / 2
        wv = (v / ymax) * (w - pad - 96)
        col = cm.get(fam, "#3b82f6")
        rows.append(
            f'<text x="6" y="{y+barh/2+4:.0f}" font-size="12" fill="#334155">'
            f"{_esc(lbl)}</text>"
            f'<rect x="{pad}" y="{y:.0f}" width="{wv:.1f}" height="{barh:.1f}" '
            f'rx="3" fill="{col}" opacity="0.9"/>'
            f'<text x="{pad+wv+6:.0f}" y="{y+barh/2+4:.0f}" font-size="11.5" '
            f'fill="#0f172a" font-weight="600">{v:.3f}{_esc(unit)}</text>')
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="inherit">'
            + "".join(grid) + "".join(rows) + "</svg>")


def svg_lines(sets: Sequence[tuple[str, Sequence[float], str]],
              w: int = 900, h: int = 320, ymin: float = 0.0,
              ymax: float = 0.85) -> str:
    """折线图（逐层 Gamma 剖面等）。sets=[(label, ys, color)]。"""
    padL, padR, padT, padB = 48, 18, 16, 36
    maxL = max(len(g) for _, g, _ in sets)
    def X(i: int) -> float:
        return padL + i / max(1, maxL - 1) * (w - padL - padR)
    def Y(v: float) -> float:
        return padT + (ymax - v) / max(ymax - ymin, 1e-9) * (h - padT - padB)
    grid = []
    for k in range(5):
        v = ymin + (ymax - ymin) * k / 4
        y = Y(v)
        grid.append(f'<line x1="{padL}" y1="{y:.0f}" x2="{w-padR}" y2="{y:.0f}" '
                    f'stroke="#eef2f7" stroke-width="1"/>'
                    f'<text x="{padL-6}" y="{y+4:.0f}" font-size="10" '
                    f'fill="#94a3b8" text-anchor="end">{v:.2f}</text>')
    for i in range(maxL):
        if i % 5 == 0 or i == maxL - 1:
            grid.append(f'<text x="{X(i):.0f}" y="{h-10}" font-size="10" '
                        f'fill="#94a3b8" text-anchor="middle">{i+1}</text>')
    polys = []
    for lbl, ys, col in sets:
        pts = " ".join(f"{X(i):.1f},{Y(vv):.1f}" for i, vv in enumerate(ys))
        last = Y(ys[-1])
        polys.append(
            f'<polyline points="{pts}" fill="none" stroke="{col}" '
            f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<text x="{X(len(ys)-1)+4:.0f}" y="{last+4:.0f}" font-size="11" '
            f'fill="{col}" font-weight="600">{_esc(lbl)}</text>')
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="inherit">'
            + "".join(grid) + "".join(polys) + "</svg>")


# ---------------------------------------------------------------- 因子判别度诊断
_FNAME = ("f1 芯坍缩", "f2 DEFF稳定", "f3 H收敛", "f4 SPL集中", "f5 低模纯度")


def _grade(span: float, rel: float) -> tuple[str, str]:
    """按跨模型跨度/相对变异给因子判"判别力"级别与配色。"""
    if span < 0.012:                                  # 近乎恒定 → 基线项，无判别力
        return "基线·近恒定(判别力低)", "#b45309"
    if rel >= 0.15:
        return "高判别", "#059669"
    if rel >= 0.06:
        return "中判别", "#2563eb"
    return "低判别", "#64748b"


def render_factor_panel(names: Sequence[str],
                        F: Sequence[Sequence[float]],
                        fnames: Sequence[str] = _FNAME) -> str:
    """跨模型逐因子判别度诊断表。

    输入：
      names: 模型名列表
      F:     n_model × n_factor 的因子值矩阵（来自各模型本轮实时复算的 f1..f5）
    输出：
      自包含 HTML 表格。逐因子展示跨模型 spread（span/相对变异）与判别力分级，
      把"f2 近恒定、对排序零贡献"这类诚实信息显式呈现，而非藏在权重里。
    """
    rows = list(zip(*F, strict=True))                 # rows[j] = 该因子在各模型的值
    thead = "".join(f"<th>{_esc(n)}</th>" for n in names)
    out = [f'<div class="card"><h2>因子判别度诊断（跨模型）</h2>'
           f'<table><thead><tr><th>因子</th>{thead}<th>跨度</th>'
           f'<th>相对变异</th><th>判别力</th></tr></thead><tbody>']
    for j, vals in enumerate(rows):
        vo = [float(v) for v in vals]
        span = max(vo) - min(vo)
        rel = span / (sum(vo) / len(vo)) if vo else 0.0
        grade, col = _grade(span, rel)
        cells = "".join(f"<td>{v:.3f}</td>" for v in vo)
        out.append(
            f'<tr><td><b>{_esc(fnames[j])}</b></td>{cells}'
            f'<td>{span:.4f}</td><td>{rel*100:.1f}%</td>'
            f'<td><span style="color:{col};font-weight:600">{_esc(grade)}</span></td></tr>')
    out.append('</tbody></table>'
               '<p class="note">判别力=该因子在基准库跨模型的数值跨度/相对变异。'
               '判别力低的因子仍维恒 AIQ 基线，但不参与模型排序；'
               '本表诚实呈现各因子对"区分模型"的实际贡献，不做暗改。</p></div>')
    return "".join(out)


# ---------------------------------------------------------------- 健康报告
def render_health(rep: dict[str, Any], source_code: str | None = None) -> str:
    """健康诊断 → HTML 卡片。"""
    a = rep["aiq"]
    d = rep["depth"]
    c = rep["curvature"]
    g = rep["gamma"]
    com = rep["compressibility"]
    hs = rep["health"]

    badge = evidence_badge(source_code, rep.get("data_source"))
    facts = (
        f"<tr><td>AIQ 几何智能商</td><td><b>{a['AIQ']:.1f}</b> / 100</td></tr>"
        f"<tr><td>五因子 f1..f5</td><td>{a['f1']:.3f} · {a['f2']:.3f} · "
        f"{a['f3']:.3f} · {a['f4']:.3f} · {a['f5']:.3f}</td></tr>"
        f"<tr><td>Gamma 均值(min/max)</td><td>{g['mean']:.4f} "
        f"({g['min']:.4f} / {g['max']:.4f})</td></tr>"
        f"<tr><td>深度剖面 浅/中/深</td><td>{d['shallow']:.3f} / "
        f"{d['mid']:.3f} / {d['deep']:.3f} → {d['pattern']}</td></tr>"
        f"<tr><td>DEFF 平台</td><td>{c['DEFF_plat']:.4f} "
        f"(偏差 {c['deff_dev_pct']:+.2f}%) 锚 π/2</td></tr>"
        f"<tr><td>曲率身份 K&lt;0% / Hmed / φmean / λ</td><td>"
        f"{c['K_neg_pct']:.2f}% / {c['H_median']:.4f} / {c['phi_mean_deg']:.2f}° "
        f"/ {c['lambda_ratio']:.4f}</td></tr>")

    judge = ""
    for tag, ok, detail in hs["judgements"]:
        mark = "✅" if ok else "❌"
        judge += (f'<li class="{"pass" if ok else "fail"}"><b>{mark} {_esc(tag)}'
                  f"</b> — {_esc(detail)}</li>")

    compr = ""
    if com.get("high") or com.get("mid") or com.get("low"):
        def fmt(dct):
            return " · ".join(f"{k}({v:.3f})" for k, v in dct.items())
        compr = (
            f'<div class="compr"><div class="ct">可压缩性评估</div>'
            f'<span class="chip h">高可压 γ≥{com["thresholds"]["high"]}: '
            f"{_esc(fmt(com['high']) or '无')} → 激进剪枝 0.25–0.4</span>"
            f'<span class="chip m">中可压 : {_esc(fmt(com["mid"]) or "无")} '
            f"→ 温和压缩 0.5–0.6</span>"
            f'<span class="chip l">难压缩 γ&lt;{com["thresholds"]["low"]}: '
            f"{_esc(fmt(com['low']) or '无')} → 保持全精度</span></div>")

    return f"""
<div class="card">
  <h2>几何健康度 · {_esc(rep['model'])} {badge}</h2>
  <table><tbody>{facts}</tbody></table>
  <div class="sect">健康判定</div><ul class="verdicts">{judge}</ul>
  {compr}
</div>"""


# ---------------------------------------------------------------- 溯源报告
def render_trace(rep: dict[str, Any]) -> str:
    r = rep
    t_badge = evidence_badge(None, r.get("data_source_target"))
    rf_badge = evidence_badge(None, r.get("data_source_ref"))
    same = r["same_family"]
    return f"""
<div class="card">
  <h2>家族溯源 · {_esc(r['target'])} vs {_esc(r['reference'])}</h2>
  <div class="trace">{t_badge} → {rf_badge}</div>
  <table><tbody>
    <tr><td>γ (谱集中均值)</td><td>target={r['gamma_target']:.4f} · ref={r['gamma_ref']:.4f}</td></tr>
    <tr><td>指纹均值差 |Δμ|</td><td>{r['d_gamma_mean']:.4f} (阈值 0.01)</td></tr>
    <tr><td>MAD 最近邻</td><td>{_esc(r['nearest_family'])} (MAD={r['mad_distance']:.4f}, 边距={r['margin']:.4f})</td></tr>
    <tr><td>SFT 稳定性 / 同族</td><td>{'指纹稳定(微调不改身份)' if same else '指纹变异'} / {'✓ 同族' if same else '✗ 异族'}</td></tr>
    <tr><td>判定</td><td>{_esc(r['verdict'])}</td></tr>
  </tbody></table>
</div>"""


# ---------------------------------------------------------------- 对比报告
def render_compare(rep: dict[str, Any]) -> str:
    r = rep
    m = r["metric"]
    t_badge = evidence_badge(None, r.get("data_source_target"))
    rf_badge = evidence_badge(None, r.get("data_source_ref"))
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(round(v, 4))}</td></tr>"
        for k, v in m.items())
    return f"""
<div class="card">
  <h2>基准对比 · {_esc(r['target'])} vs {_esc(r['reference'])}</h2>
  <div class="trace">{t_badge} → {rf_badge}</div>
  <table><tbody>{rows}</tbody></table>
  <p class="note">同架构: {'✓' if r['same_architecture'] else '✗'} · 同家族: {'✓' if r['same_family'] else '✗'} · 判定: {_esc(r['verdict'])}</p>
</div>"""


# ---------------------------------------------------------------- 整页组装
_CSS = """
* { box-sizing:border-box; }
body { margin:0; background:#f1f5f9; color:#0f172a;
  font:15px/1.65 "Noto Sans CJK SC","WenQuanYi Micro Hei","Microsoft YaHei",
  "Segoe UI",sans-serif; }
.wrap { max-width:1040px; margin:0 auto; padding:28px 20px 70px; }
.hero { background:linear-gradient(135deg,#0f172a,#1e3a8a); color:#fff;
  border-radius:16px; padding:30px 34px; margin-bottom:22px; }
.hero h1 { margin:0 0 6px; font-size:25px; }
.hero p { margin:3px 0; color:#c7d2fe; font-size:14px; }
.card { background:#fff; border:1px solid #e2e8f0; border-radius:14px;
  padding:18px 22px; margin:18px 0; }
.card h2 { margin:0 0 12px; font-size:18px; border-bottom:2px solid #e2e8f0;
  padding-bottom:8px; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th,td { padding:7px 8px; text-align:right; border-bottom:1px solid #eef2f7;
  white-space:nowrap; }
td:first-child { text-align:left; font-weight:600; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px;
  font-size:11.5px; color:#fff; font-weight:600; margin-left:6px; }
.trace { margin-bottom:8px; }
.verdicts { list-style:none; padding:0; margin:6px 0 0; }
.verdicts li { padding:6px 10px; border-radius:8px; margin:4px 0; }
.verdicts .pass { background:#f0fdf4; }
.verdicts .fail { background:#fef2f2; }
.sect { font-size:13px; font-weight:700; color:#475569; margin-top:10px; }
.compr { margin-top:10px; display:flex; flex-direction:column; gap:6px; }
.chip { padding:7px 10px; border-radius:8px; font-size:12.5px; }
.chip.h { background:#ecfdf5; color:#065f46; }
.chip.m { background:#eff6ff; color:#1e40af; }
.chip.l { background:#fef3c7; color:#92400e; }
.compr .ct { font-size:13px; font-weight:700; color:#475569; }
.note { color:#334155; font-size:13px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media(max-width:820px){ .grid2 { grid-template-columns:1fr; } }
"""


def render_full_page(title: str, subtitle: str,
                     body_parts: Iterable[str]) -> str:
    """把一组 HTML 卡片组装为自包含完整页面。"""
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<div class="hero"><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p></div>
{''.join(body_parts)}
</div></body></html>"""


# ---------------------------------------------------------------- 溯源鲁棒性 / 家族族谱渲染
def render_stability(rep: dict[str, Any]) -> str:
    """把溯源鲁棒性报告渲染为自包含 HTML 卡片。

    含逐扰动类型 × 量档的归属存活率表 + 位移监控 + 判定；
    存活判据为"扰动后归属认回自身"（位移仅监控，不否决——见 stability docstring）。
    """
    o = rep["overall"]
    body = [f'<div class="card"><h2>溯源鲁棒性 · {_esc(rep["target"])}</h2>'
            f'<p class="sect">总体归属存活率 <b>{o["survival_rate"]*100:.0f}%</b>'
            f'（阈值 ≥90% 判强鲁棒）｜参照库 {len(rep["library"])} 模型｜'
            f'每格 {rep["trials_per_cell"]} 次试验（seed={rep["seed"]} 可复现）'
            f'｜全局最大位移 {o["max_displacement"]:.4f}</p>']
    for kind, blk in rep["perturbations"].items():
        label = {"sft": "微调微扰(σ·N)", "trunc": "层谱截断",
                 "scale": "幅度改写"}[kind]
        rows = "".join(
            f"<tr><td>{r['amount']:.3f}</td><td>{r['survival']*100:.0f}%</td>"
            f"<td>{r['mean_displacement']:.4f}</td>"
            f"<td>{r['max_displacement']:.4f}</td></tr>"
            for r in blk["rows"])
        body.append(
            f'<table><thead><tr><th colspan="4">{_esc(label)}'
            f'（综合存活 {blk["overall"]*100:.0f}%）</th></tr>'
            f'<tr><th>量档</th><th>归属存活</th><th>位移μ</th>'
            f'<th>位移max</th></tr></thead><tbody>{rows}</tbody></table>')
    body.append(f'<p class="note"><b>判定</b>：{_esc(o["verdict"])}</p></div>')
    return "".join(body)


def render_family(rep: dict[str, Any]) -> str:
    """把家族族谱报告渲染为自包含 HTML 卡片（亲缘表 + 凝聚簇）。"""
    kbody = "".join(
        f"<tr><td>{_esc(n)}</td><td>{_esc(k['nearest'])}</td>"
        f"<td>{k['nn_distance']:.4f}</td><td>{_esc(k['relation'])}</td>"
        f"<td>簇{k['cluster']}</td></tr>" for n, k in rep["kinship"].items())
    cbody = "".join(
        f"<div class=\"chip h\"><b>簇{cid}</b>: {_esc(', '.join(m))}</div>"
        for cid, m in rep["clusters"].items())
    t = rep["thresholds"]
    return (f'<div class="card"><h2>家族族谱（指纹亲缘聚类）</h2>'
            f'<p class="sect">模型数 {len(rep["models"])}｜凝聚簇 {rep["n_clusters"]}｜'
            f'同族&lt;{t.get("同族(SFT级)")}｜近亲&lt;{t.get("近亲(扩参/变体)")}</p>'
            f'<table><thead><tr><th>模型</th><th>最近邻</th><th>距离(MAD)</th>'
            f'<th>亲缘</th><th>簇</th></tr></thead><tbody>{kbody}</tbody></table>'
            f'<div class="compr">{cbody}</div>'
            f'<p class="note">距离口径=指纹剖面段 MAD（详见 schema.FAM_*）；'
            f'聚类阈值保守（仅 SFT 级同族凝聚），亲缘表含完整近亲/远亲视角。</p></div>')


def render_params(rep: dict[str, Any]) -> str:
    """把参数验证体系批量结果渲染为自包含 HTML 卡片（59 参数验证报告）。

    含：按组聚合表（A-T 九组）、性能统计、逐项明细。
    """
    o = rep
    perf = o.get("performance", {})
    groups = o.get("groups", {})
    # 按组聚合表
    grow = "".join(
        f"<tr><td><b>{_esc(g)}</b></td><td>{v['passed']}/{v['total']}</td>"
        f"<td>{v['seconds']}s</td>"
        f"<td>{'✓' if v['failed'] == 0 else '✗ ' + str(v['failed'])}</td></tr>"
        for g, v in groups.items())
    # 逐项明细（失败/全过）
    detail = "".join(
        f"<tr><td>{_esc(r['name'])}</td><td>{r['group']}</td>"
        f"<td style='color:{'#059669' if r['ok'] else '#dc2626'};font-weight:600'>"
        f"{'PASS' if r['ok'] else 'FAIL'}</td><td>{r['seconds']}s</td></tr>"
        for r in o["results"])
    rate = (o["passed"] / o["total"] * 100) if o["total"] else 0
    return (
        f'<div class="card"><h2>参数验证体系（params/ 59 参数）</h2>'
        f'<p class="sect">通过率 <b>{rate:.0f}%</b>（{o["passed"]}/{o["total"]}）｜'
        f'总耗时 {perf.get("total_seconds", 0)}s｜均 {perf.get("mean_seconds", 0)}s｜'
        f'最慢 {_esc(str(perf.get("slowest", "—")))}（{perf.get("slowest_seconds", 0)}s）</p>'
        f'<table><thead><tr><th>组</th><th>通过</th><th>耗时</th><th>状态</th></tr>'
        f'</thead><tbody>{grow}</tbody></table>'
        f'<p class="sect" style="margin-top:12px">逐项明细</p>'
        f'<table><thead><tr><th>参数</th><th>组</th><th>结果</th><th>耗时</th></tr>'
        f'</thead><tbody>{detail}</tbody></table>'
        f'<p class="note">每个参数 verify.py 以子进程独立运行（零侵入）；'
        f'断言/容差/真实数据对照见各参数 README 与 _params_data.json。</p></div>')
