"""
aiq-geometric-forensics.modules.cli — 统一命令行入口
=====================================================================
一个入口覆盖全部 6 大动作，并支持结构化 JSON / 自包含 HTML 报告导出：

  python -m modules.cli list                         列出基准库
  python -m modules.cli health [NAME] [--measure DIR] 健康体检（可实时测量）
  python -m modules.cli trace TARGET REF              家族溯源
  python -m modules.cli compare TARGET REF            基准对比
  python -m modules.cli measure DIR [--out JSON]      真实测量（需 torch）
  python -m modules.cli report [--html FILE]         跨家族综合报告

全局选项（放子命令之后）：
  --json FILE   将主结果序列化为 JSON
  --html FILE   导出自包含 HTML 报告
退出码：0 成功；2 用法/运行错误。measure 需 torch+transformers（延迟加载）。
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from typing import Any

from . import observation, report, schema
from .compare import compare, format_compare
from .data import Metrics, find_baseline_models
from .forensics import format_trace, trace
from .health import diagnose, format_health

DEFAULT_REF = "Qwen2.5-0.5B-Instruct"

_log = logging.getLogger("aiq.cli")

# 跨家族展示用的家族配色（与基准库家族归类对应）
_FAM_COLOR = {
    "Qwen": "#3b82f6", "Qwen(SFT)": "#60a5fa", "Qwen(扩参)": "#93c5fd",
    "Llama": "#f59e0b", "BLOOM": "#10b981",
}


def _fam(name: str) -> str:
    n = name.lower()
    if "tinyllama" in n:
        return "Llama"
    if "bloom" in n:
        return "BLOOM"
    if "instruct" in n:
        return "Qwen(SFT)" if "0.5" in n else "Qwen(扩参)"
    return "Qwen"


# ---------------- 序列化 / 写盘 ----------------
def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- 子命令：list ----------------
def cmd_list(cfg: argparse.Namespace) -> int:
    models = find_baseline_models()
    rows = []
    for name in models:
        m = Metrics(name)
        a = m.arch()
        rows.append({
            "model": name,
            "source": m.source,
            "n_layers": a.get("n_layers"), "hidden": a.get("hidden"),
            "gamma": m.get_float("spectral.k_proj_gamma_mean", 0.0),
            "aiq": m.get_float("aiq.AIQ", 0.0),
        })
    if cfg.json:
        _write_json(cfg.json, {"models": rows})
    if cfg.html:
        cards = ("<div class='card'><h2>基准库</h2><table><thead><tr>"
                 "<th>模型</th><th>层数</th><th>Hidden</th><th>Γ</th>"
                 "<th>AIQ</th><th>数据源</th></tr></thead><tbody>" + "".join(
                     f"<tr><td>{n['model']}</td><td>{n['n_layers']}</td>"
                     f"<td>{n['hidden']}</td><td>{n['gamma']:.4f}</td>"
                     f"<td>{n['aiq']:.2f}</td><td>{n['source']}</td></tr>"
                     for n in rows) + "</tbody></table></div>")
        _write_text(cfg.html, report.render_full_page(
            "基准库一览", f"{len(rows)} 个已标定模型", [cards]))
    print(f"{'模型':<28}{'层数':>5}{'Hidden':>8}{'Γ':>9}{'AIQ':>8}  数据源")
    for n in rows:
        print(f"{n['model']:<28}{str(n['n_layers']):>5}"
              f"{str(n['hidden']):>8}{n['gamma']:>9.4f}{n['aiq']:>8.2f}  {n['source']}")
    return 0


# ---------------- 子命令：health ----------------
def cmd_health(cfg: argparse.Namespace) -> int:
    name = cfg.name
    source_code = "baseline"
    if cfg.measure:
        from .harness import measure  # 仅真实测量才加载 torch
        measurement = measure(cfg.measure, ngen=cfg.ngen, dtype=cfg.dtype)
        rep = diagnose(name, measurement=measurement)
        source_code = "real-inline"
    else:
        rep = diagnose(name)
        m = Metrics(name)
        source_code = m.source
    if cfg.json:
        _write_json(cfg.json, rep)
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            f"几何健康度 · {name}", "AIQ / Gamma / 曲率身份 / 可压缩性",
            [report.render_health(rep, source_code)]))
    print(format_health(rep))
    return 0


# ---------------- 子命令：trace ----------------
def cmd_trace(cfg: argparse.Namespace) -> int:
    rep = trace(target=cfg.target, reference=cfg.reference)
    if cfg.json:
        _write_json(cfg.json, rep)
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            f"家族溯源 · {cfg.target} vs {cfg.reference}", "几何指纹血缘判定",
            [report.render_trace(rep)]))
    print(format_trace(rep))
    return 0


# ---------------- 子命令：compare ----------------
def cmd_compare(cfg: argparse.Namespace) -> int:
    rep = compare(target=cfg.target, reference=cfg.reference)
    if cfg.json:
        _write_json(cfg.json, rep)
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            f"基准对比 · {cfg.target} vs {cfg.reference}", "逐维几何与工程偏差",
            [report.render_compare(rep)]))
    print(format_compare(rep))
    return 0


# ---------------- 子命令：stability（溯源鲁棒性）----------------
def cmd_stability(cfg: argparse.Namespace) -> int:
    from .stability import format_stability, stability  # 本地导入（保持启动轻量）

    rep = stability(target=cfg.target)
    if cfg.json:
        _write_json(cfg.json, rep)
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            f"溯源鲁棒性 · {cfg.target}",
            "扰动下指纹归属存活率（SFT/截断/缩放）",
            [report.render_stability(rep)]))
    print(format_stability(rep))
    return 0


# ---------------- 子命令：family（家族族谱）----------------
def cmd_family(cfg: argparse.Namespace) -> int:
    from .family import family_tree, format_family  # 本地导入（保持启动轻量）

    rep = family_tree()
    if cfg.json:
        _write_json(cfg.json, {
            "models": rep["models"], "n_clusters": rep["n_clusters"],
            "kinship": rep["kinship"], "clusters": rep["clusters"],
            "merges": rep["merges"], "thresholds": rep["thresholds"],
        })
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            "家族族谱（指纹亲缘聚类）",
            f"{len(rep['models'])} 个模型 · 剖面 MAD 亲缘",
            [report.render_family(rep)]))
    print(format_family(rep))
    return 0


# ---------------- 子命令：measure ----------------
def cmd_measure(cfg: argparse.Namespace) -> int:
    from .harness import measure  # 延迟加载 torch
    m = measure(cfg.dir, ngen=cfg.ngen, dtype=cfg.dtype)
    if cfg.out:
        _write_json(cfg.out, m)
        print(f"已写入测量结果: {cfg.out}")
    else:
        print(json.dumps(m, ensure_ascii=False, indent=2))
    return 0


# ---------------- 子命令：report（跨家族综合） ----------------
def cmd_report(cfg: argparse.Namespace) -> int:
    models = find_baseline_models()
    family = [_fam(n) for n in models]
    gamma = [Metrics(n).get_float("spectral.k_proj_gamma_mean", 0.0)
             for n in models]
    aiq = [Metrics(n).get_float("aiq.AIQ", 0.0) for n in models]

    bars_gamma = report.svg_bars(list(zip(models, gamma, family, strict=True)), w=920, h=360,
                                 color_map=_FAM_COLOR)
    bars_aiq = report.svg_bars(list(zip(models, aiq, family, strict=True)), w=920, h=360,
                               ymax=schema.AIQ_YMAX, color_map=_FAM_COLOR)
    # 深度剖面折线（逐层 Gamma）
    lines: list[tuple[str, Sequence[float], str]] = []
    for n in models:
        m = Metrics(n)
        layers = m.get_list("spectral.k_proj_gamma_layers")
        if layers:
            lines.append((n, layers, _FAM_COLOR[_fam(n)]))
    depth = report.svg_lines(lines, w=920, h=340) if lines else ""

    # 因子判别度诊断：运行时用诊断结果实时复算 f1..f5，而非采信存库旧权重的隐性假设
    factor_rows = []
    for n in models:
        a = diagnose(n)["aiq"]
        factor_rows.append([a["f1"], a["f2"], a["f3"], a["f4"], a["f5"]])
    factor_panel = report.render_factor_panel(models, factor_rows)

    cards = [
        '<div class="card"><h2>Γ · 谱集中度横向对比</h2>' + bars_gamma + "</div>",
        '<div class="card"><h2>AIQ · 几何智能商横向对比</h2>' + bars_aiq + "</div>",
    ]
    if depth:
        cards.insert(1, '<div class="card"><h2>Gamma 深度剖面（逐层）</h2>'
                        + depth + "</div>")
    cards.append(factor_panel)
    for n in models:
        scroll = Metrics(n)
        cards.append(report.render_health(diagnose(n), scroll.source))

    summary = [{"model": n, "gamma": g, "aiq": a, "family": f}
               for n, g, a, f in zip(models, gamma, aiq, family, strict=True)]
    if cfg.json:
        _write_json(cfg.json, summary)
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            "跨家族几何指纹综合报告",
            f"{len(models)} 个已标定模型 · 100% 真实实测 / 存档", cards))
    print(f"{'模型':<28}{'Γ':>9}{'AIQ':>8}  家族")
    for n, g, a, f in zip(models, gamma, aiq, family, strict=True):
        print(f"{n:<28}{g:>9.4f}{a:>8.2f}  {f}")
    return 0


# ---------------- 子命令：verify（参数验证体系统一入口）----------------
def cmd_verify(cfg: argparse.Namespace) -> int:
    from . import params_runner as PR  # 统一收口模块（零侵入子进程执行）

    if cfg.name:                        # 单参数验证
        res = PR.verify_one(cfg.name, timeout=cfg.timeout)
        mark = "PASS" if res.ok else "FAIL"
        print(f"[{mark}] {res.name}  exit={res.exit_code}  {res.seconds:.1f}s")
        return 0 if res.ok else 1
    # 全量 / 限数冒烟
    rep = PR.verify_all(limit=cfg.limit, timeout=cfg.timeout)
    print(PR.format_summary(rep))
    if cfg.json:
        _write_json(cfg.json, rep)
    if cfg.html:
        _write_text(cfg.html, report.render_full_page(
            "参数验证体系（59 参数）",
            f"通过率 {rep['passed']}/{rep['total']}｜总耗时 {rep['seconds']:.1f}s",
            [report.render_params(rep)]))
    return 0 if rep["ok"] else 1


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------- 子命令：selfcheck（生产完整性门禁） ----------------
def cmd_selfcheck(cfg: argparse.Namespace) -> int:
    from .selfcheck import render_report, run_selfcheck

    rep = run_selfcheck()
    if cfg.json:
        _write_json(cfg.json, rep)
    if cfg.html:
        # 自检结果渲染为 HTML（复用全页外壳）
        card = ('<div class="card"><h2>自检门禁</h2><table><thead><tr>'
                '<th>检查项</th><th>结果</th><th>说明</th></tr></thead><tbody>' + "".join(
                    f"<tr><td>{c['name']}</td><td>{'✅' if c['ok'] else '❌'}</td>"
                    f"<td>{c['detail']}</td></tr>" for c in rep["checks"])
                + "</tbody></table></div>")
        _write_text(cfg.html, report.render_full_page(
            "生产完整性自检",
            f"版本 {rep['version'].get('__init__','?')} · "
            f"PASS={rep['summary']['passed']} FAIL={rep['summary']['failed']}",
            [card]))
    render_report(rep)
    return 0 if rep["ok"] else 2


# ---------------- 主解析 ----------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modules.cli",
        description="AIQ 几何指纹诊断包 · 统一命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    _s = sub.add_parser("list", help="列出基准库")
    _s.set_defaults(fn=cmd_list)

    s = sub.add_parser("health", help="几何健康体检")
    s.add_argument("name", nargs="?", default=DEFAULT_REF,
                   help=f"模型名（默认 {DEFAULT_REF}）")
    s.add_argument("--measure", default=None,
                   help="本地模型目录；提供则先实时测量（P0 优先）")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--ngen", type=int, default=64)
    s.set_defaults(fn=cmd_health)

    s = sub.add_parser("trace", help="家族溯源判定")
    s.add_argument("target")
    s.add_argument("reference", nargs="?", default=DEFAULT_REF)
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("compare", help="与基准模型对比")
    s.add_argument("target")
    s.add_argument("reference", nargs="?", default=DEFAULT_REF)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("stability", help="溯源鲁棒性（扰动下指纹存活）")
    s.add_argument("target", nargs="?", default="Qwen2.5-0.5B",
                   help="待检模型名（默认 Qwen2.5-0.5B）")
    s.set_defaults(fn=cmd_stability)

    s = sub.add_parser("family", help="家族族谱（指纹亲缘聚类）")
    s.set_defaults(fn=cmd_family)

    s = sub.add_parser("verify", help="参数验证体系（params/ 59 参数）")
    s.add_argument("name", nargs="?", default=None,
                   help="单个参数目录名（如 A01_model）；缺省=批量全部")
    s.add_argument("--limit", type=int, default=None,
                   help="批量时只跑前 N 个参数（快速冒烟）")
    s.add_argument("--timeout", type=int, default=120,
                   help="单脚本超时秒数（默认 120）")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("measure", help="对本地模型真实测量")
    s.add_argument("dir", help="HF 模型本地目录")
    s.add_argument("--out", default=None, help="测量结果 JSON 路径")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--ngen", type=int, default=64)
    s.set_defaults(fn=cmd_measure)

    s = sub.add_parser("report", help="跨家族综合报告")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("selfcheck", help="生产完整性门禁（自检）")
    s.set_defaults(fn=cmd_selfcheck)

    for _sp in sub.choices.values():                 # 全局选项对所有子命令生效
        _sp.add_argument("--json", default=None, metavar="FILE", help="主结果写 JSON")
        _sp.add_argument("--html", default=None, metavar="FILE", help="导出自包含 HTML 报告")
        _sp.add_argument("--verbose", action="store_true", help="开启 DEBUG 级可观测日志")
        _sp.add_argument("--log", default=None, metavar="FILE",
                         help="可观测日志落盘（人类可读）")
        _sp.add_argument("--json-log", default=None, metavar="FILE",
                         help="结构化日志落盘（JSONL，CI 可解析）")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    cfg = parser.parse_args(argv)
    # 可观测层：进程级日志配置（JSONL / 文本 / 级别）
    level = logging.DEBUG if getattr(cfg, "verbose", False) else logging.INFO
    log_file = getattr(cfg, "json_log", None) or getattr(cfg, "log", None)
    observation.setup_logging(level=level, file=log_file,
                              json=bool(getattr(cfg, "json_log", None)),
                              stream=None, force=True)
    _log.info("CLI 开始 subcommand=%s run_id=%s",
              getattr(cfg, "cmd", "?"), observation.trace_id())
    try:
        rc = int(cfg.fn(cfg))
        _log.info("CLI 完成 subcommand=%s exit=%d", getattr(cfg, "cmd", "?"), rc)
        return rc
    except KeyboardInterrupt:
        _log.warning("CLI 被用户中断")
        return 130
    except Exception as e:                            # 统一错误出口，退出码 2
        _log.exception("命令执行失败: %s (%s)", type(e).__name__, e)
        return 2


def sys_stderr():
    import sys
    return sys.stderr


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
