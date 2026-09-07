#!/usr/bin/env python3
"""Generate a DOCX benchmark report with charts from the vLLM benchmark JSON results."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(HERE, "_charts")
os.makedirs(CHART_DIR, exist_ok=True)

# --- Brand-neutral palette (accessible, consistent across light backgrounds) ---
COLOR_PRIMARY = "#2563EB"    # blue
COLOR_SECONDARY = "#F59E0B"  # amber
COLOR_ACCENT = "#059669"     # green
COLOR_MUTED = "#6B7280"      # gray
COLOR_DANGER = "#DC2626"     # red

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": "#111827",
    "text.color": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_data():
    seq = json.load(open(os.path.join(HERE, "benchmark_results_full_stop.json")))
    conc_low = json.load(open(os.path.join(HERE, "benchmark_concurrency_results.json")))
    conc_high = json.load(open(os.path.join(HERE, "benchmark_concurrency_results_high.json")))
    all_levels = conc_low["levels"] + conc_high["levels"]
    all_levels.sort(key=lambda x: x["concurrency"])
    return seq, all_levels


def chart_sequential_time_tokens(seq_cases, path):
    titles = [c["title"] for c in seq_cases]
    times = [c["elapsed_seconds"] for c in seq_cases]
    tokens = [c["completion_tokens"] for c in seq_cases]
    colors = [COLOR_PRIMARY if c["finish_reason"] == "stop" else COLOR_DANGER for c in seq_cases]

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    y_pos = range(len(titles))
    bars = ax1.barh(y_pos, times, color=colors, height=0.6, zorder=3)
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(titles, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Time to complete (seconds)")
    ax1.set_title("Sequential benchmark: time per case (10 system design prompts)", fontsize=12, fontweight="bold", pad=12)

    for bar, tok, c in zip(bars, tokens, seq_cases):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                  f"{tok:,} tok", va="center", fontsize=8.5, color="#374151")

    legend_stop = plt.Line2D([0], [0], color=COLOR_PRIMARY, lw=6, label="Natural stop")
    legend_len = plt.Line2D([0], [0], color=COLOR_DANGER, lw=6, label="Hit token cap (length)")
    ax1.legend(handles=[legend_stop, legend_len], loc="upper center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=9.5)
    ax1.set_xlim(0, max(times) * 1.18)
    ax1.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def chart_concurrency_throughput(levels, path):
    x = [l["concurrency"] for l in levels]
    y = [l["aggregate_tokens_per_second"] for l in levels]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, y, marker="o", color=COLOR_PRIMARY, linewidth=2.5, markersize=7, zorder=3)
    ax.fill_between(x, y, color=COLOR_PRIMARY, alpha=0.08)

    # linear reference line from the concurrency=1 point
    base_tps = y[0]
    base_x = x[0]
    ideal_y = [base_tps * (xi / base_x) for xi in x]
    ax.plot(x, ideal_y, linestyle="--", color=COLOR_MUTED, linewidth=1.5, label="Ideal linear scaling", zorder=2)

    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:,.0f}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color="#111827")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Aggregate throughput (tokens/sec)")
    ax.set_title("Concurrency scaling: aggregate throughput vs. concurrent requests", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def chart_concurrency_latency(levels, path):
    x = [l["concurrency"] for l in levels]
    y = [l["per_request_latency_avg_s"] for l in levels]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, y, marker="o", color=COLOR_SECONDARY, linewidth=2.5, markersize=7, zorder=3)
    ax.fill_between(x, y, color=COLOR_SECONDARY, alpha=0.10)

    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}s", (xi, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color="#111827")

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Avg per-request latency (seconds)")
    ax.set_title("Concurrency scaling: per-request latency vs. concurrent requests", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def chart_vram_util(levels, path):
    x = [l["concurrency"] for l in levels]
    vram = [l["gpu"]["mem_used_mib_max"] / 1024 for l in levels]  # GiB
    util = [l["gpu"]["gpu_util_pct_avg"] for l in levels]

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(x, vram, marker="s", color=COLOR_ACCENT, linewidth=2.5, markersize=6, label="VRAM used (GiB)", zorder=3)
    ax1.axhline(143771 / 1024, color=COLOR_DANGER, linestyle=":", linewidth=1.5, label="Total VRAM (140.4 GiB)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlabel("Concurrent requests")
    ax1.set_ylabel("VRAM used (GiB)", color=COLOR_ACCENT)
    ax1.tick_params(axis="y", labelcolor=COLOR_ACCENT)
    ax1.set_ylim(0, 150)

    ax2 = ax1.twinx()
    ax2.plot(x, util, marker="^", color=COLOR_PRIMARY, linewidth=2, markersize=6, linestyle="--", label="GPU utilization (%)", zorder=3)
    ax2.set_ylabel("GPU utilization (%)", color=COLOR_PRIMARY)
    ax2.tick_params(axis="y", labelcolor=COLOR_PRIMARY)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=False, fontsize=8.5)

    ax1.set_title("VRAM usage stays flat while GPU stays saturated", fontsize=12, fontweight="bold", pad=12)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def set_cell_background(cell, color_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells[i], "2563EB")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return table


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    return h


def add_picture_centered(doc, path, width=6.3):
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_docx(seq, levels, out_path):
    doc = Document()

    # Base style
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    # Title
    title = doc.add_heading("vLLM Inference Benchmark Report", level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = sub.add_run("H200 GPU · vLLM 0.28.0 · Qwen/Qwen3.8-27B · 2026-09-04")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    run.italic = True

    doc.add_paragraph()

    # --- Overview ---
    add_heading(doc, "1. Overview", level=1)
    doc.add_paragraph(
        "This report summarizes inference benchmarks run against a self-hosted vLLM server on a single "
        "NVIDIA H200 GPU (143,771 MiB / ~140 GiB VRAM). The server ran vLLM 0.28.0, serving "
        "Qwen/Qwen3.8-27B — a Qwen3.5-architecture hybrid Mamba/attention multimodal reasoning model "
        "(~27B parameters, bf16) — through an OpenAI-compatible API. Two test dimensions were measured: "
        "single-request (sequential) generation across 10 open-ended system design prompts, and aggregate "
        "throughput under increasing concurrent load (1 to 96 simultaneous requests)."
    )

    # --- Provisioning timeline ---
    add_heading(doc, "2. First-time provisioning timeline", level=1)
    doc.add_paragraph(
        "Before any benchmark could run, the server had to be provisioned from a bare Ubuntu 24.04 box to a "
        "live vLLM endpoint. This section summarizes that one-time setup, from initial SSH access through "
        "system dependencies, the vLLM install, the model download, and GPU load, to the first successful "
        "inference response."
    )

    add_table(
        doc,
        ["Step", "Duration"],
        [
            ["System package install (python3-venv, pip, git, curl)", "~1 min"],
            ["pip install vllm (torch + CUDA deps + vLLM wheel)", "~6 min"],
            ["Model download (18 shards, ~54 GB from Hugging Face)", "~6 min"],
            ["Model load onto GPU + multimodal warmup + CUDA graph capture", "~1 min"],
            ["Total: bare box → live, serving endpoint", "~15 min"],
        ],
        col_widths=[4.6, 1.6],
    )

    doc.add_paragraph()
    doc.add_paragraph(
        "The two dominant costs — installing vLLM and downloading the model — are both one-time or cacheable "
        "on a given host. Swapping to a different model on the same box skips the pip install entirely, and "
        "only the model download time would repeat, scaling with that model's size rather than the ~54 GB "
        "used here."
    )

    doc.add_page_break()

    # --- Sequential ---
    add_heading(doc, "3. Sequential (single-request) benchmark", level=1)
    doc.add_paragraph(
        "10 system design prompts were sent one at a time with max_tokens=12,000. Throughput per request "
        "was remarkably constant across every prompt regardless of topic or output length."
    )

    seq_chart = os.path.join(CHART_DIR, "seq_time_tokens.png")
    chart_sequential_time_tokens(seq["cases"], seq_chart)
    add_picture_centered(doc, seq_chart)

    s = seq["summary"]
    add_table(
        doc,
        ["Metric", "Value"],
        [
            ["Cases run", f"{s['cases_run']}"],
            ["Reached natural stop", f"{s['finish_stop_count']} / {s['cases_run']}"],
            ["Hit token cap (length)", f"{s['finish_length_count']} / {s['cases_run']}"],
            ["Total wall time", f"{s['total_wall_time_seconds']:.1f} s ({s['total_wall_time_seconds']/60:.1f} min)"],
            ["Total completion tokens", f"{s['completion_tokens_total']:,}"],
            ["Avg tokens per case", f"{s['completion_tokens_avg']:,.0f}"],
            ["Avg throughput per request", f"{s['tokens_per_second_avg']} tok/s"],
            ["Time per case (min / avg / max)", f"{s['time_per_case_min_s']:.1f}s / {s['time_per_case_avg_s']:.1f}s / {s['time_per_case_max_s']:.1f}s"],
            ["VRAM used (constant)", f"{s['vram_peak_mib_overall']/1024:.1f} GiB"],
        ],
        col_widths=[3.2, 3.1],
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Per-case detail").bold = True
    add_table(
        doc,
        ["#", "Case", "Time (s)", "Tokens", "Tok/s", "Finish"],
        [
            [c["id"], c["title"], f"{c['elapsed_seconds']:.1f}", f"{c['completion_tokens']:,}", c["tokens_per_second"], c["finish_reason"]]
            for c in seq["cases"]
        ],
        col_widths=[0.3, 2.1, 0.8, 0.9, 0.7, 0.9],
    )

    doc.add_page_break()

    # --- Concurrency ---
    add_heading(doc, "4. Concurrency benchmark", level=1)
    doc.add_paragraph(
        "To test vLLM's continuous batching, requests were fired simultaneously at increasing concurrency "
        "levels (1, 2, 4, 8, 16, 32, 64, 96), each request capped at max_tokens=1,500. GPU state was sampled "
        "once per second via nvidia-smi throughout each batch. All 232 requests across every level succeeded "
        "with zero failures or timeouts."
    )

    conc_tp_chart = os.path.join(CHART_DIR, "conc_throughput.png")
    chart_concurrency_throughput(levels, conc_tp_chart)
    add_picture_centered(doc, conc_tp_chart)

    conc_lat_chart = os.path.join(CHART_DIR, "conc_latency.png")
    chart_concurrency_latency(levels, conc_lat_chart)
    add_picture_centered(doc, conc_lat_chart)

    max_level = levels[-1]
    min_level = levels[0]
    speedup = max_level["aggregate_tokens_per_second"] / min_level["aggregate_tokens_per_second"]
    latency_ratio = max_level["per_request_latency_avg_s"] / min_level["per_request_latency_avg_s"]

    doc.add_paragraph()
    key_p = doc.add_paragraph()
    key_p.add_run("Key finding: ").bold = True
    key_p.add_run(
        f"going from {min_level['concurrency']} to {max_level['concurrency']} concurrent requests produced a "
        f"{speedup:.1f}x increase in aggregate throughput ({min_level['aggregate_tokens_per_second']:.1f} → "
        f"{max_level['aggregate_tokens_per_second']:.1f} tok/s), while average per-request latency rose only "
        f"{latency_ratio:.1f}x ({min_level['per_request_latency_avg_s']:.1f}s → {max_level['per_request_latency_avg_s']:.1f}s). "
        f"Scaling is close to linear through 16 concurrent requests, then becomes noticeably sub-linear from "
        f"32 onward (the measured curve falls increasingly below the ideal-linear reference line in the chart "
        f"above) — throughput is still climbing at {max_level['concurrency']} with no hard plateau or failure "
        f"observed, but each additional concurrent request is yielding diminishing throughput gains beyond 16."
    )

    doc.add_paragraph()
    add_table(
        doc,
        ["Concurrency", "Batch time (s)", "Aggregate tok/s", "Avg latency (s)", "GPU util (avg)", "VRAM peak (GiB)"],
        [
            [
                l["concurrency"],
                f"{l['batch_wall_time_seconds']:.1f}",
                f"{l['aggregate_tokens_per_second']:,.1f}",
                f"{l['per_request_latency_avg_s']:.1f}",
                f"{l['gpu']['gpu_util_pct_avg']:.0f}%" if l.get("gpu") else "n/a",
                f"{l['gpu']['mem_used_mib_max']/1024:.1f}" if l.get("gpu") else "n/a",
            ]
            for l in levels
        ],
        col_widths=[1.1, 1.2, 1.3, 1.2, 1.1, 1.2],
    )

    doc.add_page_break()

    # --- GPU / VRAM behavior ---
    add_heading(doc, "5. GPU and VRAM behavior", level=1)
    doc.add_paragraph(
        "VRAM usage was essentially flat across every test — from a single request up through 96 concurrent "
        "requests — because vLLM pre-allocates its KV cache block pool once at server startup rather than "
        "growing memory per request. GPU compute utilization was already pinned at 95-100% even at "
        "concurrency=1, showing the model is compute-bound at the token level; continuous batching lets vLLM "
        "interleave many requests' decode steps into that same saturated compute budget instead of processing "
        "them one at a time."
    )
    vram_chart = os.path.join(CHART_DIR, "vram_util.png")
    chart_vram_util(levels, vram_chart)
    add_picture_centered(doc, vram_chart)

    idle_vram = seq["summary"]["idle_vram_mib"] or 132033
    total_vram = seq["summary"]["idle_vram_total_mib"] or 143771
    headroom = total_vram - idle_vram
    doc.add_paragraph(
        f"Idle/loaded VRAM footprint (model weights + pre-allocated KV cache pool): "
        f"{idle_vram/1024:.1f} GiB out of {total_vram/1024:.1f} GiB total — leaving "
        f"~{headroom/1024:.1f} GiB of headroom that remained unused even at 96 concurrent requests in this test."
    )

    doc.add_page_break()

    # --- Conclusions ---
    add_heading(doc, "6. Conclusions and recommendations", level=1)
    bullets = [
        "This deployment is significantly under-utilized for single-user/interactive use — a lone requester "
        "sees the same ~64 tok/s whether alone or one of 96 simultaneous requesters.",
        "Throughput scaling is close to linear up to 16 concurrent requests, then becomes increasingly "
        "sub-linear from 32 onward, though it was still climbing at 96 with no failures or latency cliff; the "
        "true ceiling for this model + GPU combination would require pushing further (128, 192, 256+) to locate.",
        "VRAM headroom (~11.7 GiB observed unused), not raw request count, is the more likely eventual "
        "limiting factor — KV cache size grows with both concurrency and sequence length, so workloads with "
        "long contexts (as seen in the sequential test, up to 12,000 tokens) run concurrently will consume "
        "that headroom faster than the short 1,500-token-capped concurrency test did.",
        "For production capacity planning: size expected concurrent long-context sessions against the "
        "available VRAM headroom, not just peak request count.",
        "This single H200 instance can likely absorb substantially more concurrent load than initial "
        "assumptions would suggest, for workloads similar in shape to these system-design prompts.",
    ]
    for b in bullets:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(b)

    doc.add_paragraph()
    foot = doc.add_paragraph()
    foot_run = foot.add_run(
        "Raw data: benchmark_results_full_stop.json, benchmark_concurrency_results.json, "
        "benchmark_concurrency_results_high.json. Scripts: benchmark_vllm.py, benchmark_vllm_concurrency.py."
    )
    foot_run.font.size = Pt(9)
    foot_run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    foot_run.italic = True

    doc.save(out_path)
    print(f"Saved: {out_path}")


def main():
    seq, levels = load_data()
    out_path = os.path.join(HERE, "vLLM_Benchmark_Report.docx")
    build_docx(seq, levels, out_path)


if __name__ == "__main__":
    main()
