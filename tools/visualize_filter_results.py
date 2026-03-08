"""
Visualize filter_stats.json and filtered sample images from tracer parquet files.

Usage:
    python tools/visualize_filter_results.py --trace_dir <work_dir>/trace --output filter_report.html
"""

import argparse
import base64
import json
import os

import numpy as np
import pandas as pd


# Map op name keywords to the __dj__stats__ key most relevant for that filter
OP_STAT_HINTS = {
    "image_text_matching": "image_text_similarity",
    "image_text_similarity": "image_text_similarity",
    "image_aesthetics": "image_aesthetics_scores",
    "image_blurriness": "image_blurriness_scores",
    "image_brightness": "image_brightness_scores",
    "image_entropy": "image_entropy_scores",
    "image_aspect_ratio": "aspect_ratios",
    "image_shape": ["image_width", "image_height"],
    "image_size": "image_sizes",
    "image_border_variance": "image_border_variance_scores",
    "image_nsfw": "image_nsfw_scores",
    "image_watermark": "image_watermark_scores",
    "image_safe_aigc": "image_safe_aigc_scores",
    "perplexity": "perplexity",
    "character_repetition": "char_rep_ratio",
    "flagged_words": "flagged_words_ratio",
    "special_characters": "special_char_ratio",
}


def get_stat_value(stats_dict, op_name):
    """Extract the most relevant stat value for display given an op name."""
    if not isinstance(stats_dict, dict):
        return None
    for key, stat_key in OP_STAT_HINTS.items():
        if key in op_name:
            if isinstance(stat_key, list):
                parts = []
                for sk in stat_key:
                    v = stats_dict.get(sk)
                    if v is not None:
                        parts.append(f"{sk}: {_format_stat(v)}")
                return ", ".join(parts) if parts else None
            else:
                v = stats_dict.get(stat_key)
                if v is not None:
                    return f"{stat_key}: {_format_stat(v)}"
    return None


def _format_stat(v):
    if isinstance(v, (np.ndarray, list)):
        vals = list(v) if isinstance(v, np.ndarray) else v
        if len(vals) == 1:
            return f"{vals[0]:.4f}" if isinstance(vals[0], float) else str(vals[0])
        return "[" + ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in vals[:3]) + "]"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def extract_image_bytes(row):
    """Extract first image bytes from image_bytes column (numpy array of bytes)."""
    val = row.get("image_bytes")
    if val is None:
        return None
    # numpy array of bytes objects
    if isinstance(val, np.ndarray):
        if len(val) == 0:
            return None
        raw = val[0]
        return bytes(raw) if not isinstance(raw, bytes) else raw
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, list) and val:
        raw = val[0]
        return bytes(raw) if not isinstance(raw, bytes) else raw
    return None


def bytes_to_img_tag(raw):
    if raw is None:
        return None
    if raw[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif raw[:2] == b'\xff\xd8':
        mime = "image/jpeg"
    elif raw[:4] == b'GIF8':
        mime = "image/gif"
    elif raw[:4] == b'RIFF' and len(raw) > 12 and raw[8:12] == b'WEBP':
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f'<img src="data:{mime};base64,{b64}">'


def load_filter_parquets(trace_dir):
    result = {}
    if not os.path.isdir(trace_dir):
        return result
    for fname in sorted(os.listdir(trace_dir)):
        if fname.startswith("filter-") and fname.endswith(".parquet"):
            op_name = fname[len("filter-"):-len(".parquet")]
            try:
                result[op_name] = pd.read_parquet(os.path.join(trace_dir, fname))
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    return result


def render_card(row, op_name, idx):
    raw = extract_image_bytes(row)
    img_tag = bytes_to_img_tag(raw)
    img_html = img_tag if img_tag else "<div class='no-img'>no image</div>"

    # Text caption
    text = row.get("text") or row.get("caption") or ""
    text_html = f"<div class='card-text'>{str(text)[:200]}</div>" if text else ""

    # Relevant stat
    stat_html = ""
    stats = row.get("__dj__stats__")
    stat_val = get_stat_value(stats, op_name)
    if stat_val:
        stat_html = f"<div class='card-stat'>{stat_val}</div>"

    return f"""<div class="card">
  <div class="card-img">{img_html}</div>
  {text_html}
  {stat_html}
</div>"""


def build_op_panel(op_name, entry, df, max_samples):
    filtered = entry.get("filtered", 0)
    before = entry.get("before", 0)
    kept_ratio = entry.get("kept_ratio", 1.0)
    time_s = entry.get("time_s", "-")

    stats_html = f"""<div class="op-stats">
  <span class="stat-item s-filtered">Filtered: {filtered:,} ({(1-kept_ratio)*100:.1f}%)</span>
  <span class="stat-item s-kept">Kept: {entry.get('after',0):,} ({kept_ratio*100:.1f}%)</span>
  <span class="stat-item">Before: {before:,}</span>
  <span class="stat-item">Time: {time_s}s</span>
</div>"""

    if df is None or len(df) == 0:
        body = "<p class='no-samples'>No trace parquet found or no filtered samples recorded.</p>"
    else:
        cards = [render_card(row, op_name, i) for i, (_, row) in enumerate(df.head(max_samples).iterrows())]
        body = f"<div class='cards'>{''.join(cards)}</div>"

    return f"""<div class="op-panel" id="panel-{op_name}" style="display:none">
  <h2>{op_name}</h2>
  {stats_html}
  {body}
</div>"""


def build_html(trace_dir, max_samples=20):
    stats_path = os.path.join(trace_dir, "filter_stats.json")
    filter_stats = []
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            filter_stats = json.load(f)
    else:
        print(f"Warning: filter_stats.json not found in {trace_dir}")

    filter_parquets = load_filter_parquets(trace_dir)

    summary_rows = []
    for entry in filter_stats:
        op = entry["op"]
        kept_ratio = entry.get("kept_ratio", 1.0)
        has_panel = "has-panel" if filter_parquets.get(op) is not None else ""
        summary_rows.append(f"""<tr class="summary-row {has_panel}" onclick="showPanel('{op}')" data-op="{op}">
  <td>{op}</td>
  <td>{entry['before']:,}</td>
  <td>{entry['after']:,}</td>
  <td class="filtered">{entry['filtered']:,} ({(1-kept_ratio)*100:.1f}%)</td>
  <td class="kept">{kept_ratio*100:.1f}%</td>
  <td>{entry.get('time_s','-')}s</td>
</tr>""")

    panels = [build_op_panel(op, entry, filter_parquets.get(op), max_samples)
              for entry in filter_stats for op in [entry["op"]]]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data-Juicer Filter Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f2f5; color: #222; }}

  header {{ padding: 16px 28px; background: #1a1a2e; color: #fff; font-size: 20px; font-weight: 700; letter-spacing: 0.02em; }}

  .layout {{ display: flex; height: calc(100vh - 52px); }}

  /* Sidebar */
  .sidebar {{ width: 480px; min-width: 360px; background: #fff; border-right: 1px solid #dde; overflow-y: auto; padding: 16px; }}
  .sidebar-title {{ font-size: 11px; font-weight: 700; color: #999; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  th {{ background: #f7f7f9; padding: 7px 9px; text-align: left; color: #666; font-weight: 600; border-bottom: 2px solid #eee; white-space: nowrap; }}
  td {{ padding: 7px 9px; border-bottom: 1px solid #f2f2f2; white-space: nowrap; }}
  .summary-row {{ cursor: default; transition: background 0.1s; }}
  .summary-row.has-panel {{ cursor: pointer; }}
  .summary-row.has-panel:hover td:first-child {{ text-decoration: underline; }}
  .summary-row.has-panel:hover {{ background: #f4f6ff; }}
  .summary-row.active {{ background: #e8eeff !important; }}
  .filtered {{ color: #c0392b; font-weight: 600; }}
  .kept {{ color: #27ae60; font-weight: 600; }}

  /* Panel area */
  .panel-area {{ flex: 1; overflow-y: auto; padding: 28px 32px; }}
  .placeholder {{ color: #bbb; text-align: center; margin-top: 100px; font-size: 15px; }}

  .op-panel h2 {{ font-size: 18px; font-weight: 700; color: #1a1a2e; margin-bottom: 14px; word-break: break-all; }}
  .op-stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }}
  .stat-item {{ background: #f7f7f9; border: 1px solid #e2e2e2; border-radius: 6px; padding: 5px 13px; font-size: 13px; }}
  .stat-item.s-filtered {{ background: #fff0ee; border-color: #f5c6c0; color: #c0392b; font-weight: 600; }}
  .stat-item.s-kept {{ background: #eefaf3; border-color: #b2e0c6; color: #27ae60; font-weight: 600; }}

  /* Cards */
  .cards {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .card {{ background: #fff; border: 1px solid #e4e4e4; border-radius: 8px; padding: 10px; width: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: box-shadow 0.15s; }}
  .card:hover {{ box-shadow: 0 4px 14px rgba(0,0,0,0.10); }}
  .card-img {{ text-align: center; margin-bottom: 8px; background: #fafafa; border-radius: 4px; min-height: 36px; display:flex; align-items:center; justify-content:center; }}
  .card-img img {{ max-width: 180px; max-height: 180px; object-fit: contain; border-radius: 4px; display: block; }}
  .no-img {{ color: #ccc; font-size: 11px; padding: 16px; }}
  .card-text {{ font-size: 11px; color: #555; margin-bottom: 5px; line-height: 1.4; word-break: break-word; }}
  .card-stat {{ font-size: 11px; font-weight: 700; color: #c0392b; background: #fff5f4; border-radius: 4px; padding: 3px 7px; word-break: break-all; }}
  .no-samples {{ color: #aaa; font-style: italic; margin-top: 16px; }}
</style>
</head>
<body>
<header>Data-Juicer Filter Report</header>
<div class="layout">
  <div class="sidebar">
    <div class="sidebar-title">Filter Summary — click to inspect</div>
    <table>
      <thead><tr><th>Operator</th><th>Before</th><th>After</th><th>Filtered</th><th>Kept%</th><th>Time</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
  </div>
  <div class="panel-area" id="panel-area">
    <div class="placeholder" id="placeholder">← Select a filter to view filtered samples</div>
    {''.join(panels)}
  </div>
</div>
<script>
  function showPanel(op) {{
    document.getElementById('placeholder').style.display = 'none';
    document.querySelectorAll('.op-panel').forEach(p => p.style.display = 'none');
    document.querySelectorAll('.summary-row').forEach(r => r.classList.remove('active'));
    var panel = document.getElementById('panel-' + op);
    if (panel) panel.style.display = 'block';
    var row = document.querySelector('[data-op="' + op + '"]');
    if (row) row.classList.add('active');
    document.getElementById('panel-area').scrollTop = 0;
  }}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize Data-Juicer filter results as HTML.")
    parser.add_argument("--trace_dir", required=True, help="Path to the trace directory")
    parser.add_argument("--output", default="filter_report.html", help="Output HTML file path")
    parser.add_argument("--max_samples", type=int, default=20, help="Max filtered samples to show per op")
    args = parser.parse_args()

    print(f"Loading from: {args.trace_dir}")
    html = build_html(args.trace_dir, max_samples=args.max_samples)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
