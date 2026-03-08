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


def to_bytes(val):
    """Coerce a value to bytes if possible, else return None."""
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, memoryview):
        return bytes(val)
    if isinstance(val, np.ndarray):
        return val.tobytes()
    # list of ints (byte array stored as list)
    if isinstance(val, list) and val and isinstance(val[0], int):
        return bytes(val)
    return None


def bytes_to_base64_img(raw):
    """Convert raw bytes to an <img> HTML tag with embedded base64 data."""
    if raw is None:
        return None
    if raw[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif raw[:2] == b'\xff\xd8':
        mime = "image/jpeg"
    elif raw[:4] == b'GIF8':
        mime = "image/gif"
    elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    return f'<img src="data:{mime};base64,{b64}">'


def find_image_bytes_column(df):
    """Find the column most likely containing binary image data."""
    # Prefer columns named image_bytes, then any bytes column
    preferred = ["image_bytes", "image", "img", "bytes"]
    candidates = []
    for col in df.columns:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if to_bytes(sample) is not None:
            candidates.append(col)
    for p in preferred:
        if p in candidates:
            return p
    return candidates[0] if candidates else None


def is_binary_col(series):
    sample = series.dropna().iloc[0] if not series.dropna().empty else None
    return to_bytes(sample) is not None


def load_filter_parquets(trace_dir):
    result = {}
    if not os.path.isdir(trace_dir):
        return result
    for fname in sorted(os.listdir(trace_dir)):
        if fname.startswith("filter-") and fname.endswith(".parquet"):
            op_name = fname[len("filter-"):-len(".parquet")]
            path = os.path.join(trace_dir, fname)
            try:
                result[op_name] = pd.read_parquet(path)
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    return result


def render_sample_card(row, img_col, meta_cols, idx):
    """Render a single sample card as HTML."""
    img_html = "<div class='no-img'>no image</div>"
    if img_col:
        raw = to_bytes(row.get(img_col))
        tag = bytes_to_base64_img(raw)
        if tag:
            img_html = tag

    meta_html = ""
    for col in meta_cols:
        val = row.get(col)
        if val is not None and str(val).strip():
            meta_html += f"<div><span class='col-name'>{col}:</span> <span class='col-val'>{str(val)[:300]}</span></div>"

    return f"""<div class="card" id="card-{idx}">
  <div class="card-img">{img_html}</div>
  <div class="card-meta">{meta_html}</div>
</div>"""


def build_op_panel(op_name, entry, df, max_samples):
    filtered = entry.get("filtered", 0)
    before = entry.get("before", 0)
    kept_ratio = entry.get("kept_ratio", 1.0)
    time_s = entry.get("time_s", "-")

    stats_html = f"""<div class="op-stats">
      <span class="stat-item filtered">Filtered: {filtered:,} ({(1-kept_ratio)*100:.1f}%)</span>
      <span class="stat-item kept">Kept: {entry.get('after',0):,} ({kept_ratio*100:.1f}%)</span>
      <span class="stat-item">Before: {before:,}</span>
      <span class="stat-item">Time: {time_s}s</span>
    </div>"""

    if df is None or len(df) == 0:
        return f"""<div class="op-panel" id="panel-{op_name}" style="display:none">
      <h2>{op_name}</h2>{stats_html}
      <p class="no-samples">No trace parquet found or no filtered samples.</p>
    </div>"""

    img_col = find_image_bytes_column(df)
    meta_cols = [c for c in df.columns
                 if c != img_col and not is_binary_col(df[c])]
    # exclude very long / unhelpful columns
    meta_cols = [c for c in meta_cols if c not in ("__dj__stats__",)][:8]

    sample_df = df.head(max_samples)
    cards = [render_sample_card(row, img_col, meta_cols, f"{op_name}-{i}")
             for i, (_, row) in enumerate(sample_df.iterrows())]

    return f"""<div class="op-panel" id="panel-{op_name}" style="display:none">
  <h2>{op_name}</h2>
  {stats_html}
  <div class="cards">{''.join(cards)}</div>
</div>"""


def build_html(trace_dir, max_samples=10):
    stats_path = os.path.join(trace_dir, "filter_stats.json")
    filter_stats = []
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            filter_stats = json.load(f)
    else:
        print(f"Warning: filter_stats.json not found in {trace_dir}")

    filter_parquets = load_filter_parquets(trace_dir)

    # Summary table rows (clickable)
    summary_rows = []
    for entry in filter_stats:
        op = entry["op"]
        kept_ratio = entry.get("kept_ratio", 1.0)
        has_panel = "has-panel" if (filter_parquets.get(op) is not None and len(filter_parquets.get(op, [])) > 0) else ""
        summary_rows.append(f"""<tr class="summary-row {has_panel}" onclick="showPanel('{op}')" data-op="{op}">
          <td>{op}</td>
          <td>{entry['before']:,}</td>
          <td>{entry['after']:,}</td>
          <td class="filtered">{entry['filtered']:,} ({(1-kept_ratio)*100:.1f}%)</td>
          <td class="kept">{kept_ratio*100:.1f}%</td>
          <td>{entry.get('time_s','-')}s</td>
        </tr>""")

    # Op panels
    panels = []
    for entry in filter_stats:
        op = entry["op"]
        panels.append(build_op_panel(op, entry, filter_parquets.get(op), max_samples))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data-Juicer Filter Report</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f0f2f5; color: #222; }}
  h1 {{ margin: 0; padding: 20px 32px; background: #1a1a2e; color: #fff; font-size: 22px; }}
  .layout {{ display: flex; height: calc(100vh - 60px); }}

  /* Left sidebar */
  .sidebar {{ width: 420px; min-width: 320px; background: #fff; border-right: 1px solid #dde; overflow-y: auto; padding: 16px; }}
  .sidebar h2 {{ font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 12px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th {{ background: #f7f7f9; padding: 8px 10px; text-align: left; color: #555; font-weight: 600; border-bottom: 2px solid #eee; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #f0f0f0; }}
  .summary-row {{ cursor: default; }}
  .summary-row.has-panel {{ cursor: pointer; }}
  .summary-row.has-panel:hover {{ background: #f0f4ff; }}
  .summary-row.active {{ background: #e8efff; font-weight: 600; }}
  .filtered {{ color: #c0392b; font-weight: 600; }}
  .kept {{ color: #27ae60; font-weight: 600; }}

  /* Right panel */
  .panel-area {{ flex: 1; overflow-y: auto; padding: 24px 32px; }}
  .op-panel h2 {{ font-size: 20px; margin: 0 0 12px; color: #1a1a2e; }}
  .op-stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
  .stat-item {{ background: #f7f7f9; border: 1px solid #e0e0e0; border-radius: 6px; padding: 6px 14px; font-size: 13px; }}
  .stat-item.filtered {{ background: #fff0ee; border-color: #f5c6c0; color: #c0392b; }}
  .stat-item.kept {{ background: #eefaf3; border-color: #b2e0c6; color: #27ae60; }}

  /* Cards */
  .cards {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .card {{ background: #fff; border: 1px solid #e4e4e4; border-radius: 8px; padding: 12px; width: 220px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); transition: box-shadow 0.2s; }}
  .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
  .card-img {{ text-align: center; margin-bottom: 10px; min-height: 40px; }}
  .card-img img {{ max-width: 196px; max-height: 196px; object-fit: contain; border-radius: 4px; }}
  .no-img {{ color: #bbb; font-size: 12px; padding: 20px 0; }}
  .card-meta {{ font-size: 11px; color: #666; word-break: break-word; }}
  .col-name {{ font-weight: 700; color: #444; }}
  .col-val {{ color: #555; }}
  .card-meta div {{ margin-bottom: 4px; }}

  .no-samples {{ color: #999; font-style: italic; }}
  .placeholder {{ color: #aaa; text-align: center; margin-top: 80px; font-size: 16px; }}
</style>
</head>
<body>
<h1>Data-Juicer Filter Report</h1>
<div class="layout">
  <div class="sidebar">
    <h2>Filter Summary</h2>
    <table>
      <thead><tr><th>Operator</th><th>Before</th><th>After</th><th>Filtered</th><th>Kept</th><th>Time</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
  </div>
  <div class="panel-area" id="panel-area">
    <div class="placeholder" id="placeholder">← Click a filter to see filtered samples</div>
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
    parser.add_argument("--trace_dir", required=True, help="Path to the trace directory (contains filter_stats.json and filter-*.parquet)")
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
