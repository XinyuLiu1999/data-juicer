"""
Visualize filter_stats.json and filtered sample images from tracer parquet files.

Usage:
    python tools/visualize_filter_results.py --tracer_dir <work_dir>/tracer --output filter_report.html
"""

import argparse
import base64
import io
import json
import os

import pandas as pd


def image_to_base64(img_bytes):
    if img_bytes is None:
        return None
    if isinstance(img_bytes, memoryview):
        img_bytes = bytes(img_bytes)
    return base64.b64encode(img_bytes).decode("utf-8")


def find_image_bytes_column(df):
    """Find the column containing binary image data."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, (bytes, memoryview)):
                return col
    return None


def load_filter_parquets(tracer_dir):
    """Load all filter-*.parquet files, return dict of op_name -> DataFrame."""
    result = {}
    for fname in sorted(os.listdir(tracer_dir)):
        if fname.startswith("filter-") and fname.endswith(".parquet"):
            op_name = fname[len("filter-"):-len(".parquet")]
            path = os.path.join(tracer_dir, fname)
            try:
                df = pd.read_parquet(path)
                result[op_name] = df
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    return result


def render_image_cell(img_bytes, max_width=200):
    if img_bytes is None:
        return "<td><em>no image</em></td>"
    b64 = image_to_base64(img_bytes)
    if b64 is None:
        return "<td><em>no image</em></td>"
    # Try to detect format from magic bytes
    raw = img_bytes if isinstance(img_bytes, bytes) else bytes(img_bytes)
    if raw[:4] == b'\x89PNG':
        mime = "image/png"
    elif raw[:2] in (b'\xff\xd8',):
        mime = "image/jpeg"
    elif raw[:4] == b'GIF8':
        mime = "image/gif"
    elif raw[:4] == b'RIFF':
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f'<td><img src="data:{mime};base64,{b64}" style="max-width:{max_width}px;max-height:{max_width}px;object-fit:contain;"></td>'


def build_html(tracer_dir, max_samples=10):
    # Load filter_stats.json
    stats_path = os.path.join(tracer_dir, "filter_stats.json")
    filter_stats = []
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            filter_stats = json.load(f)
    else:
        print(f"Warning: filter_stats.json not found in {tracer_dir}")

    # Load filter parquet files
    filter_parquets = load_filter_parquets(tracer_dir)

    # Build HTML
    sections = []

    # --- Summary table ---
    if filter_stats:
        rows = []
        for entry in filter_stats:
            kept_pct = f"{entry.get('kept_ratio', 1.0) * 100:.1f}%"
            filtered_pct = f"{(1 - entry.get('kept_ratio', 1.0)) * 100:.1f}%"
            rows.append(f"""
            <tr>
                <td>{entry['op']}</td>
                <td>{entry['before']:,}</td>
                <td>{entry['after']:,}</td>
                <td class="filtered">{entry['filtered']:,} ({filtered_pct})</td>
                <td class="kept">{kept_pct}</td>
                <td>{entry.get('time_s', '-')}s</td>
            </tr>""")

        sections.append(f"""
        <section>
            <h2>Filter Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Operator</th>
                        <th>Before</th>
                        <th>After</th>
                        <th>Filtered</th>
                        <th>Kept</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </section>""")

    # --- Per-op filtered samples ---
    for entry in filter_stats:
        op_name = entry['op']
        df = filter_parquets.get(op_name)

        img_col = find_image_bytes_column(df) if df is not None else None
        text_cols = []
        if df is not None:
            text_cols = [c for c in df.columns
                         if c != img_col
                         and df[c].dtype == object
                         and not isinstance(df[c].dropna().iloc[0] if not df[c].dropna().empty else None, (bytes, memoryview, dict, list))]

        header = f"""
        <section>
            <h2>{op_name}</h2>
            <p>Filtered <strong>{entry['filtered']:,}</strong> samples
               ({(1 - entry.get('kept_ratio', 1.0)) * 100:.1f}% of {entry['before']:,})</p>"""

        if df is None or len(df) == 0:
            sections.append(header + "<p><em>No trace parquet found or no filtered samples.</em></p></section>")
            continue

        sample_df = df.head(max_samples)

        # Build sample cards
        cards = []
        for _, row in sample_df.iterrows():
            img_html = ""
            if img_col:
                img_html = render_image_cell(row.get(img_col)).replace("<td>", "").replace("</td>", "")

            meta_items = ""
            for col in text_cols[:6]:
                val = row.get(col, "")
                if val is not None:
                    meta_items += f"<div><span class='col-name'>{col}:</span> {str(val)[:200]}</div>"

            cards.append(f"""
            <div class="card">
                <div class="card-img">{img_html}</div>
                <div class="card-meta">{meta_items}</div>
            </div>""")

        sections.append(header + f"""
            <div class="cards">{''.join(cards)}</div>
        </section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data-Juicer Filter Report</title>
<style>
    body {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; color: #222; }}
    h1 {{ color: #333; }}
    h2 {{ color: #444; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 40px; }}
    section {{ background: white; padding: 24px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #f0f0f0; font-weight: 600; }}
    tr:hover {{ background: #fafafa; }}
    .filtered {{ color: #c0392b; font-weight: 600; }}
    .kept {{ color: #27ae60; font-weight: 600; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; }}
    .card {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; width: 240px; }}
    .card-img {{ text-align: center; margin-bottom: 8px; }}
    .card-img img {{ max-width: 200px; max-height: 200px; object-fit: contain; border-radius: 4px; }}
    .card-meta {{ font-size: 12px; color: #555; word-break: break-word; }}
    .col-name {{ font-weight: 600; color: #333; }}
</style>
</head>
<body>
<h1>Data-Juicer Filter Report</h1>
{''.join(sections)}
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Visualize Data-Juicer filter results as HTML.")
    parser.add_argument("--tracer_dir", required=True, help="Path to the trace directory (contains filter_stats.json and filter-*.parquet)")
    parser.add_argument("--output", default="filter_report.html", help="Output HTML file path")
    parser.add_argument("--max_samples", type=int, default=10, help="Max filtered samples to show per op")
    args = parser.parse_args()

    print(f"Loading from: {args.tracer_dir}")
    html = build_html(args.tracer_dir, max_samples=args.max_samples)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
