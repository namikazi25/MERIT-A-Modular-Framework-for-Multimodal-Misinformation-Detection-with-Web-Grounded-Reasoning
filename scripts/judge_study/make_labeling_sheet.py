"""Task E: build the Phase 3 labeling sheet (dev only).

- 150 samples from dev, stratified across fake_cls x J1-correct/incorrect.
- Includes every dev sample where J2 flipped J1's correct verdict to wrong,
  with BOTH J1 and J2 labels + rationales visible on those rows.
- No label column pre-filled; empty evidence_sufficient (Y/N) and notes columns.
- Evidence snippets grouped by question (truncated).
- Outputs labeling/phase3_sheet.csv + .html (thumbnails embedded) + manifest.

Usage:
  python -m scripts.judge_study.make_labeling_sheet --dev-rationales results/judge_study/dev_rationales.jsonl       --split manifests/split-500-v1.json --out labeling/phase3_sheet
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())

POS = "Misinformation"


def thumb_data_url(path: str, max_w: int = 240) -> str:
    try:
        from PIL import Image
        import io
        im = Image.open(path)
        im.thumbnail((max_w, max_w))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def evidence_html(bundle: dict) -> str:
    docs = bundle.get("documents") or {}
    parts = []
    for q, items in docs.items():
        parts.append(f"<b>Q:</b> {html.escape(str(q))}<br>")
        if not items:
            parts.append("&nbsp;&nbsp;<i>(no documents)</i><br>")
        for d in items[:5]:
            desc = html.escape((d.get("description") or "")[:350])
            parts.append(f'&nbsp;&nbsp;<a href="{html.escape(d.get("url") or "#")}">{html.escape(d.get("title") or "")}</a> '
                         f'<span style="color:#666">{desc}...</span><br>')
    return "".join(parts) if parts else "(no evidence)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-rationales", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="labeling/phase3_sheet")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    sp = json.load(open(args.split))
    bundle_by_path = {d["image_path"]: d["bundle"] for d in sp["dev"] + sp["holdout"]}

    recs = [json.loads(l) for l in open(args.dev_rationales)]
    dev_set = {d["image_path"] for d in sp["dev"]}
    recs = [r for r in recs if r["image_path"] in dev_set]
    print(f"dev rationales: {len(recs)}")

    def j1_correct(r):
        return r["J1"]["label"] == (POS if str(r["gt"]) == "Fake" else "Not Misinformation")

    flipped = [r for r in recs if j1_correct(r) and r["J2"]["label"] != (POS if str(r["gt"]) == "Fake" else "Not Misinformation")]
    print(f"J2-flipped (J1 correct -> J2 wrong) in dev: {len(flipped)}")

    import random
    rng = random.Random(args.seed)
    selected = list(flipped)
    selected_ids = {r["image_path"] for r in selected}
    pool = [r for r in recs if r["image_path"] not in selected_ids]
    # stratified fill: fake_cls x J1-correct/incorrect
    cells = {}
    for r in pool:
        key = (r["fake_cls"], j1_correct(r))
        cells.setdefault(key, []).append(r)
    for key in sorted(cells):
        rng.shuffle(cells[key])
    n_needed = args.n - len(selected)
    quota = {}
    total_pool = len(pool)
    for key, lst in cells.items():
        quota[key] = round(len(lst) / total_pool * n_needed)
    filled = []
    for key, q in sorted(quota.items()):
        filled.extend(cells[key][:q])
    # top up if rounding left gaps (filled may already reach n_needed)
    remaining = [r for r in pool if r not in filled]
    rng.shuffle(remaining)
    if len(filled) < n_needed:
        filled.extend(remaining[: n_needed - len(filled)])
    selected.extend(filled[:n_needed])
    selected = selected[: args.n]
    rng.shuffle(selected)
    print(f"final sheet size: {len(selected)} (flipped included: {sum(1 for r in selected if r in flipped)})")
    from collections import Counter
    print("strata:", dict(Counter((r["fake_cls"], j1_correct(r)) for r in selected)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # CSV
    csv_path = args.out + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet_row", "sample_id", "image_path", "claim", "fake_cls", "gt",
                    "J1_label", "J1_conf", "J1_rationale", "J2_label", "J2_conf", "J2_rationale",
                    "evidence_sufficient", "notes"])
        for i, r in enumerate(selected, start=1):
            is_flip = r in flipped
            w.writerow([i, r["sample_id"], r["image_path"], r.get("claim", ""), r["fake_cls"], r["gt"],
                        r["J1"]["label"] if is_flip else "", r["J1"]["confidence"] if is_flip else "",
                        (r["J1"]["rationale"] or "") if is_flip else "",
                        r["J2"]["label"] if is_flip else "", r["J2"]["confidence"] if is_flip else "",
                        (r["J2"]["rationale"] or "") if is_flip else "",
                        "", ""])
    # HTML
    rows_html = []
    for i, r in enumerate(selected, start=1):
        b = json.load(open(bundle_by_path[r["image_path"]]))
        is_flip = r in flipped
        thumb = thumb_data_url(b["image_path"])
        j1cell = (f"<b>{html.escape(str(r['J1']['label']))}</b> ({r['J1']['confidence']})<br>"
                  f"<i>{html.escape((r['J1']['rationale'] or '')[:400])}</i>") if is_flip else ""
        j2cell = (f"<b>{html.escape(str(r['J2']['label']))}</b> ({r['J2']['confidence']})<br>"
                  f"<i>{html.escape((r['J2']['rationale'] or '')[:400])}</i>") if is_flip else ""
        img_tag = f'<img src="{thumb}" width=180>' if thumb else html.escape(b["image_path"])
        rows_html.append(f"<tr><td>{i}</td><td>{html.escape(str(r['sample_id']))}</td>"
                         f"<td>{html.escape((b.get('claim') or '')[:200])}</td>"
                         f"<td>{img_tag}</td>"
                         f"<td>{html.escape(str(r['fake_cls']))}</td>"
                         f"<td>{j1cell}</td><td>{j2cell}</td>"
                         f"<td>{evidence_html(b)}</td>"
                         f"<td></td><td></td></tr>")
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Phase 3 labeling sheet</title>
<style>table{{border-collapse:collapse;font-family:helvetica,arial;font-size:12px}}td,th{{border:1px solid #999;padding:6px;vertical-align:top}}
th{{background:#eee}} .flip{{background:#fff3cd}}</style></head><body>
<h2>Phase 3 labeling sheet (dev only, n={len(selected)}; <span style="color:#b58900">yellow rows = J2-flipped, both rationales shown</span>)</h2>
<p>Columns: <b>evidence_sufficient</b> (Y/N): is the retrieved evidence sufficient to determine whether the claim is true or false?
<b>notes</b>: anything relevant. Do not label the overall veracity.</p>
<table><tr><th>#</th><th>id</th><th>claim</th><th>image</th><th>fake_cls</th><th>J1 (flipped only)</th><th>J2 (flipped only)</th><th>evidence by question</th><th>evidence_sufficient</th><th>notes</th></tr>
{''.join(rows_html)}</table></body></html>"""
    html_path = args.out + ".html"
    open(html_path, "w").write(page)

    manifest = {
        "task": "E", "n": len(selected), "seed": args.seed,
        "flipped_included": sum(1 for r in selected if r in flipped),
        "dev_only": all(r["image_path"] in dev_set for r in selected),
        "sample_list_hash": hashlib.sha256("\n".join(sorted(r["image_path"] for r in selected)).encode()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(manifest, open(args.out + ".manifest.json", "w"), indent=1)
    print("sheet written:", csv_path, html_path)
    print("manifest:", args.out + ".manifest.json")


if __name__ == "__main__":
    main()
