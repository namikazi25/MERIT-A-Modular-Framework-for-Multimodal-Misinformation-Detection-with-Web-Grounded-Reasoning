from __future__ import annotations

"""
Generate an HTML report from pipeline outputs.

Two entry points:
  1) render_html_report(objs, metrics, output_path): called from main.py after a run
  2) CLI mode: read JSONL of outputs and render a report, with optional metrics computed via scripts.evaluate

Sections per sample:
  - Headline and image
  - Collapsible: Relevancy
  - Collapsible: Visual Veracity
  - Collapsible: Q/A (best per chain)
  - Collapsible: Final Judgement
Top section shows metrics if provided.
"""

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv

from scripts.utils.dataset import load_dataset_records, normalize_record_path
from scripts.utils.media import image_to_data_url


def _relpath(path: str, base: Optional[str]) -> str:
    try:
        if not base:
            return path
        return os.path.relpath(path, start=os.path.dirname(base))
    except Exception:
        return path


def _escape(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def _bool_from_any(x: Any) -> Optional[bool]:
    """Best-effort convert common truthy/falsey strings/bools to bool.
    Returns None if unknown.
    """
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def _label_to_pred(label: Any) -> Optional[int]:
    """Map judge label to 1 (Misinformation), 0 (Not), or None (Uncertain/unknown)."""
    if not isinstance(label, str):
        return None
    low = label.strip().lower()
    if "not" in low and "mis" in low:
        return 0
    if low.startswith("mis"):
        return 1
    if low.startswith("uncertain") or low.startswith("unknown"):
        return None
    return None


def _gt_to_truth(gt: Any, fake_cls: Any = None) -> Optional[int]:
    """Map ground-truth to 1 (Misinformation) or 0 (Not), else None.

    Supports multiple dataset encodings:
      - gt as boolean or str: True/"True"/"aligned" -> 0; False/"False"/"misaligned" -> 1
      - gt as str: "Real" -> 0; "Fake" -> 1
      - fallback to fake_cls: "original"/"real"/"authentic" -> 0; other non-empty -> 1
    """
    # Try direct boolean
    if isinstance(gt, bool):
        return 0 if gt else 1
    # Try common string encodings
    if isinstance(gt, (int, float)):
        if gt in (1, 1.0):
            return 0
        if gt in (0, 0.0):
            return 1
    if isinstance(gt, str):
        s = gt.strip().lower()
        if s in {"real", "true", "aligned", "1", "yes", "y"}:
            return 0
        if s in {"fake", "false", "misaligned", "0", "no", "n"}:
            return 1
    # Fallback: infer from fake_cls if provided
    if isinstance(fake_cls, str):
        fs = fake_cls.strip().lower()
        if fs in {"original", "real", "authentic", "genuine"}:
            return 0
        if fs:
            return 1
    return None


def _load_dataset_records_for_html(dataset_json_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Best-effort loader for dataset records used only for HTML enrichment."""
    if not dataset_json_path:
        return None
    try:
        return load_dataset_records(Path(dataset_json_path))
    except Exception:
        return None


def _normalize_lookup_path(path: Any, image_root: Optional[str]) -> Optional[str]:
    return normalize_record_path(path, Path(image_root) if image_root else None)


def _attach_dataset_details(
    objs: List[Dict[str, Any]],
    dataset_json_path: Optional[str],
    dataset_image_root: Optional[str],
) -> None:
    """Augment per-sample objects with dataset ground truth if not already present."""

    records = _load_dataset_records_for_html(dataset_json_path)
    if not records:
        return

    idx_map: Dict[int, Dict[str, Any]] = {}
    img_map: Dict[str, Dict[str, Any]] = {}

    for idx, rec in enumerate(records):
        details: Dict[str, Any] = {"dataset_index": idx}
        if "gt_answers" in rec:
            details["gt_answers"] = rec["gt_answers"]
        if rec.get("fake_cls") is not None:
            details["fake_cls"] = rec.get("fake_cls")
        if rec.get("text_source"):
            details["text_source"] = rec.get("text_source")
        if rec.get("image_source"):
            details["image_source"] = rec.get("image_source")

        idx_map[idx] = details

        norm_path = _normalize_lookup_path(rec.get("image_path"), dataset_image_root)
        if norm_path:
            img_map[norm_path] = details

    if not idx_map and not img_map:
        return

    for obj in objs:
        if not isinstance(obj, dict):
            continue

        current_details = obj.get("sample_details")
        if isinstance(current_details, dict) and "gt_answers" in current_details:
            # Already has ground truth; leave untouched
            continue

        details: Optional[Dict[str, Any]] = None

        idx = obj.get("dataset_order_index")
        if isinstance(idx, int):
            details = idx_map.get(idx)

        if details is None:
            norm_img = _normalize_lookup_path(obj.get("image_path"), None)
            if norm_img:
                details = img_map.get(norm_img)

        if details:
            merged = dict(details)
            if isinstance(current_details, dict):
                merged.update({k: v for k, v in current_details.items() if k not in merged})
            obj["sample_details"] = merged


def _cm_to_data_url(cm: Dict[str, int], title: str) -> Optional[str]:
    try:
        import io
        import base64
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        # Matrix layout: rows=true [Not, Misinfo], cols=pred [Not, Misinfo]
        mat = [
            [cm.get("TN", 0), cm.get("FP", 0)],
            [cm.get("FN", 0), cm.get("TP", 0)],
        ]
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=160)
        # Draw heatmap
        im = ax.imshow(mat, cmap="Blues")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([0, 1], labels=["Pred Not", "Pred Misinfo"], rotation=20, ha="right", fontsize=7)
        ax.set_yticks([0, 1], labels=["True Not", "True Misinfo"], fontsize=7)
        # Annotate cells with contrast-aware text color so values
        # remain readable on dark tiles (e.g., high counts)
        max_val = max(max(row) for row in mat) if mat else 0
        threshold = (max_val / 2.0) if max_val else 0
        for i in range(2):
            for j in range(2):
                val = mat[i][j]
                # Use white text on darker (higher) cells, dark text otherwise
                txt_color = "#ffffff" if val > threshold and max_val > 0 else "#08306b"
                ax.text(j, i, str(val), ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout(pad=0.8)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


def _render_metric_block(name: str, cm: Dict[str, int], m: Dict[str, Any]) -> str:
    # Compose a clear block with bullets and a confusion matrix image
    acc = m.get("accuracy", 0)
    rec_pos = m.get("recall_pos", m.get("recall", 0))
    rec_neg = m.get("recall_neg", 0)
    f1 = m.get("f1", 0)
    support = m.get("support", 0)
    img = _cm_to_data_url(cm, f"{name} Confusion Matrix")
    cm_table = (
        "<table class=cm-table>"
        "<tr><th></th><th>Pred Not</th><th>Pred Misinfo</th></tr>"
        f"<tr><th>True Not</th><td>{cm.get('TN',0)}</td><td>{cm.get('FP',0)}</td></tr>"
        f"<tr><th>True Misinfo</th><td>{cm.get('FN',0)}</td><td>{cm.get('TP',0)}</td></tr>"
        "</table>"
    )
    return (
        f"<div class=metric-block>"
        f"<h3>{_escape(name)}</h3>"
        f"<ul>"
        f"<li><b>Accuracy:</b> {acc}</li>"
        f"<li><b>Recall (Misinformation):</b> {rec_pos}</li>"
        f"<li><b>Recall (Not Misinformation):</b> {rec_neg}</li>"
        f"<li><b>F1:</b> {f1}</li>"
        f"<li><b>Support:</b> {support}</li>"
        f"</ul>"
        + (f"<img class=cm-img src='{img}' alt='Confusion Matrix'>" if img else cm_table)
        + "</div>"
    )


def _render_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return """<div class=metrics><p>No metrics available yet. Provide dataset + outputs to compute.</p></div>"""
    jm = metrics.get("judge_metrics", {})  # filtered (excludes Uncertain)
    jm_all = metrics.get("judge_metrics_all", {})  # all predictions (Uncertain penalized)
    ai_m = metrics.get("visual_veracity_ai_detection", {})  # filtered (pred available)
    ai_m_all = metrics.get("visual_veracity_ai_detection_all", {})  # all predictions (missing penalized)
    counts = metrics.get("counts", {})

    parts = []
    parts.append("<div class=metrics>")
    parts.append("<h2>Run Metrics</h2>")
    total = counts.get('total_outputs')
    missing = counts.get('missing_in_gt_map')
    uncertain_c = counts.get('uncertain_count')
    uncertain_r = counts.get('uncertain_rate')
    vv_gt = counts.get('visual_veracity_gt_known')
    vv_cov = counts.get('visual_veracity_pred_covered')
    vv_rate = counts.get('visual_veracity_coverage_rate')
    parts.append(
        f"<p>Total outputs: {_escape(total)}; Missing GT matches: {_escape(missing)}; Uncertain: {_escape(uncertain_c)} ({_escape(uncertain_r)})</p>"
    )
    parts.append(
        f"<p>Visual veracity coverage: {_escape(vv_cov)} / {_escape(vv_gt)} covered ({_escape(vv_rate)})</p>"
    )
    parts.append(_render_metric_block("Judge (filtered; excludes Uncertain)", jm.get("confusion_matrix", {}), jm.get("metrics", {})))
    if jm_all:
        parts.append(_render_metric_block("Judge (all predictions; Uncertain penalized)", jm_all.get("confusion_matrix", {}), jm_all.get("metrics", {})))
    parts.append(_render_metric_block("Visual veracity: AI-generated detection (filtered)", cm=ai_m.get("confusion_matrix", {}), m=ai_m.get("metrics", {})))
    if ai_m_all:
        parts.append(_render_metric_block("Visual veracity: AI-generated detection (all; missing penalized)", cm=ai_m_all.get("confusion_matrix", {}), m=ai_m_all.get("metrics", {})))
    parts.append("</div>")
    return "\n".join(parts)


def _to_data_url(path: str) -> Optional[str]:
    try:
        return image_to_data_url(path, default_mime="application/octet-stream")
    except Exception:
        return None


def _render_sample(idx: int, obj: Dict[str, Any], html_out_path: Optional[str], *, inline_images: bool = False) -> str:
    headline = obj.get("headline")
    img_path = obj.get("image_path")
    rel = obj.get("relevancy", {}) or {}
    ver = obj.get("visual_veracity", {}) or {}
    qa = obj.get("best_qa_per_chain", []) or []
    j = obj.get("judgement", {}) or {}

    # Prepare image src: inline as data URL if requested and file exists
    img_src = ""
    if isinstance(img_path, str) and img_path:
        if inline_images:
            data_url = _to_data_url(img_path)
            if data_url:
                img_src = data_url
        if not img_src:
            img_src = _relpath(img_path, html_out_path)

    # Determine correctness vs ground truth for highlighting.
    # Prefer dataset-provided GT in sample_details.gt_answers when present.
    sd = obj.get("sample_details") or {}
    gt_truth = _gt_to_truth(sd.get("gt_answers"), sd.get("fake_cls"))
    j = obj.get("judgement", {}) or {}
    pred = _label_to_pred(j.get("label"))

    is_incorrect = (gt_truth is not None and pred is not None and gt_truth != pred)
    is_uncertain = (pred is None)

    parts: List[str] = []
    card_cls = "card misaligned" if is_incorrect else "card"
    parts.append(f"<div class='{card_cls}' id='sample-{idx}'>")
    # Badges for quick scanning: correctness vs GT and prediction
    if is_uncertain:
        badge = "<span class='badge unknown'>Uncertain</span>"
    elif is_incorrect:
        badge = "<span class='badge bad'>Incorrect vs GT</span>"
    else:
        # Only mark as correct if both GT and pred are known and match
        if gt_truth is not None and pred is not None:
            if pred == 1:
                badge = "<span class='badge ok'>Correct: Misinformation</span>"
            else:
                badge = "<span class='badge ok'>Correct: Not Misinformation</span>"
        else:
            badge = "<span class='badge unknown'>No GT</span>"
    parts.append(f"<h3>Sample {idx} {badge}</h3>")
    parts.append(f"<p class=headline>{_escape(headline)}</p>")
    if img_src:
        parts.append(f"<img src='{_escape(img_src)}' alt='image' loading='lazy'>")
    else:
        parts.append("<p><i>(image path missing)</i></p>")

    # Collapsible sections
    def section(title: str, content_html: str, open_attr: str = "") -> str:
        return f"<details {open_attr}><summary>{_escape(title)}</summary><div class=section>{content_html}</div></details>"

    # Sample details (from dataset)
    if sd:
        details_html = "<ul>" + "".join(
            f"<li><b>{_escape(k)}:</b> {_escape(v)}</li>" for k, v in sd.items()
        ) + "</ul>"
        parts.append(section("Sample Details", details_html, open_attr="open"))

    # Token usage per sample (if available)
    tu = obj.get("token_usage") or {}
    if isinstance(tu, dict) and (tu.get("prompt") is not None or tu.get("completion") is not None or tu.get("total") is not None):
        parts.append(section(
            "Token Usage",
            f"<p><b>prompt:</b> {_escape(tu.get('prompt'))} | <b>completion:</b> {_escape(tu.get('completion'))} | <b>total:</b> {_escape(tu.get('total'))}</p>"
        ))

    # Relevancy (kept for transparency; no longer drives highlight)
    rel_html = (
        f"<p><b>aligned:</b> {_escape(rel.get('aligned'))} | <b>confidence:</b> {_escape(rel.get('confidence'))}</p>"
        f"<p>{_escape(rel.get('explanation'))}</p>"
    )
    parts.append(section("Relevancy", rel_html))

    # Visual veracity
    anomalies = ver.get("anomalies") or []
    ver_html = (
        f"<p><b>ai_generated:</b> {_escape(ver.get('ai_generated'))} | <b>confidence:</b> {_escape(ver.get('confidence'))}</p>"
        f"<p>{_escape(ver.get('explanation'))}</p>"
        + (f"<p><b>anomalies:</b> {_escape(', '.join(map(str, anomalies)))}</p>" if anomalies else "")
    )
    parts.append(section("Visual Veracity", ver_html))

    # Q/A
    if qa:
        qa_items = []
        for i, it in enumerate(qa, start=1):
            cits = it.get("citations") or []
            cits_html = "".join(
                f"<li>{_escape(ci.get('title') if isinstance(ci, dict) else ci)} {_escape(ci.get('url') if isinstance(ci, dict) else '')}</li>"
                for ci in cits
            )
            qa_items.append(
                "<div class=qa>"
                f"<p><b>Q{ i }:</b> {_escape(it.get('question'))}</p>"
                f"<p><b>A{ i }:</b> {_escape(it.get('answer'))}</p>"
                f"<p><b>confidence:</b> {_escape(it.get('confidence'))}</p>"
                + (f"<ul class=citations>{cits_html}</ul>" if cits else "")
                + "</div>"
            )
        parts.append(section("Best Q/A per chain", "\n".join(qa_items)))

    # Judgement
    gt_str = sd.get("gt_answers") if isinstance(sd, dict) else None
    j_items = [
        f"<p><b>label:</b> {_escape(j.get('label'))} | <b>confidence:</b> {_escape(j.get('confidence'))} | <b>GT:</b> {_escape(gt_str)}</p>",
        f"<p>{_escape(j.get('rationale'))}</p>",
    ]
    kf = j.get("key_factors") or []
    if kf:
        j_items.append("<ul>" + "".join(f"<li>{_escape(x)}</li>" for x in kf) + "</ul>")
    parts.append(section("Final Judgement", "\n".join(j_items), open_attr="open"))

    parts.append("</div>")
    return "\n".join(parts)


def _render_misaligned_index(objs: List[Dict[str, Any]]) -> str:
    mis_ids: List[int] = []
    uncertain_ids: List[int] = []
    for i, o in enumerate(objs, start=1):
        if not isinstance(o, dict):
            continue
        sd = o.get("sample_details") or {}
        gt_truth = _gt_to_truth(sd.get("gt_answers"), sd.get("fake_cls"))
        j = o.get("judgement", {}) or {}
        pred = _label_to_pred(j.get("label"))
        if pred is None:
            uncertain_ids.append(i)
            continue
        if gt_truth is not None and pred != gt_truth:
            mis_ids.append(i)
    if not mis_ids and not uncertain_ids:
        return ""
    parts: List[str] = []
    parts.append("<div class='mis-idx'>")
    if mis_ids:
        links = " ".join(f"<a href='#sample-{i}'>#{i}</a>" for i in mis_ids)
        parts.append("<div><b>Incorrect vs GT:</b> " + links + "</div>")
    if uncertain_ids:
        links_u = " ".join(f"<a href='#sample-{i}'>#{i}</a>" for i in uncertain_ids)
        parts.append("<div><b>Uncertain predictions:</b> " + links_u + "</div>")
    parts.append("</div>")
    return "".join(parts)


def _render_run_params(objs: List[Dict[str, Any]]) -> str:
    rp = None
    for o in objs:
        rp = o.get("run_params") or None
        if rp:
            break
    if not rp:
        # Try to synthesize from legacy fields
        if objs:
            o = objs[0]
            rp = {
                "provider": o.get("provider"),
                "model": o.get("model"),
            }
    if not rp:
        return ""
    items = []
    for k in ["provider", "model", "temperature", "q_chains", "q_per_chain", "answer_questions", "answer_max_sources"]:
        if k in rp:
            items.append(f"<li><b>{_escape(k)}:</b> {_escape(rp.get(k))}</li>")
    return f"<details open><summary>Run Parameters</summary><div class=section><ul>{''.join(items)}</ul></div></details>"


def _compute_token_usage(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_p = 0
    total_c = 0
    total_t = 0
    rows: List[Dict[str, Any]] = []
    for i, o in enumerate(objs, start=1):
        tu = o.get("token_usage") or {}
        p = int(tu.get("prompt") or 0)
        c = int(tu.get("completion") or 0)
        t = int(tu.get("total") or (p + c))
        total_p += p
        total_c += c
        total_t += t
        rows.append({
            "index": i,
            "image_path": o.get("image_path"),
            "headline": o.get("headline"),
            "prompt": p,
            "completion": c,
            "total": t,
        })
    return {"total": {"prompt": total_p, "completion": total_c, "total": total_t}, "rows": rows}


def _render_token_summary_html(objs: List[Dict[str, Any]]) -> str:
    stats = _compute_token_usage(objs)
    tot = stats.get("total", {})
    p = tot.get("prompt", 0)
    c = tot.get("completion", 0)
    t = tot.get("total", 0)
    return (
        "<div class=metrics>"
        "<h2>Token Usage (Run Total)</h2>"
        f"<p><b>prompt:</b> {_escape(p)} | <b>completion:</b> {_escape(c)} | <b>total:</b> {_escape(t)}</p>"
        "</div>"
    )


def render_html_report(
    objs: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    output_path: str,
    title: str = "Pipeline Results",
    *,
    inline_images: bool = False,
    dataset_json_path: Optional[str] = None,
    dataset_image_root: Optional[str] = None,
) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Enrich per-sample objects with ground truth from dataset if available.
    _attach_dataset_details(objs, dataset_json_path, dataset_image_root)

    cards_html = "\n".join(_render_sample(i, o, output_path, inline_images=inline_images) for i, o in enumerate(objs, start=1))

    # Write per-sample token usage CSV next to HTML, if any data present
    token_stats = _compute_token_usage(objs)
    token_rows = token_stats.get("rows", [])
    csv_path = None
    if token_rows:
        base, _ = os.path.splitext(output_path)
        csv_path = base + ".tokens.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["sample_index", "image_path", "headline", "prompt_tokens", "completion_tokens", "total_tokens"])
                for r in token_rows:
                    w.writerow([r.get("index"), r.get("image_path"), r.get("headline"), r.get("prompt"), r.get("completion"), r.get("total")])
        except Exception:
            csv_path = None

    html_doc = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }}
    header {{ margin-bottom: 24px; }}
    .metrics {{ background: #f7f7f9; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 8px; }}
    .card {{ border: 1px solid #eaeaea; border-radius: 8px; padding: 12px 16px; margin: 16px 0; }}
    .card.misaligned {{ border-color: #f1aeb5; box-shadow: 0 0 0 2px #f1aeb5 inset; background: #fff5f5; }}
    .headline {{ font-weight: 600; margin: 4px 0 10px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
    details > summary {{ cursor: pointer; font-weight: 600; margin: 8px 0; }}
    .section {{ padding: 4px 8px; }}
    ul.citations {{ margin: 4px 0 0 18px; }}
    .qa {{ margin-bottom: 10px; }}
    .metric-block {{ border-top: 1px dashed #ddd; padding-top: 8px; margin-top: 8px; }}
    .cm-img {{ max-width: 260px; display: block; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; }}
    .cm-table {{ border-collapse: collapse; margin: 6px 0; }}
    .cm-table th, .cm-table td {{ border: 1px solid #ccc; padding: 4px 6px; font-size: 12px; }}
    .badge {{ display: inline-block; font-size: 11px; padding: 2px 6px; border-radius: 10px; vertical-align: middle; margin-left: 6px; border: 1px solid #ccc; }}
    .badge.ok {{ background: #e6f4ea; color: #0f5132; border-color: #badbcc; }}
    .badge.bad {{ background: #f8d7da; color: #842029; border-color: #f1aeb5; }}
    .badge.unknown {{ background: #e2e3e5; color: #41464b; border-color: #d3d6d8; }}
    .mis-idx {{ margin-top: 10px; padding: 8px 10px; background: #fff5f5; border: 1px solid #f1aeb5; border-radius: 6px; }}
    .mis-idx a {{ margin-right: 6px; }}
  </style>
  <script>
    // No heavy JS required; <details> handles collapsibles.
  </script>
  </head>
<body>
  <header>
    <h1>{_escape(title)}</h1>
    { _render_metrics(metrics) }
    { _render_token_summary_html(objs) }
    { _render_misaligned_index(objs) }
  </header>
  <main>
    {_render_run_params(objs)}
    {cards_html}
  </main>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description="Render HTML report from pipeline outputs JSONL")
    ap.add_argument("--outputs", required=True, help="JSONL file with final objects")
    ap.add_argument("--html", required=True, help="Output HTML path")
    ap.add_argument("--title", default="Pipeline Results", help="Report title")
    ap.add_argument("--dataset-json", default="", help="Dataset JSON path for metrics (optional)")
    ap.add_argument("--image-root", default="", help="Image root for metrics alignment (optional)")
    ap.add_argument("--inline-images", action="store_true", help="Embed images as base64 data URLs in HTML")
    args = ap.parse_args()

    objs = _load_jsonl(args.outputs)

    metrics = None
    if args.dataset_json and args.image_root:
        try:
            # Lazy import to avoid circular
            from scripts.evaluate import evaluate
            metrics = evaluate(Path(args.outputs), Path(args.dataset_json), Path(args.image_root), save_report=None)
        except Exception as exc:
            print(f"Warning: failed to compute metrics – {exc}")
            metrics = None

    render_html_report(
        objs,
        metrics,
        args.html,
        args.title,
        inline_images=bool(args.inline_images),
        dataset_json_path=(args.dataset_json or None),
        dataset_image_root=(args.image_root or None),
    )


if __name__ == "__main__":
    main()
