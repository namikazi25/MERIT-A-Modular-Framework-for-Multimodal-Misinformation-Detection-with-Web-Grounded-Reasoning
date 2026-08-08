"""Citation re-fetch pass for evidence bundles (Phase 0 completion).

Frozen bundles from pre-split runs store citations as {url, title} without
extracted text. This tool re-fetches every cited URL, extracts visible text,
and writes it back into the bundle documents as `description` plus fetch
metadata (status / error / fetched_at). Updates in place and writes a
refetch manifest with per-run stats. Run once per evidence dir; re-runs skip
URLs already fetched successfully unless --force.

Usage:
  python -m scripts.judge_study.refetch_citations --evidence-dir evidence/frozen-gpt4omini-200-v1 [--workers 8]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
TAG_RE = re.compile(r"<[^>]+>")
MAX_TEXT = 2000


def extract_text(html_src: str) -> str:
    """Crude HTML-to-text: title + stripped body text, collapsed whitespace."""
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html_src, re.S | re.I)
    if m:
        title = html.unescape(TAG_RE.sub(" ", m.group(1))).strip()
    body = TAG_RE.sub(" ", html_src)
    body = html.unescape(body)
    body = re.sub(r"\s+", " ", body).strip()
    if title and body.startswith(title):
        body = body[len(title):].strip()
    text = (title + "\n" + body).strip() if title else body
    return text[:MAX_TEXT]


def fetch(url: str, timeout: float = 15.0) -> dict:
    import requests
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code == 200 and "text" in ctype.lower() or "html" in ctype.lower() or "xml" in ctype.lower():
            text = extract_text(r.text)
            return {"url": url, "ok": True, "status": r.status_code, "final_url": r.url,
                    "title": text.split("\n")[0][:200] if text else "",
                    "description": text, "error": None, "fetched_at": datetime.now(timezone.utc).isoformat()}
        return {"url": url, "ok": False, "status": r.status_code, "final_url": r.url,
                "title": "", "description": None, "error": f"HTTP {r.status_code} content-type {ctype}",
                "fetched_at": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        return {"url": url, "ok": False, "status": None, "final_url": None,
                "title": "", "description": None, "error": f"{type(e).__name__}: {str(e)[:120]}",
                "fetched_at": datetime.now(timezone.utc).isoformat()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delay", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = args.evidence_dir
    bundle_files = [f for f in sorted(os.listdir(d)) if f.endswith(".json") and f != "manifest.json"]
    todo: list = []  # (bundle_file, question, doc_index, url)
    for bf in bundle_files:
        b = json.load(open(os.path.join(d, bf)))
        for q, docs in (b.get("documents") or {}).items():
            for i, doc in enumerate(docs):
                url = doc.get("url")
                if not url:
                    continue
                if doc.get("description") and not args.force:
                    continue  # already has text (e.g. live-gathered bundles)
                todo.append((bf, q, i, url))

    print(f"{len(bundle_files)} bundles, {len(todo)} URLs to fetch", flush=True)
    results = {}

    def work(item):
        bf, q, i, url = item
        time.sleep(args.delay * ((hash(url) % 5) / 4.0))  # jitter
        return item, fetch(url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for item, res in ex.map(work, todo):
            bf, q, i, url = item
            results.setdefault(bf, []).append(res)
            if (len(results) * 10) % 100 == 0 or len(results) == len(todo):
                pass

    # write back
    stats = {"total": len(todo), "ok": 0, "failed": 0, "errors": []}
    for bf in bundle_files:
        b = json.load(open(os.path.join(d, bf)))
        for res in results.get(bf, []):
            for q, docs in (b.get("documents") or {}).items():
                for doc in docs:
                    if doc.get("url") == res["url"]:
                        doc["description"] = res["description"]
                        doc["fetch_status"] = "ok" if res["ok"] else "failed"
                        doc["fetch_error"] = res["error"]
                        doc["fetched_at"] = res["fetched_at"]
                        stats["ok" if res["ok"] else "failed"] += 1
                        if not res["ok"]:
                            stats["errors"].append({"url": res["url"], "error": res["error"]})
        json.dump(b, open(os.path.join(d, bf), "w"), ensure_ascii=False, indent=1)

    stats["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(d, "refetch_manifest.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"refetch done: ok={stats['ok']} failed={stats['failed']} of {stats['total']}", flush=True)
    if stats["failed"]:
        print("sample failures:", json.dumps(stats["errors"][:5], indent=1))


if __name__ == "__main__":
    main()
