"""Capture J1+J2 (gpt) labels AND rationales for a sample list (for Task E labeling sheet).

Usage:
  python -m scripts.judge_study.rationale_capture --split manifests/split-500-v1.json \
      --subset dev --out results/judge_study/dev_rationales.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.getcwd())
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.judges import judge_j1, judge_j2
from scripts.judge_study.manifest import make_manifest, write_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    sp = json.load(open(args.split))
    items = sp[args.subset]
    bundles = [json.load(open(d["bundle"])) for d in items]
    print(f"{args.subset}: {len(bundles)} bundles")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    loader = LLMModelLoader(ModelConfig(provider="openai", model="gpt-4o-mini-2024-07-18", temperature=0.0))
    manifest = make_manifest(phase="task-e", stage="B", run_id=os.path.basename(args.out).replace(".jsonl",""),
        model={"provider": "openai", "id": "gpt-4o-mini-2024-07-18", "revision": "n/a", "quant": "n/a"},
        prompt_files=["prompts/judge_study/j1_rule_prompt.txt", "prompts/judge_study/j2_evidence_grounded.txt"],
        evidence_version=os.path.basename(args.split), seed=None,
        sample_ids=[b.get("sample_id") or d["image_path"] for b, d in zip(bundles, items)],
        temperature=0.0, repo_root=args.repo_root,
        notes="J1+J2 rationale capture for Task E labeling sheet (dev only).")
    write_manifest(manifest, os.path.join(os.path.dirname(args.out), "rationale_manifest.json"))

    with open(args.out, "w") as fh:
        for i, (b, d) in enumerate(zip(bundles, items), start=1):
            j1 = judge_j1(b, loader)
            j2 = judge_j2(b, loader)
            rec = {"sample_id": b.get("sample_id"), "image_path": d["image_path"],
                   "fake_cls": d.get("fake_cls"), "gt": d.get("gt"),
                   "J1": {"label": j1.get("label"), "confidence": j1.get("confidence"),
                          "rationale": (j1.get("rationale") or "")[:1500]},
                   "J2": {"label": j2.get("label"), "confidence": j2.get("confidence"),
                          "rationale": (j2.get("rationale") or "")[:1500],
                          "evidence_items": (j2.get("extra") or {}).get("evidence_items") or []}}
            fh.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                print(f"[{i}/{len(bundles)}]", flush=True)
    print("done ->", args.out)

if __name__ == "__main__":
    main()
