"""Per-sample evidence-sufficiency ratings on gpt-4o-mini (hybrid prerequisite).

One cheap call per sample: claim + module signals + evidence -> {evidence_sufficient, reason}.
Writes results/judge_study/sufficiency_dev.jsonl (dev) — also used by H1/H3 and Phase 3 contrast.
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv; load_dotenv()
from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.candidate_judge import render_modules, render_evidence
from scripts.utils.json_utils import extract_json_object

PROMPT = ("You are an evidence-sufficiency rater. Given a claim (image+headline pair), module signals, "
          "and retrieved web evidence, decide whether the retrieved evidence is SUFFICIENT to determine "
          "whether the claim is true or false. Consider: does the evidence address the claim's key "
          "entities and assertions? Is it specific enough, or generic/off-topic? "
          "Return ONLY JSON: {\"evidence_sufficient\": true|false, \"reason\": \"...\"}")

def main():
    sp = json.load(open('manifests/split-500-v1.json'))
    items = sp['dev']
    print('dev samples:', len(items))
    loader = LLMModelLoader(ModelConfig(provider='openai', model='gpt-4o-mini-2024-07-18', temperature=0.0))
    out_path = 'results/judge_study/sufficiency_dev.jsonl'
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            done.add(json.loads(l)['sample_id'])
    with open(out_path, 'a') as fh:
        for i, d in enumerate(items, start=1):
            sid = str(d.get('sample_id') or d['image_path'])
            if sid in done:
                continue
            b = json.load(open(d['bundle']))
            user = (f"CLAIM: {b.get('claim')}\n\nMODULE SIGNALS:\n{render_modules(b)}\n\n"
                    f"RETRIEVED EVIDENCE:\n{render_evidence(b, 'top5_500')}\n\nReturn your rating JSON.")
            resp = loader.get_model().invoke([{"role": "system", "content": PROMPT},
                                              {"role": "user", "content": user}])
            parsed = extract_json_object(str(getattr(resp, 'content', resp)), fallback=lambda p: {})
            rec = {"sample_id": sid, "image_path": d['image_path'],
                   "evidence_sufficient": bool(parsed.get('evidence_sufficient')),
                   "reason": (parsed.get('reason') or '')[:500]}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f'[{i}/{len(items)}]', flush=True)
    print('sufficiency pass done ->', out_path)

if __name__ == '__main__':
    main()
