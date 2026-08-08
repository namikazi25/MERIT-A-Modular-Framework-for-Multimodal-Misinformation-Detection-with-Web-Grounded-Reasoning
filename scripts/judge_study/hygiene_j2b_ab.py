"""Supplementary Task C A/B on an evidence-READING judge (J2B), dev-core, gpt.
J1 is evidence-invariant (never reads documents), so this measures real effects."""
import sys, os, json, csv
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv; load_dotenv()
from scripts.llm_loader import LLMModelLoader, ModelConfig
from scripts.judge_study.judges import judge_j2b
from scripts.judge_study.hygiene import dedup_documents, strip_boilerplate, rerank_evidence
from scripts.judge_study.hygiene_ab import load_devcore_bundles

bundles = load_devcore_bundles('manifests/devcore-100-v1.json', 'manifests/split-500-v1.json')
loader = LLMModelLoader(ModelConfig(provider='openai', model='gpt-4o-mini-2024-07-18', temperature=0.0))
os.makedirs('results/judge_study/taskC-j2b', exist_ok=True)
for name, fn in [('none', lambda b: (b, {})), ('C1', dedup_documents), ('C2', strip_boilerplate)]:
    rows = []
    for b in bundles:
        nb, st = fn(b)
        try:
            j = judge_j2b(nb, loader)
            rows.append({'sample_id': b['sample_id'], 'label': j['label'], 'confidence': j['confidence'], 'stats': json.dumps(st)})
        except Exception as e:
            rows.append({'sample_id': b['sample_id'], 'label': 'JUDGE_ERROR', 'confidence': None, 'stats': json.dumps(st)})
        if len(rows) % 25 == 0: print(f'{name} [{len(rows)}/100]', flush=True)
    with open(f'results/judge_study/taskC-j2b/labels_{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id', 'label', 'confidence', 'stats'])
        w.writeheader(); w.writerows(rows)
    print(name, 'done', flush=True)
print('ALL DONE')
