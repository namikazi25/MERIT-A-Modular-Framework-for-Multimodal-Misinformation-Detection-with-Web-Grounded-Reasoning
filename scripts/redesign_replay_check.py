"""One declared diagnostic: live models, unchanged old evidence, no fresh search."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from decimal import Decimal
from scripts.redesign.experiment import atomic,hashes,load_sample
from scripts.redesign.controlled import PHASES,failure
from scripts.redesign.guard import Admission
from scripts.redesign.ledger import Ledger
from scripts.redesign.clients import GLMClient,JevClient
from scripts.redesign.retrieval import SnapshotStore,passages
from scripts.redesign.pipeline import Features,run_frozen
from scripts.redesign.evaluation import evaluate


def verify_evidence(records,store):
    for row in records:
        doc=store.get(row['snapshot_id']);payload=doc['payload']
        if row['kind']=='snippet':
            if not any(r['url']==row['url'] and r['snippet']==row['text'] for r in payload['results']):raise ValueError('Snippet differs from immutable snapshot')
        else:
            originals=passages(dict(payload,snapshot_id=row['snapshot_id']),row['claim_id'])
            if not any(p['id']==row['id'] and p['text']==row['text'] and p['url']==row['url'] for p in originals):raise ValueError('Passage differs from immutable snapshot')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--register-only',action='store_true');a=ap.parse_args()
    run=Path(a.run_dir);folder=run/'replay_005_v3';folder.mkdir(exist_ok=True)
    ids=json.loads((run/'batch_005/protocol.json').read_text())['ids'];guard=Admission('.');store=SnapshotStore(run/'snapshots_web')
    queries=json.loads((run/'controlled_005_v2/frozen_queries.json').read_text())
    evidence={};statuses={};source_artifacts={}
    for sid in ids:
        guard.check(sid);path=run/'batch_005/cases'/(sid+'.json');source_artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        ret=json.loads(path.read_text()).get('retrieval',{})
        evidence[sid]=ret.get('pages',[])+ret.get('snippets',[]);verify_evidence(evidence[sid],store)
        statuses[sid]=ret.get('tool_status',[])+[{'tool':'discovery','status':'FROZEN_REPLAY_ONLY','detail':'No new search. Three old Bing collections; two absent collections explicitly empty.'}]
    conditions={'B1_replayed_pages':Features(False,False),**PHASES['frozen']}
    pinned=hashes();pinned[__file__]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol={'mode':'LIVE_MODELS_FROZEN_OLD_EVIDENCE_DIAGNOSTIC','registered_utc':datetime.now(timezone.utc).isoformat(),
      'ids':ids,'conditions':list(conditions),'source_artifacts':source_artifacts,'source_config_hashes':pinned,
      'queries_sha256':hashlib.sha256((run/'controlled_005_v2/frozen_queries.json').read_bytes()).hexdigest(),
      'maximum_complete_batch_usd':5,'no_fresh_discovery':True,
      'reason':'Single bounded structured-response/candidate-link diagnostic after original failures; Brave collection is blocked by rate limiting.',
      'limitations':'Same five repeatedly inspected development cases. Old Bing evidence mostly irrelevant; two cases have none. Tests robustness, not retrieval superiority or generalisation. No further outcome-driven reruns.'}
    pp=folder/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text())
        if any(old[k]!=protocol[k] for k in ('source_artifacts','source_config_hashes','queries_sha256','ids')):raise SystemExit('Diagnostic input/source drift')
    else:atomic(pp,protocol)
    if a.register_only:print('Registered',pp);return
    if (folder/'summary.json').exists():print('Diagnostic already complete; no duplicates');return
    state=json.loads((run/'state.json').read_text());ledger=Ledger(run/'spend.jsonl',15,applicable_remaining=15,deadline=state['deadline_utc'])
    before=ledger.summary()
    if before['pending_ids'] or before['halted'] or Decimal(before['session_available_usd'])<5:raise SystemExit('Ledger cannot cover whole bounded diagnostic')
    from dotenv import load_dotenv
    load_dotenv('.env',override=False)
    cfg=json.loads(Path('config/redesign_runtime.json').read_text())
    glm=GLMClient(json.loads(Path('config/llm.json').read_text()),guard,ledger,max_output=cfg['max_output_tokens'],reasoning=cfg['reasoning_effort'],audit_dir=folder/'model_responses')
    jev=JevClient(guard,ledger);bank=json.loads(Path('config/jev_questions_v1.json').read_text())
    from scripts.redesign.detector import GenerationDetector
    detector=GenerationDetector(guard);source=json.loads(Path('data/MMFakeBench_test/source/MMFakeBench_test.json').read_text())
    cases=folder/'cases';cases.mkdir(exist_ok=True)
    for sid in ids:
        target=cases/(sid+'.json');record=json.loads(target.read_text()) if target.exists() else {'sample_id':sid,'conditions':{}}
        sample=load_sample(sid,guard,source)
        for name,features in conditions.items():
            if name in record['conditions']:continue
            start=ledger.summary();timer=time.monotonic()
            try:result=run_frozen(sample,queries[sid],evidence[sid],statuses[sid],glm,features=features,jev=jev,bank=bank,detector=detector)
            except Exception as exc:result=dict(failure(exc),sample_id=sid)
            end=ledger.summary();result.update(wall_latency_s=time.monotonic()-timer,rated_cost_usd=str(Decimal(end['spent_usd'])-Decimal(start['spent_usd'])))
            record['conditions'][name]=result;atomic(target,record)
            if end['pending_ids'] or end['halted']:raise SystemExit('Unresolved charge; no automatic retry')
        print(sid,'complete',flush=True)
    records=[json.loads((cases/(sid+'.json')).read_text()) for sid in ids];truth={sid:guard.check(sid)['eval_only']['gt_answers'] for sid in ids}
    metrics={k:evaluate(truth,[r['conditions'][k] for r in records]) for k in conditions}
    atomic(folder/'detector_results.json',detector.results)
    summary={'mode':protocol['mode'],'metrics':metrics,'before_spend':before,'after_spend':ledger.summary(),'n':5,'fresh_discovery_requests':0}
    atomic(folder/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
