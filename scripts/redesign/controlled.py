"""Versioned five-core-case diagnostic comparisons; no automatic sample expansion."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from scripts.redesign.experiment import atomic,hashes,load_sample
from scripts.redesign.guard import Admission
from scripts.redesign.ledger import Ledger
from scripts.redesign.clients import GLMClient,JevClient,ProviderFailure
from scripts.redesign.retrieval import Discovery,Extraction,SnapshotStore,passages
from scripts.redesign.pipeline import Features,extract_claims,run_frozen,select_evidence
from scripts.redesign.investigation import investigate
from scripts.redesign.baseline import historical_baseline
from scripts.redesign.evaluation import evaluate

PHASES={
 'baseline':{'RT3_Brave_snippets':Features(False,False),'RT4_Brave_pages_B1':Features(False,False),'B2_historical_Brave':None},
 'frozen':{'A1_detector_only':Features(False,False,detector=True),'A2_claim_linking_only':Features(True,False),
           'A3_image_role_only':Features(False,True),'C1':Features(detector=True),
           'C2':Features(evidence_checker='glm',detector=True),'C3':Features(evidence_checker='jev',detector=True)},
 'live':{'L1':Features(detector=True,image_origin=True),'L2':Features(evidence_checker='glm',detector=True,image_origin=True),
         'L3':Features(evidence_checker='jev',detector=True,image_origin=True)}}
# Upper bounds per whole five-case phase: model input limits are enforced per
# dispatch. Baseline <=8 image-bearing old-stage calls + extraction +2 judges;
# frozen six final images + six text checks; live <=6 image calls +36 text checks
# and up to36 Jev calls per case, with 0 retries and two logical searches/condition.
PHASE_BOUND={'baseline':10.0,'frozen':6.0,'live':8.0}


def failure(exc):
    out={'prediction':None,'status':'FAILURE','error_type':type(exc).__name__}
    # Only our own fixed semantic diagnostics; never SDK/HTTP exception bodies.
    if isinstance(exc,ProviderFailure):out['diagnostic']=str(exc)[:250]
    return out


def collect(discovery,extraction,sid,query):
    r=discovery.search(sid,query)
    statuses=[{'tool':'searxng_brave','status':r['status'],'detail':r['snapshot_id']}]
    rows=[x for x in r['results'] if 'mmfakebench' not in (x['url']+' '+x['title']).lower()]
    snippets=[{'id':r['snapshot_id']+':snippet:'+str(i),'claim_id':'c1','text':x['snippet'],'kind':'snippet','url':x['url'],'snapshot_id':r['snapshot_id']} for i,x in enumerate(rows)]
    pages=[]
    for x in rows[:2]:
        try:
            p=extraction.scrape(sid,x['url']);statuses.append({'tool':'firecrawl_self_hosted','status':p['status'],'detail':p['snapshot_id']});pages.extend(passages(p,'c1'))
        except (ValueError,OSError):statuses.append({'tool':'firecrawl_self_hosted','status':'UNSAFE_OR_UNRESOLVABLE_URL'})
    return {'snippets':snippets,'pages':pages,'tool_status':statuses}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--register-only',action='store_true')
    a=p.parse_args();run=Path(a.run_dir);folder=run/'controlled_005_v2';folder.mkdir(exist_ok=True)
    ids=json.loads((run/'batch_005/protocol.json').read_text())['ids']
    guard=Admission('.')
    for sid in ids:guard.check(sid)
    if len(ids)!=5:raise ValueError('Only existing registered five-case diagnostic allowed')
    cfg=json.loads(Path('config/redesign_runtime.json').read_text())
    if cfg['searxng_engines']!=['brave'] or not cfg['live_enabled']:raise ValueError('Explicit Brave live configuration required')
    prior=folder/'baseline_summary.json'
    if a.phase!='baseline' and not prior.exists():raise SystemExit('Baseline collection phase must finish first')
    state=json.loads((run/'state.json').read_text());ledger=Ledger(run/'spend.jsonl',15,applicable_remaining=15,deadline=state['deadline_utc'])
    protocol={'version':'controlled_005_v2','phase':a.phase,'ids':ids,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'features':{k:asdict(v) if v else 'original historical stages' for k,v in PHASES[a.phase].items()},
        'stack':{'discovery':'SearXNG Brave only','candidate_cap':5,'page_cap':2,'extraction':'local Firecrawl','fallback':'none'},
        'evidence_cap_utf8_bytes':12000,'selection':'A2/C: two lexical-overlap candidate passages per extracted claim; ties stable; not entailment.',
        'query_policy':'Reuse three valid initial frozen queries; single diagnosed extraction attempt for two earlier failures; freeze all before new verdicts.',
        'scope':'Exploratory engineering follow-up on the SAME five IDs after operational failures. Earlier Bing results retained. No reserved data or scaling.',
        'live_policy':'Fresh investigation per condition, same initial extracted claims, <=2 text queries, <=4 rounds, unavailable image-origin explicitly recorded.',
        'fact_check_policy':'Fact-check sources allowed equally. MMFakeBench-named mirrors excluded by URL/title heuristic; not complete semantic contamination prevention.',
        'phase_maximum_usd':PHASE_BOUND[a.phase],'source_config_hashes':hashes()}
    pp=folder/(a.phase+'_protocol.json')
    if pp.exists():
        old=json.loads(pp.read_text())
        if old['source_config_hashes']!=protocol['source_config_hashes'] or old['ids']!=ids:raise SystemExit('Resume source drift requires explicit review')
    else:atomic(pp,protocol)
    if a.register_only:print('Registered',pp);return
    summary_path=folder/(a.phase+'_summary.json')
    if summary_path.exists():print('Phase already complete; no duplicate requests');return
    if ledger.summary()['pending_ids'] or ledger.summary()['halted']:raise SystemExit('Unresolved/billing-blocked ledger')
    if float(ledger.summary()['session_available_usd'])<PHASE_BOUND[a.phase]:raise SystemExit('Insufficient remaining balance for complete phase maximum')
    from dotenv import load_dotenv
    load_dotenv('.env',override=False)
    glm=GLMClient(json.loads(Path('config/llm.json').read_text()),guard,ledger,max_output=cfg['max_output_tokens'],reasoning=cfg['reasoning_effort'],audit_dir=folder/'model_responses')
    jev=JevClient(guard,ledger);bank=json.loads(Path('config/jev_questions_v1.json').read_text())
    store=SnapshotStore(run/'snapshots_web');discovery=Discovery(guard,ledger,store,backend='searxng',endpoint=cfg['searxng_url'],engines=('brave',));extraction=Extraction(guard,ledger,store,endpoint=cfg['firecrawl_endpoint'])
    if a.phase=='live' and discovery._block_state():
        raise SystemExit('Fresh investigation blocked by recorded discovery rate/access failure; no model dispatch')
    source=json.loads(Path('data/MMFakeBench_test/source/MMFakeBench_test.json').read_text())
    qp=folder/'frozen_queries.json'
    queries=json.loads(qp.read_text()) if qp.exists() else {}
    previous=json.loads((run/'batch_005/frozen_queries.json').read_text())
    for sid in ids:
        if sid in queries:continue
        if 'error_type' not in previous[sid]:queries[sid]=dict(previous[sid],origin='initial_frozen_valid_query')
        else:
            try:queries[sid]=extract_claims(load_sample(sid,guard,source),glm)
            except Exception as exc:queries[sid]=failure(exc)
        atomic(qp,queries)
        if ledger.summary()['pending_ids'] or ledger.summary()['halted']:raise SystemExit('Unresolved charge; do not retry')
    detector=None
    if a.phase!='baseline':
        from scripts.redesign.detector import GenerationDetector
        detector=GenerationDetector(guard)
    cases=folder/'cases';cases.mkdir(exist_ok=True)
    initial_spend=ledger.summary()
    for sid in ids:
        target=cases/(sid+'.json');r=json.loads(target.read_text()) if target.exists() else {'sample_id':sid,'conditions':{}}
        sample=load_sample(sid,guard,source);q=queries[sid]
        if 'retrieval' not in r and 'error_type' not in q:
            r['retrieval']=collect(discovery,extraction,sid,q['query']);atomic(target,r)
        def get_evidence(sid,query):
            ret=collect(discovery,extraction,sid,query)
            return select_evidence(ret['pages']+ret['snippets'])[0],ret['tool_status']
        for name,features in PHASES[a.phase].items():
            if name in r['conditions']:continue
            before=ledger.summary()
            import time
            started=time.monotonic()
            try:
                if name=='B2_historical_Brave':result=historical_baseline(sample,glm,get_evidence)
                elif 'error_type' in q:result={'sample_id':sid,'prediction':None,'status':'QUERY_FAILURE','query_diagnostic':q}
                elif a.phase=='live':result=investigate(sample,q,get_evidence,glm,features=features,jev=jev,bank=bank,detector=detector)
                else:
                    ret=r['retrieval'];e=ret['snippets'] if name=='RT3_Brave_snippets' else ret['pages']+ret['snippets']
                    result=run_frozen(sample,q,e,ret['tool_status'],glm,features=features,jev=jev,bank=bank,detector=detector)
            except Exception as exc:result=dict(failure(exc),sample_id=sid)
            after=ledger.summary();result.update(wall_latency_s=time.monotonic()-started,rated_cost_usd=str(float(after['spent_usd'])-float(before['spent_usd'])))
            r['conditions'][name]=result;atomic(target,r)
            if after['pending_ids'] or after['halted']:raise SystemExit('Unresolved charge; phase checkpointed, no automatic retry')
        if detector:atomic(folder/'detector_results.json',detector.results)
        print(a.phase,sid,'finished',flush=True)
    truth={sid:guard.check(sid)['eval_only']['gt_answers'] for sid in ids}
    records=[json.loads((cases/(sid+'.json')).read_text()) for sid in ids]
    metrics={name:evaluate(truth,[r['conditions'][name] for r in records]) for name in PHASES[a.phase]}
    failures=sum(r['conditions'][name]['status']!='OK' for r in records for name in PHASES[a.phase])
    result={'mode':'LIVE_EXPLORATORY_SAME_FIVE','phase':a.phase,'metrics':metrics,'failures':failures,
        'initial_spend':initial_spend,'final_spend':ledger.summary(),'scale_authorized':False,
        'source_config_hashes':hashes(),'frozen_queries_sha256':hashlib.sha256(qp.read_bytes()).hexdigest()}
    atomic(summary_path,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
