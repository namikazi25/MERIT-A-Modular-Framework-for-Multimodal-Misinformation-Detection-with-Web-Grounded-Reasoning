"""Registered, resumable nested development batches; never loads a non-core case for inference."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from scripts.redesign.guard import Admission
from scripts.redesign.ledger import Ledger,BudgetBlocked
from scripts.redesign.clients import GLMClient,JevClient,ProviderFailure
from scripts.redesign.retrieval import Discovery,Extraction,SnapshotStore,passages
from scripts.redesign.pipeline import Features,extract_claims,run_frozen,select_evidence
from scripts.redesign.baseline import historical_baseline
from scripts.redesign.evaluation import evaluate


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


def hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('scripts/redesign').glob('*.py'),*Path('config').glob('*.json')]}


def register(run,n):
    if n not in (5,20,100):raise ValueError('Only nested batches 5,20,100 permitted')
    registry=json.loads(Path('manifests/redesign_data_registry_v2.json').read_text())
    ids=sorted(registry['roles']['development_core'])[:n]
    if len(registry['roles']['development_core'])!=100:raise ValueError('Development boundary changed')
    protocol={'version':'development_protocol_v1','n':n,'ids':ids,'registered_utc':datetime.now(timezone.utc).isoformat(),
              'condition_names':['RT3_bing_snippets','RT4_bing_pages_B1','B2_sanitised_historical_stages','C1','C2','C3'],
              'blocked_conditions':{'RT1':'DDG access failure/CAPTCHA','RT2':'DDG access failure/CAPTCHA','A1':'Dedicated detector not yet integrated','A3_origin':'No authorised reverse-image service'},
              'stack':{'discovery':'self-hosted SearXNG','engines':['bing'],'extraction':'self-hosted Firecrawl','page_cap':2,'candidate_cap':5,'fallback':'none'},
              'evidence_cap':{'utf8_bytes':12000,'meaning':'Conservative text-token upper-bound proxy, not exact GLM token count. Same cap for every evidence condition.'},
              'baseline':'B2 uses original 3 sequential chains x up to 3 questions, original stage prompts, original best-QA selection and original final judge prompt. Sanitised inputs, GLM, explicit failures, guarded requests, contemporary common retrieval are changes. Not recovered rebuttal repairs.',
              'comparison_scope':'RT3/RT4 frozen queries and paired contemporary retrieval; B2 retains its own historical query procedure. C1/C2/C3 use identical frozen evidence. No cloud-Firecrawl performance/cost claim.',
              'fact_check_policy':'Fact-check sources allowed equally; direct MMFakeBench dataset/answer-mirror URLs/titles excluded before selection.',
              'scale_gate':'No pending charges; complete paired batch within remaining conservative bound; zero operational failures required before next batch. Never scale on favourable accuracy.',
              'source_config_hashes':hashes()}
    folder=run/f'batch_{n:03d}';folder.mkdir(exist_ok=True)
    p=folder/'protocol.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old['ids']!=ids or old['source_config_hashes']!=protocol['source_config_hashes']:raise ValueError('Resume dependency drift; review/invalidate acceptance before proceeding')
        return folder,old
    with p.open('x') as f:json.dump(protocol,f,indent=2)
    return folder,protocol


def load_sample(sid,guard,source):
    row=guard.check(sid);raw=source[row['dataset_index']]
    path=Path('data/MMFakeBench_test')/row['image_path_rel'].lstrip('/')
    return {'sample_id':sid,'headline':raw['text'],'image_bytes':path.read_bytes(),'image_path':path}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--n',type=int,choices=(5,20,100),default=5);p.add_argument('--register-only',action='store_true')
    a=p.parse_args();run=Path(a.run_dir)
    if a.n>5:
        prior=run/f'batch_{5 if a.n==20 else 20:03d}/summary.json'
        if not prior.exists() or not json.loads(prior.read_text()).get('scale_gate_passed'):raise SystemExit('Prior nested operational gate has not passed')
    folder,protocol=register(run,a.n)
    if a.register_only:print('Registered',folder);return
    config=json.loads(Path('config/redesign_runtime.json').read_text())
    if not config['live_enabled']:raise SystemExit('Live requests disabled')
    state=json.loads((run/'state.json').read_text())
    ledger=Ledger(run/'spend.jsonl',15,applicable_remaining=15,deadline=state['deadline_utc'])
    if ledger.summary()['pending_ids']:raise SystemExit('Outstanding reservations; reconcile before resume')
    # For five cases: <=8 image-bearing B2 requests, 1 extraction, 5 other judges per
    # case, plus <=17 bounded text requests and six Jev pairs. Conservative $2.80/case with text requests capped at 65,536 input-bound tokens.
    # Larger batches must cover the *additional* cases; do not reset completed spending.
    previous=set()
    for size in (5,20):
        if size<a.n and (run/f'batch_{size:03d}/summary.json').exists():previous.update(json.loads((run/f'batch_{size:03d}/protocol.json').read_text())['ids'])
    new_ids=[sid for sid in protocol['ids'] if sid not in previous]
    if float(ledger.summary()['session_available_usd'])<len(new_ids)*2.80:raise SystemExit('Remaining balance cannot cover conservative maximum complete comparison batch')
    from dotenv import load_dotenv
    load_dotenv('.env',override=False)
    guard=Admission('.');store=SnapshotStore(run/'snapshots_web')
    glm=GLMClient(json.loads(Path('config/llm.json').read_text()),guard,ledger,max_output=config['max_output_tokens'],reasoning=config['reasoning_effort']);jev=JevClient(guard,ledger)
    discovery=Discovery(guard,ledger,store,backend='searxng',endpoint='http://127.0.0.1:8080',engines=('bing',))
    extraction=Extraction(guard,ledger,store,endpoint='http://127.0.0.1:3002/v2/scrape')
    source=json.loads(Path('data/MMFakeBench_test/source/MMFakeBench_test.json').read_text())
    bank=json.loads(Path('config/jev_questions_v1.json').read_text())
    calls_dir=folder/'cases';calls_dir.mkdir(exist_ok=True)
    queries_path=folder/'frozen_queries.json'
    queries=json.loads(queries_path.read_text()) if queries_path.exists() else {}
    # Freeze all common queries before examining any verdict outcome.
    for sid in new_ids:
        if sid not in queries:
            sample=load_sample(sid,guard,source)
            try:queries[sid]=extract_claims(sample,glm)
            except Exception as exc:queries[sid]={'error_type':type(exc).__name__}
            atomic(queries_path,queries)
            if ledger.summary()['pending_ids'] or ledger.summary()['halted']:raise SystemExit('Query request unresolved/billing halted; no automatic retry')
    def collect(sid,query):
        r=discovery.search(sid,query);statuses=[{'tool':'searxng_bing','status':r['status'],'detail':r['snapshot_id']}]
        snippets=[];pages=[]
        candidates=[x for x in r['results'] if 'mmfakebench' not in (x['url']+' '+x['title']).lower()]
        for i,row in enumerate(candidates):
            snippets.append({'id':r['snapshot_id']+':snippet:'+str(i),'claim_id':'c1','text':row['snippet'],'kind':'snippet','url':row['url'],'snapshot_id':r['snapshot_id']})
        for row in candidates[:2]:
            try:
                page=extraction.scrape(sid,row['url']);statuses.append({'tool':'firecrawl_self_hosted','status':page['status'],'detail':page['snapshot_id']});pages.extend(passages(page,'c1'))
            except (ValueError,OSError):statuses.append({'tool':'firecrawl_self_hosted','status':'UNSAFE_OR_UNRESOLVABLE_URL'})
        return snippets,pages,statuses
    for index,sid in enumerate(new_ids):
        target=calls_dir/(sid+'.json')
        record=json.loads(target.read_text()) if target.exists() else {'sample_id':sid,'conditions':{}}
        sample=load_sample(sid,guard,source)
        q=queries[sid]
        if 'error_type' in q:
            for cond in protocol['condition_names']:record['conditions'][cond]={'sample_id':sid,'status':'QUERY_FAILURE','prediction':None}
            atomic(target,record);continue
        if 'retrieval' not in record:
            snippets,pages,statuses=collect(sid,q['query'])
            record['retrieval']={'snippets':snippets,'pages':pages,'tool_status':statuses};atomic(target,record)
        ret=record['retrieval'];common=ret['pages']+ret['snippets']
        for condition in protocol['condition_names']:
            if condition in record['conditions']:continue
            try:
                if condition=='B2_sanitised_historical_stages':
                    def baseline_collect(sid,query):
                        snippets,pages,statuses=collect(sid,query)
                        return select_evidence(pages+snippets)[0],statuses
                    result=historical_baseline(sample,glm,baseline_collect)
                else:
                    f=Features(claim_linking=condition in ('C1','C2','C3'),image_role=condition in ('C1','C2','C3'),evidence_checker='glm' if condition=='C2' else 'jev' if condition=='C3' else 'none')
                    evidence=ret['snippets'] if condition=='RT3_bing_snippets' else common
                    result=run_frozen(sample,q,evidence,ret['tool_status'],glm,features=f,jev=jev,bank=bank)
                record['conditions'][condition]=result
            except Exception as exc:record['conditions'][condition]={'sample_id':sid,'prediction':None,'status':'FAILURE','error_type':type(exc).__name__}
            atomic(target,record)
            if ledger.summary()['pending_ids'] or ledger.summary()['halted']:raise SystemExit('Outstanding/billing-failed request; stop before further dispatch')
        print(f'Completed case {index+1}/{len(new_ids)}',flush=True)
    all_records={}
    for batch in sorted(run.glob('batch_*/cases/*.json')):
        r=json.loads(batch.read_text())
        if r['sample_id'] in protocol['ids']:all_records[r['sample_id']]=r
    truth={sid:guard.check(sid)['eval_only']['gt_answers'] for sid in protocol['ids']}
    metrics={};failures=0
    for cond in protocol['condition_names']:
        predictions=[r['conditions'][cond] for r in all_records.values() if cond in r['conditions']]
        metrics[cond]=evaluate(truth,predictions);failures+=sum(r.get('status')!='OK' for r in predictions)
    tool_failures=sum(x['status'] not in ('SUCCESS','USABLE') for r in all_records.values() for x in r.get('retrieval',{}).get('tool_status',[]))
    summary={'mode':'LIVE_DEVELOPMENT','n':a.n,'metrics':metrics,'operational_failures':failures,'common_tool_failures':tool_failures,'scale_gate_passed':failures==0 and tool_failures==0 and len(all_records)==a.n,'spend':ledger.summary(),'limitations':protocol['comparison_scope']}
    atomic(folder/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
