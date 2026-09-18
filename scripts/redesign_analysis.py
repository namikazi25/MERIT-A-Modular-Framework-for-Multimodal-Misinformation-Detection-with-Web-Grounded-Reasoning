"""Offline paired analysis and unannotated human package for executed core cases."""
import argparse
import base64
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import hashlib
import html
import io
import json
from pathlib import Path
import random
from scripts.redesign.guard import Admission
from scripts.redesign.evaluation import LABELS,evaluate
from scripts.redesign.experiment import atomic,load_sample


def answered(row):return row.get('status')=='OK' and row.get('prediction') in LABELS
def correct(row,label):return answered(row) and LABELS[row['prediction']]==LABELS[label]


def paired(truth,left,right,groups,*,draws=2000,seed=20260917):
    ids=sorted(truth)
    if set(left)!=set(ids) or set(right)!=set(ids) or set(groups)!=set(ids):raise ValueError('Paired identity/group mismatch')
    clusters=defaultdict(list)
    for sid in ids:clusters[groups[sid]].append(sid)
    counts=Counter(errors_fixed=0,new_errors=0,correct_after_previous_nonanswer=0,previous_correct_now_nonanswer=0,both_answered=0)
    delta={}
    for sid in ids:
        a,b=left[sid],right[sid];ca,cb=correct(a,truth[sid]),correct(b,truth[sid])
        counts['errors_fixed']+=answered(a) and not ca and cb
        counts['new_errors']+=ca and answered(b) and not cb
        counts['correct_after_previous_nonanswer']+=not answered(a) and cb
        counts['previous_correct_now_nonanswer']+=ca and not answered(b)
        counts['both_answered']+=answered(a) and answered(b)
        delta[sid]=int(cb)-int(ca)
    rng=random.Random(seed);keys=sorted(clusters);boots=[]
    if len(keys)>=2:
        for _ in range(draws):
            sampled=[sid for key in rng.choices(keys,k=len(keys)) for sid in clusters[key]]
            boots.append(sum(delta[sid] for sid in sampled)/len(sampled))
        boots.sort();interval=[boots[int(.025*draws)],boots[min(draws-1,int(.975*draws))]]
    else:interval=None
    return dict(counts,n=len(ids),n_verified_groups=len(keys),all_case_accuracy_delta=sum(delta.values())/len(ids),
                grouped_bootstrap_95_percentile=interval,bootstrap_draws=draws,seed=seed,
                limitation='Exploratory cluster bootstrap; five repeatedly used cases cannot establish generalisation. Nonanswers count as not correct in this all-case metric.')


def ledger_analysis(path):
    events=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()];reserved={};sums=defaultdict(Decimal);counts=Counter();usage=defaultdict(Counter)
    for e in events:
        if e['event']=='reserve':reserved[e['request_id']]=e
        elif e['event']=='settle':
            r=reserved.pop(e['request_id']);name=r['service'];sums[name]+=Decimal(e['actual_usd']);counts[name]+=1
            for k,v in e.get('usage',{}).items():
                if type(v) is int:usage[name][k]+=v
    return {'rated_cost_by_service_usd':{k:str(v) for k,v in sums.items()},'settled_requests_by_service':dict(counts),
            'token_usage_by_service':{k:dict(v) for k,v in usage.items()},'pending':list(reserved),
            'cost_basis':'Published uncached rates applied to returned usage, not invoice reconciliation. Local discovery/extraction and CPU detector have zero metered provider charge; host resources are not costed.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-name',default='analysis_v1');p.add_argument('--verify-existing',action='store_true');a=p.parse_args()
    run=Path(a.run_dir);out=run/a.output_name
    if a.verify_existing:
        saved=json.loads((out/'results.json').read_text());guard=Admission('.');checked=0
        for name,section in saved.items():
            if name=='spending':continue
            records=[json.loads(p.read_text()) for p in (run/name/'cases').glob('*.json')]
            truth={r['sample_id']:guard.check(r['sample_id'])['eval_only']['gt_answers'] for r in records}
            expected=json.loads((run/'batch_005/protocol.json').read_text())['ids']
            if set(truth)!=set(expected) or len(records)!=len(expected):raise ValueError('Stored analysis identity drift')
            for condition,metrics in section['metrics'].items():
                if evaluate(truth,[r['conditions'][condition] for r in records])!=metrics:raise ValueError('Stored metric mismatch')
                checked+=1
        if ledger_analysis(run/'spend.jsonl')!=saved['spending']:raise ValueError('Ledger changed since analysis')
        print(f'Verified {checked} stored condition metrics and complete ledger accounting; no model or retrieval requests.')
        return
    if out.exists():raise SystemExit('Analysis output exists; choose an explicit new version')
    out.mkdir();ids=json.loads((run/'batch_005/protocol.json').read_text())['ids'];guard=Admission('.')
    rows={sid:guard.check(sid) for sid in ids};truth={sid:row['eval_only']['gt_answers'] for sid,row in rows.items()};groups={sid:row['group_id'] for sid,row in rows.items()}
    aggregate={};batches={}
    for name in ('batch_005','controlled_005_v2','replay_005_v3'):
        paths=list((run/name/'cases').glob('*.json'))
        if not paths:continue
        batch={json.loads(p.read_text())['sample_id']:json.loads(p.read_text()) for p in paths}
        if set(batch)!=set(ids):raise ValueError('Incomplete/mismatched batch, not a paired analysis')
        batches[name]=batch;conditions=sorted(set.intersection(*(set(v['conditions']) for v in batch.values())))
        metrics={};timings={}
        for condition in conditions:
            preds=[batch[sid]['conditions'][condition] for sid in ids];metrics[condition]=evaluate(truth,preds)
            timings[condition]={'recorded_wall_latency_s':sum(x.get('wall_latency_s',0) for x in preds),
              'rated_cost_usd':str(sum((Decimal(x.get('rated_cost_usd','0')) for x in preds),Decimal(0))),
              'complete_per_condition_accounting':all('rated_cost_usd' in x for x in preds)}
        pairs={}
        for left,right in [('C1','C2'),('C1','C3'),('C2','C3'),('B1_replayed_pages','A1_detector_only'),('B1_replayed_pages','A2_claim_linking_only'),('B1_replayed_pages','A3_image_role_only'),('RT3_Brave_snippets','RT4_Brave_pages_B1')]:
            if left in conditions and right in conditions:pairs[left+' -> '+right]=paired(truth,{sid:batch[sid]['conditions'][left] for sid in ids},{sid:batch[sid]['conditions'][right] for sid in ids},groups)
        aggregate[name]={'metrics':metrics,'paired':pairs,'timing_and_condition_cost':timings}
    aggregate['spending']=ledger_analysis(run/'spend.jsonl')
    atomic(out/'results.json',aggregate)
    text=['# Development results','', 'All conditions use the same five approved core IDs; no reserve or holdout evaluation. Zero-coverage macro-F1 is the implementation\'s zero convention and has no answered-classification interpretation. No accuracy or calibration claim follows from an empty-evidence deferral.', '',
          '| Batch / condition | Correct / all | Coverage | TP/TN/FP/FN | Answered macro-F1 | States |','|---|---:|---:|---|---:|---|']
    for batch,section in aggregate.items():
        if batch=='spending':continue
        for condition,m in section['metrics'].items():
            cm=m['confusion_answered'];f1=f"{m['macro_f1_answered']:.3f}" if m['coverage'] else 'N/A'
            text.append(f"| {batch} / {condition} | {round(m['accuracy_all']*len(ids))}/{len(ids)} | {m['coverage']:.0%} | {cm['TP']}/{cm['TN']}/{cm['FP']}/{cm['FN']} | {f1} | {m['statuses']} |")
    text+=['','Full class coverage, authentic false flags, misinformation recall, paired fixed/new errors, group bootstrap intervals, usage and latency are in results.json. The raw initial batch did not retain all failed response content, so failure causes and condition-level costs there cannot be reconstructed completely.','', 'Human calibration labels are absent. Jev probabilities are retained, not treated as gold and not multiplied.']
    (out/'results.md').write_text('\n'.join(text)+'\n')
    # All available initial baseline errors/failures and correct controls; no core expansion.
    base=batches['batch_005'];bad=[];good=[]
    for sid in ids:
        (good if correct(base[sid]['conditions']['B2_sanitised_historical_stages'],truth[sid]) else bad).append(sid)
    selected=bad[:50]+good[:25];annotation=out/'annotation';annotation.mkdir()
    source=json.loads(Path('data/MMFakeBench_test/source/MMFakeBench_test.json').read_text())
    fields=['sample_id','human_claim_faithfulness','human_source_relevance','human_evidence_status','human_image_role','human_failure_category','human_notes','annotator','completed_utc']
    with (annotation/'human_labels.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({'sample_id':sid} for sid in selected)
    machine={}
    for sid in selected:
        sample=load_sample(sid,guard,source);r=base[sid];role='error_or_failure' if sid in bad else 'correct_control'
        from PIL import Image
        with Image.open(io.BytesIO(sample['image_bytes'])) as image:mime=Image.MIME[image.format]
        machine[sid]={'machine_generated':True,'selection_role':role,'benchmark_label':truth[sid],
                      'category_suggestion':'authentic_false_flag' if truth[sid]=='True' and answered(r['conditions']['B2_sanitised_historical_stages']) else 'operational_or_evidence_review' if sid in bad else 'correct_control',
                      'same_label_control_available':any(truth[k]==truth[sid] for k in good)}
        evidence=r.get('retrieval',{}).get('pages',[])+r.get('retrieval',{}).get('snippets',[])
        body=['<!doctype html><meta charset="utf-8"><title>MERIT human evidence review</title>',
          '<style>body{max-width:960px;margin:32px auto;font:16px system-ui;line-height:1.5}img{max-width:600px;max-height:420px}pre{white-space:pre-wrap}article{border-top:1px solid #bbb;padding:10px 0}</style>',
          '<h1>Objective evidence — review before opening model rationales</h1>',f'<p>Case {html.escape(sid)}</p>',
          '<h2>Input claim</h2><p>'+html.escape(sample['headline'])+'</p>',
          '<img alt="Approved development image" src="data:'+mime+';base64,'+base64.b64encode(sample['image_bytes']).decode()+'">',
          '<h2>Stored source passages and tool states</h2><pre>'+html.escape(json.dumps(r.get('retrieval',{}).get('tool_status',[]),indent=2))+'</pre>']
        for e in evidence:body.append('<article><p>'+html.escape(e['id'])+' — '+html.escape(e.get('url') or '')+'</p><p>'+html.escape(e['text'])+'</p></article>')
        body+=['<p>Record independent observations in human_labels.csv. Every human field is blank; this package is not completed annotation.</p>',
               '<details><summary>Model outputs — inspect after objective evidence</summary><pre>'+html.escape(json.dumps(r['conditions'],indent=2,ensure_ascii=False))+'</pre></details>']
        (annotation/(sid+'.html')).write_text('\n'.join(body))
    atomic(annotation/'machine_suggestions_and_selection.json',machine)
    (annotation/'README.md').write_text(f'# Annotation preparation only\n\n{len(bad)} available error/failure cases and {len(good)} correct controls from the initial five-case historical baseline. The requested approximate 50/25 quota cannot be met from five executed cases. No additional cases were opened. Controls share the misinformation label with failed cases where possible; the authentic false-positive has no authentic correct control here.\n\nOpen each HTML file locally, inspect objective input/evidence first, record human observations in human_labels.csv, then optionally open model outputs. Suggested machine categories and benchmark labels are isolated in machine_suggestions_and_selection.json. Do not mistake them for human judgments. No human annotations or question-level calibration have been completed.\n')
    print('Wrote offline analysis and blank annotation package:',out)

if __name__=='__main__':main()
