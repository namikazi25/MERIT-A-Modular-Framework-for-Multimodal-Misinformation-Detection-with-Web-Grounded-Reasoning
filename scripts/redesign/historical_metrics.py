"""Aggregate-only reconstruction of historical test outputs, never a redesign replay."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.redesign.evaluation import evaluate


def exact_relative(value):
    if type(value) is not str:
        raise ValueError('Missing historical image identity')
    s=value.replace('\\', '/')
    marker='MMFakeBench_test/'
    if marker in s:
        s=s.split(marker,1)[1]
    elif 'MMFakeBench_' in s:
        raise ValueError('Other dataset root; cannot resolve against test labels')
    s=s.lstrip('/')
    if not s.startswith(('fake/','real/')) or '..' in s.split('/'):
        raise ValueError('Unsupported historical identity')
    return s


def reconstruct(dataset, outputs):
    records=json.loads(Path(dataset).read_text())
    gt={}
    for row in records:
        ident=exact_relative(row['image_path'])
        if ident in gt: raise ValueError('Duplicate ground-truth identity')
        gt[ident]=row['gt_answers']
    predictions=[]
    for line in Path(outputs).read_text().splitlines():
        row=json.loads(line)
        ident=exact_relative(row.get('image_path'))
        if ident not in gt: raise ValueError('Historical identity not found')
        judge=row.get('judgement')
        pred=judge.get('label') if isinstance(judge,dict) else None
        predictions.append({'sample_id':ident,'prediction':pred})
    # The historical intended scope is the identities in this existing output file.
    # Missing intended records cannot be inferred without its original run manifest.
    scope={r['sample_id']:gt[r['sample_id']] for r in predictions}
    result=evaluate(scope,predictions)
    return {'mode':'AGGREGATE_HISTORICAL_RECONSTRUCTION','metrics':result,
            'limitations':['Historical leaked inputs; not corrected GLM results.',
                            'Scope is recorded outputs, not proof that every originally intended prediction exists.'],
            'hashes':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (dataset,outputs)}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--outputs',required=True)
    parser.add_argument('--report',required=True)
    a=parser.parse_args()
    result=reconstruct(a.dataset,a.outputs)
    with Path(a.report).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result['metrics'],indent=2))

if __name__=='__main__':main()
