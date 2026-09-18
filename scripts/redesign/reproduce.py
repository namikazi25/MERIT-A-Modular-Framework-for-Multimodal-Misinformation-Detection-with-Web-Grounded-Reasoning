"""Export and verify evaluation-only metrics without model/search dependencies.

Export requires the real registry and admits the previously run five core cases.
Verification requires only the Python standard library and this evaluator. A
bundle has labels and is evaluation-only: never use it as a model-input record.
Export is local preparation, not permission to redistribute data or evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.redesign.evaluation import evaluate

SCHEMA = 'merit_metrics_bundle_v1'
ROOT = Path(__file__).resolve().parents[2]
BATCHES = ('batch_005', 'controlled_005_v2', 'replay_005_v3')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON value')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def evaluator_hash():
    return hashlib.sha256((ROOT / 'scripts/redesign/evaluation.py').read_bytes()).hexdigest()


def identifier(value):
    return type(value) is str and re.fullmatch(r'[A-Za-z0-9_+-]{1,100}', value) is not None


def metrics(payload):
    if type(payload) is not dict or set(payload) != {'truth', 'groups', 'batches'}:
        raise ValueError('Unsupported bundle payload fields')
    truth, groups, batches = (payload[k] for k in ('truth', 'groups', 'batches'))
    if type(truth) is not dict or not 1 <= len(truth) <= 100 or any(not identifier(k) for k in truth):
        raise ValueError('Invalid bounded evaluation identities')
    if type(groups) is not dict or set(groups) != set(truth) or any(not identifier(g) for g in groups.values()):
        raise ValueError('Group membership must bind every evaluation identity')
    if type(batches) is not dict or not 1 <= len(batches) <= 10:
        raise ValueError('Invalid batch collection')
    result = {}
    for batch, conditions in batches.items():
        if not identifier(batch) or type(conditions) is not dict or not 1 <= len(conditions) <= 50:
            raise ValueError('Invalid named batch or conditions')
        result[batch] = {}
        for name, rows in conditions.items():
            if not identifier(name) or type(rows) is not list or len(rows) > len(truth):
                raise ValueError('Invalid named condition or records')
            for row in rows:
                if type(row) is not dict or set(row) != {'sample_id', 'status', 'prediction'}:
                    raise ValueError('Only typed evaluation fields are permitted')
                if row['status'] not in ('OK', 'ERROR') or row['prediction'] not in (
                        None, 'Fake', 'True', 'Misinformation', 'Not Misinformation', 'ABSTAIN', 'Uncertain'):
                    raise ValueError('Unexpected evaluation value')
            result[batch][name] = evaluate(truth, rows)
    return result


def make_bundle(payload):
    computed = metrics(payload)
    body = {'schema': SCHEMA, 'evaluation_only': True, 'evaluator_sha256': evaluator_hash(),
            'payload': payload, 'expected_metrics': computed}
    return dict(body, bundle_sha256=digest(body))


def verify_bundle(bundle):
    if type(bundle) is not dict or set(bundle) != {
            'schema', 'evaluation_only', 'evaluator_sha256', 'payload', 'expected_metrics', 'bundle_sha256'}:
        raise ValueError('Unsupported bundle schema or extra metadata')
    if bundle['schema'] != SCHEMA or bundle['evaluation_only'] is not True:
        raise ValueError('An evaluation-only bundle is required')
    body = {k: v for k, v in bundle.items() if k != 'bundle_sha256'}
    if digest(body) != bundle['bundle_sha256']:
        raise ValueError('Bundle checksum mismatch')
    if bundle['evaluator_sha256'] != evaluator_hash():
        raise ValueError('Evaluator version mismatch')
    computed = metrics(bundle['payload'])
    if computed != bundle['expected_metrics']:
        raise ValueError('Recomputed metrics differ from recorded metrics')
    return {'status': 'VERIFIED', 'evaluation_only': True,
            'n_examples': len(bundle['payload']['truth']),
            'n_conditions': sum(len(v) for v in computed.values()), 'metrics': computed,
            'bundle_sha256': bundle['bundle_sha256'],
            'limitation': 'Consistency with recorded references, not authenticity or release approval. No model, retrieval, evidence-quality or calibration rerun.'}


def export_run(run_dir):
    # Import admission only for export; independent verification is stdlib-only.
    from scripts.redesign.guard import Admission
    run = Path(run_dir)
    protocol = read_json(run / 'batch_005/protocol.json')
    ids = protocol['ids']
    if type(ids) is not list or len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError('Only the existing five-case comparison can be exported')
    guard = Admission(ROOT)
    admitted = {sid: guard.check(sid) for sid in ids}
    truth = {sid: r['eval_only']['gt_answers'] for sid, r in admitted.items()}
    groups = {sid: r['group_id'] for sid, r in admitted.items()}
    analysis = read_json(run / 'analysis_v1/results.json')
    batches = {}
    for batch in BATCHES:
        records = [read_json(run / batch / 'cases' / (sid + '.json')) for sid in ids]
        if any(r.get('sample_id') != sid for sid, r in zip(ids, records)):
            raise ValueError('Case identity mismatch')
        conditions = set(analysis[batch]['metrics'])
        if any(set(r['conditions']) != conditions for r in records):
            raise ValueError('Case conditions differ from the recorded analysis')
        batches[batch] = {}
        for condition in sorted(conditions):
            original = [r['conditions'][condition] for r in records]
            if any(row.get('sample_id') != sid for sid, row in zip(ids, original)):
                raise ValueError('Prediction identity mismatch')
            if evaluate(truth, original) != analysis[batch]['metrics'][condition]:
                raise ValueError('Stored analysis differs from original predictions')
            # Malformed values remain malformed (null); failure details, arbitrary
            # text, evidence, paths and nested provider fields are never exported.
            rows = []
            for row in original:
                prediction = row.get('prediction')
                if type(prediction) is not str or prediction not in (
                        'Fake', 'True', 'Misinformation', 'Not Misinformation', 'ABSTAIN', 'Uncertain'):
                    prediction = None
                rows.append({'sample_id': row['sample_id'],
                             'status': 'OK' if row.get('status', 'OK') == 'OK' else 'ERROR',
                             'prediction': prediction})
            if evaluate(truth, rows) != analysis[batch]['metrics'][condition]:
                raise ValueError('Projection changed evaluation semantics')
            batches[batch][condition] = rows
    return make_bundle({'truth': truth, 'groups': groups, 'batches': batches})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run-dir', required=True)
    export.add_argument('--output', required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('bundle')
    verify.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'export':
        bundle = export_run(args.run_dir)
        result = verify_bundle(bundle)
        with Path(args.output).open('x') as stream:
            json.dump(bundle, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(f"Exported evaluation-only bundle: {result['n_examples']} cases, {result['n_conditions']} conditions. Release review still required.")
    else:
        result = verify_bundle(read_json(args.bundle))
        if args.output:
            with Path(args.output).open('x') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write('\n')
        print(f"Verified {result['n_conditions']} condition metrics for {result['n_examples']} cases; no model/search calls.")


if __name__ == '__main__':
    main()
