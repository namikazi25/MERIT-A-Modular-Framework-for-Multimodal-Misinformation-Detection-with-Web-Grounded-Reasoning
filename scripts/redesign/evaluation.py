"""Exact-ID binary selective evaluation; failures and abstention retain denominators."""
from collections import Counter

LABELS = {'Fake': 1, 'True': 0, 'Misinformation': 1, 'Not Misinformation': 0}


def evaluate(truth, predictions):
    if not truth or any(type(k) is not str or not k for k in truth):
        raise ValueError('Nonempty exact identity ground truth required')
    if any(v not in ('Fake', 'True') for v in truth.values()):
        raise ValueError('Unknown ground-truth label')
    seen = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get('sample_id'), str):
            raise ValueError('Malformed prediction identity')
        ident = row['sample_id']
        if ident not in truth:
            raise ValueError('Prediction identity is absent from ground truth')
        if ident in seen:
            raise ValueError('Duplicate prediction identity')
        seen[ident] = row
    if not seen:
        raise ValueError('Zero matched ground truth')
    cm = Counter(TP=0, TN=0, FP=0, FN=0)
    statuses = Counter()
    class_total, class_answered = Counter(), Counter()
    correct = 0
    for ident, label in truth.items():
        y = LABELS[label]
        class_total[label] += 1
        row = seen.get(ident)
        if row is None:
            statuses['missing'] += 1
            continue
        pred = row.get('prediction')
        if row.get('status', 'OK') != 'OK':
            statuses['failure'] += 1
        elif pred in ('ABSTAIN', 'Uncertain'):
            statuses['abstain'] += 1
        elif not isinstance(pred, str) or pred not in LABELS:
            statuses['malformed'] += 1
        else:
            p = LABELS[pred]
            statuses['answered'] += 1
            class_answered[label] += 1
            cm['TP' if y and p else 'FN' if y else 'FP' if p else 'TN'] += 1
            correct += p == y
    def ratio(a, b):
        return a / b if b else None
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'n_expected': len(truth), 'n_matched': len(seen), 'statuses': dict(statuses),
            'confusion_answered': dict(cm), 'accuracy_all': correct / len(truth),
            'accuracy_answered': ratio(correct, statuses['answered']),
            'macro_f1_answered': (f1(cm['TP'], cm['FP'], cm['FN']) + f1(cm['TN'], cm['FN'], cm['FP'])) / 2,
            'coverage': statuses['answered'] / len(truth),
            'class_coverage': {k: ratio(class_answered[k], class_total[k]) for k in ('Fake', 'True')},
            'false_flags_all_authentic': ratio(cm['FP'], class_total['True']),
            'false_positives_answered_authentic': ratio(cm['FP'], class_answered['True']),
            'misinformation_detected_all': ratio(cm['TP'], class_total['Fake']),
            'misinformation_recall_answered': ratio(cm['TP'], class_answered['Fake'])}
