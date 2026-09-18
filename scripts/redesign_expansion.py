"""Preparation-only benchmark interfaces. No downloading, image I/O or inference.

These normalizers are exercised with synthetic fixtures. New benchmark IDs are
not admitted by the MERIT development guard. Official evidence grading remains
separate from the simple local classification diagnostic below.
"""
from collections import Counter
from copy import deepcopy

AVERITEC_LABELS=('Supported','Refuted','Not Enough Evidence','Conflicting Evidence/Cherrypicking')
AVERIMATEC_LABELS=('Supported','Refuted','Not Enough Evidence','Conflicting')


def _identity(dataset,split,index):
    if split not in ('train','dev','test','synthetic') or type(index) is not int or index<0:
        raise ValueError('Explicit split and original nonnegative row index required')
    return f'{dataset}:{split}:{index}'


def _text(value):
    if type(value) is not str or not value.strip():raise ValueError('Required claim text is malformed')
    return value


def averitec(row,*,split,index):
    label=row.get('label')
    if label is not None and label not in AVERITEC_LABELS:raise ValueError('Unknown AVeriTeC label; do not guess a binary mapping')
    return {'sample_id':_identity('averitec',split,index),'model_input':{'claim':_text(row.get('claim')),'image_references':[]},
        'evaluation_only':{'label':label,'questions':deepcopy(row.get('questions',[])),
                           'justification':row.get('justification'),'fact_checking_article':row.get('fact_checking_article')},
        'provenance':{'dataset':'AVeriTeC','split':split,'original_row_index':index,'schema_source':'MichSchli/AVeriTeC README'},
        'execution_authorized':False}


def averimatec(row,*,split,index,gold_label=None):
    """Input keys verified against official src/mm_checker.py, pinned in report.

    The public loader does not establish a gold-label field name. It must be
    mapped explicitly after licensed release/schema review, never inferred.
    Image reference strings remain intact to avoid basename collisions.
    """
    images=row.get('claim_images')
    if type(images) is not list or any(type(x) is not str or not x for x in images):raise ValueError('Malformed image reference list')
    if gold_label is not None and gold_label not in AVERIMATEC_LABELS:raise ValueError('Unknown AVerImaTeC label')
    return {'sample_id':_identity('averimatec',split,index),'model_input':{'claim':_text(row.get('claim_text')),'image_references':list(images)},
        'evaluation_only':{'label':gold_label,'questions':deepcopy(row.get('questions',[])),'fact_checking_article':row.get('article')},
        'provenance':{'dataset':'AVerImaTeC','split':split,'original_row_index':index,
                      'schema_revision':'5fd2e03763e198eb6ddcea9e41b7210020f37127','date':row.get('date'),'location':row.get('location')},
        'execution_authorized':False}


def classification_diagnostic(truth,predictions,labels):
    if not truth:raise ValueError('No reference identities')
    if any(v not in labels for v in truth.values()):raise ValueError('Unknown reference label')
    seen={}
    for row in predictions:
        sid=row.get('sample_id')
        if sid not in truth:raise ValueError('Prediction identity not in reference split')
        if sid in seen:raise ValueError('Duplicate prediction identity')
        seen[sid]=row
    if not seen:raise ValueError('Zero matched identities')
    matrix={actual:{pred:0 for pred in labels} for actual in labels};states=Counter();correct=0
    for sid,label in truth.items():
        row=seen.get(sid)
        if row is None:states['missing']+=1;continue
        pred=row.get('prediction')
        if row.get('status')!='OK':states['failure']+=1
        elif pred=='ABSTAIN':states['abstain']+=1
        elif pred not in labels:states['invalid_label']+=1
        else:matrix[label][pred]+=1;states['answered']+=1;correct+=pred==label
    return {'n_expected':len(truth),'n_matched':len(seen),'statuses':dict(states),'confusion_answered':matrix,
            'coverage':states['answered']/len(truth),'accuracy_all':correct/len(truth),
            'official_evidence_score':None,'scope':'Local classification diagnostic only; not official evidence-aware benchmark scoring.'}
