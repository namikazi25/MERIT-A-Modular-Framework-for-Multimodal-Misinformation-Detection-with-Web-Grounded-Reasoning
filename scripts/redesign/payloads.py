"""Typed projection for new evidence requests; evaluation records never pass wholesale."""
import math
from scripts.ai_judge import _project_relevancy, _project_visual_veracity


def text(value, *, required=False):
    if type(value) is str:
        return value
    if required:
        raise ValueError('Required text field has malformed type')
    return None


def probability(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid probability')
    return value


def records(value, fields):
    if value is None: return []
    if type(value) is not list: raise ValueError('Expected list of typed records')
    result=[]
    for row in value:
        if type(row) is not dict: raise ValueError('Expected typed record')
        result.append({key:text(row.get(key),required=required) for key,required in fields.items()})
    return result


def project(payload):
    if type(payload) is not dict: raise ValueError('Expected structured payload')
    checks=[]
    for row in payload.get('checks',[]):
        if type(row) is not dict: raise ValueError('Malformed evidence check')
        checks.append({key:text(row.get(key),required=True) for key in ('question_id','claim_id','source_id','evaluator')}
                      | {'probability':probability(row.get('probability'))})
    return {
        'headline':text(payload.get('headline'),required=True),
        'claims':records(payload.get('claims'),{'id':True,'text':True,'source_span':True,'qualifiers':False,'image_dependency':False}),
        'evidence':records(payload.get('evidence'),{'id':True,'claim_id':True,'text':True,'kind':True,'url':False,'snapshot_id':True,'publication_date':False,'date_origin':False}),
        'tool_status':records(payload.get('tool_status'),{'tool':True,'status':True,'detail':False}),
        'relevancy':_project_relevancy(payload.get('relevancy')),
        'visual_veracity':_project_visual_veracity(payload.get('visual_veracity')),
        'image_role':text(payload.get('image_role')),
        'checks':checks,
    }
