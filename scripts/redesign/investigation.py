"""Inspectable bounded investigation; repeated Jev/GLM checks precede actions."""
from scripts.redesign.pipeline import Features,check_evidence,select_evidence,link_evidence,validate_verdict,FINAL_PROMPT
from scripts.redesign.payloads import project
from scripts.redesign.clients import ProviderFailure


def investigate(sample,extracted,collect,client,*,features=Features(evidence_checker='jev'),
                jev=None,bank=None,detector=None,initial_evidence=(),initial_status=(),max_searches=2):
    if type(max_searches) is not int or not 0<=max_searches<=2:
        raise ValueError('At most two logical searches permitted')
    claims=extracted['claims'] if features.claim_linking else [{'id':'c1','text':sample['headline'],'source_span':sample['headline']}]
    evidence=list(initial_evidence);statuses=list(initial_status);actions=[];rounds=[];calls=[]
    search_count=0;last_query=None;checks=[];detector_signal=None
    if features.detector:
        if detector is None:statuses.append({'tool':'dedicated_generation_detector','status':'UNAVAILABLE'})
        else:detector_signal=detector(sample);actions.append({'action':'RUN_FORENSIC_DETECTOR'})
    if features.image_origin:
        statuses.append({'tool':'image_origin','status':'UNAVAILABLE','detail':'No authorised reverse-image matching provider'})
        actions.append({'action':'IMAGE_ORIGIN_UNAVAILABLE'})
    if features.image_role and extracted.get('image_role')=='unclear':
        r=client.complete(sample['sample_id'],{'headline':sample['headline'],'claims':claims},
            'Describe only the image relationship to these claims. Return JSON {"image_role":"literal/illustration/stock/satire/unclear","explanation":"computed visual finding"}. The image alone cannot establish external event provenance.',sample['image_bytes'])
        calls.append(r)
        if r['parsed'].get('image_role') not in ('literal','illustration','stock','satire','unclear') or type(r['parsed'].get('explanation')) is not str:
            raise ProviderFailure('Malformed image-role recheck')
        extracted=dict(extracted,image_role=r['parsed']['image_role']);actions.append({'action':'RECHECK_IMAGE_TEXT','finding':r['parsed']})
    for step in range(features.max_steps):
        selected,omitted=select_evidence(link_evidence(claims,evidence) if features.claim_linking else evidence)
        payload=project({'headline':sample['headline'],'claims':claims,'evidence':selected,'tool_status':statuses,
                         'checks':[],'image_role':extracted.get('image_role') if features.image_role else None,
                         'visual_veracity':detector_signal})
        checks=[]
        if selected and features.evidence_checker!='none':
            checks,check_calls=check_evidence(sample,payload,features.evidence_checker,client,jev,bank);calls.extend(check_calls)
        payload['checks']=checks
        unresolved=[c for c in claims if not any(x['claim_id']==c['id'] and x['probability']>=.6 for x in checks)]
        rounds.append({'step':step,'payload':payload,'omitted_evidence_ids':omitted})
        can_search=search_count<max_searches and step<features.max_steps-1
        should_search=not selected or (features.evidence_checker!='none' and unresolved)
        if should_search and can_search:
            query=extracted['query'] if search_count==0 else (unresolved[0]['text'] if unresolved else sample['headline'])
            query=query[:2000]
            if query==last_query:
                actions.append({'action':'DEFER_DUPLICATE_QUERY'});break
            action='SEARCH_TEXT' if search_count==0 else 'REFORMULATE_QUERY'
            actions.append({'action':action,'query':query});last_query=query;search_count+=1
            new,status=collect(sample['sample_id'],query);statuses.extend(status)
            known={p['id'] for p in evidence}
            evidence.extend(p for p in new if p['id'] not in known)
            continue
        actions.append({'action':'FINALISE_WITH_GLM' if selected else 'DEFER_UNRESOLVED'});break
    if not payload['evidence']:
        verdict={'prediction':'ABSTAIN','confidence':0,'explanation':'No usable evidence after bounded investigation.','source_ids':[],'evidence_status':'UNRESOLVED'}
    else:
        r=client.complete(sample['sample_id'],payload,FINAL_PROMPT,sample['image_bytes']);calls.append(r)
        verdict=validate_verdict(r['parsed'],payload['evidence'])
    return {'sample_id':sample['sample_id'],'prediction':verdict['prediction'],'status':'OK','verdict':verdict,
            'payload':payload,'actions':actions,'rounds':rounds,'calls':calls,'logical_searches':search_count,
            'policy':'bounded_v1: 0.6 provisional directional probability; <=2 searches; <=4 rounds; unavailable tools explicit'}
