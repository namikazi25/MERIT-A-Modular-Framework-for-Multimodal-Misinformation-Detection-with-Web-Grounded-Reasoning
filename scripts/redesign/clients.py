"""Explicit GLM and Jev clients with typed requests, no fallback and zero hidden retries."""
import base64
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import time
from PIL import Image
from scripts.redesign import payloads
from scripts.redesign.ledger import BudgetBlocked
from scripts.redesign_registry import claim_sha256


class ProviderFailure(RuntimeError):
    pass


def bind(admission, sample_id, payload, image=None):
    row=admission.check(sample_id)
    clean=payloads.project(payload)
    if claim_sha256(clean['headline']) != row['claim_sha256']:
        raise ValueError('Claim does not match admitted identity')
    if image is not None and hashlib.sha256(image).hexdigest()!=row['image_byte_sha256']:
        raise ValueError('Image does not match admitted identity')
    return clean


def usage_counts(usage, in_key, out_key):
    if type(usage) is not dict or any(type(usage.get(k)) is not int or usage[k]<0 for k in (in_key,out_key)):
        raise ProviderFailure('Missing or malformed token usage; reservation remains outstanding')
    return usage[in_key],usage[out_key]


def structured_object(raw):
    """Accept JSON or one explicit JSON code fence; never extract a guessed substring."""
    if type(raw) is not str:
        raise ProviderFailure('Missing text output')
    value=raw.strip()
    if value.startswith('```json\n') and value.endswith('\n```'):
        value=value[8:-4]
    try: parsed=json.loads(value)
    except (ValueError,TypeError): raise ProviderFailure('Malformed structured output') from None
    if type(parsed) is not dict: raise ProviderFailure('Structured output must be an object')
    return parsed


class GLMClient:
    def __init__(self, config, admission, ledger, *, client=None, max_output=1024, reasoning='low', audit_dir=None):
        self.config=dict(config); self.admission=admission;self.ledger=ledger
        if config.get('provider')!='baseten' or config.get('base_url')!='https://inference.baseten.co/v1':
            raise ValueError('Explicit reviewed Baseten endpoint required')
        if config.get('model')!='zai-org/GLM-5.3-Flash':
            raise ValueError('Changed model requires a new capability/pricing review')
        if type(max_output) is not int or not 1<=max_output<=8192 or reasoning not in ('low','high','max'):
            raise ValueError('Unsupported output/reasoning limit')
        self.max_output=max_output;self.reasoning=reasoning;self.client=client
        self.audit_dir=Path(audit_dir) if audit_dir else None
        if self.audit_dir:self.audit_dir.mkdir(parents=True,exist_ok=True)

    def _record(self,rid,record):
        if self.audit_dir:
            with (self.audit_dir/(rid+'.json')).open('x') as f:
                json.dump(record,f,indent=2,ensure_ascii=False)

    def complete(self, sample_id, payload, instruction, image=None):
        clean=bind(self.admission,sample_id,payload,image)
        if type(instruction) is not str: raise ValueError('Instruction must be text')
        content=[{'type':'text','text':json.dumps(clean,ensure_ascii=False)}]
        if image is not None:
            with Image.open(io.BytesIO(image)) as im:
                fmt=im.format
                if fmt not in ('PNG','JPEG','WEBP'): raise ValueError('Unsupported image format')
                im.verify()
            mime={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[fmt]
            content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(image).decode()}})
        return self._dispatch([{'role':'system','content':instruction}, {'role':'user','content':content}])

    def native(self,sample_id,messages):
        """Guarded bridge for the frozen stage functions; no provider or heuristic fallback."""
        row=self.admission.check(sample_id)
        clean=[]
        for m in messages:
            if type(m) is not dict or m.get('role') not in ('system','user','assistant'):
                raise ValueError('Malformed native message')
            content=m.get('content')
            if type(content) is list:
                parts=[]
                for part in content:
                    if type(part) is not dict:raise ValueError('Malformed content part')
                    if part.get('type')=='text':parts.append({'type':'text','text':payloads.text(part.get('text'),required=True)})
                    elif part.get('type')=='image_url':
                        image_part=part.get('image_url')
                        url=image_part.get('url') if type(image_part) is dict else image_part
                        if type(url) is not str or not url.startswith(('data:image/png;base64,','data:image/jpeg;base64,','data:image/webp;base64,')):
                            raise ValueError('Only inline admitted image bytes allowed')
                        data=base64.b64decode(url.split(',',1)[1],validate=True)
                        if hashlib.sha256(data).hexdigest()!=row['image_byte_sha256']:raise ValueError('Image identity mismatch')
                        parts.append({'type':'image_url','image_url':{'url':url}})
                    else:raise ValueError('Unsupported native content')
                content=parts
            else:content=payloads.text(content,required=True)
            clean.append({'role':m['role'],'content':content})
        return self._dispatch(clean,structured=False)

    def _dispatch(self,messages,structured=True):
        has_image=any(isinstance(m.get('content'),list) and any(p.get('type')=='image_url' for p in m['content']) for m in messages)
        input_bound=1048576 if has_image else len(json.dumps(messages,ensure_ascii=False).encode())+4096
        if input_bound>(1048576 if has_image else 65536):raise ValueError('Input context bound exceeded')
        maximum=Decimal(input_bound)*Decimal('.15')/1000000 + Decimal(self.max_output)*Decimal('.50')/1000000
        client=self.client
        if client is None:
            key=os.environ.get(self.config['api_key_env'])
            if not key: raise ProviderFailure('Baseten key unavailable in client environment')
        rid=self.ledger.reserve('baseten:GLM-5.3-Flash',maximum)
        started=time.monotonic()
        try:
            if client is None:
                from openai import OpenAI
                client=OpenAI(api_key=key,base_url=self.config['base_url'],max_retries=0,timeout=45)
            response=client.chat.completions.create(model=self.config['model'],
                messages=messages,
                max_tokens=self.max_output,reasoning_effort=self.reasoning)
        except Exception as exc:
            if getattr(exc,'status_code',None) in (402,403):self.ledger.halt('Provider billing/access failure')
            self._record(rid,{'request_id':rid,'status':'TRANSPORT_FAILURE','error_type':type(exc).__name__,
                              'latency_s':time.monotonic()-started,'reservation_retained':True})
            raise ProviderFailure(type(exc).__name__+'; reservation retained') from None
        finally:
            if self.client is None and client is not None: client.close()
        usage=response.usage.model_dump() if response.usage else None
        # Preserve returned content even when later schema/finish/citation checks fail.
        # Never serialize SDK client configuration, request headers or exception bodies.
        choice=response.choices[0] if response.choices else None
        self._record(rid,{'request_id':rid,'status':'RESPONSE','model':response.model,'usage':usage,
                          'finish_reason':choice.finish_reason if choice else None,
                          'raw':choice.message.content if choice else None,'latency_s':time.monotonic()-started})
        n_in,n_out=usage_counts(usage,'prompt_tokens','completion_tokens')
        # Use uncached rate for conservative rated usage; do not claim invoice reconciliation.
        rated=(Decimal(n_in)*Decimal('.15')+Decimal(n_out)*Decimal('.50'))/1000000
        self.ledger.settle(rid,rated,{'prompt_tokens':n_in,'completion_tokens':n_out,'basis':'published_uncached_rate'})
        if response.model!=self.config['model']:raise ProviderFailure('Unexpected returned model')
        if not response.choices or response.choices[0].finish_reason!='stop':raise ProviderFailure('Incomplete model output')
        raw=response.choices[0].message.content
        if type(raw) is not str: raise ProviderFailure('Missing text output')
        if not structured:
            return {'raw':raw,'model':response.model,'usage':usage,'request_id':rid,'latency_s':time.monotonic()-started}
        parsed=structured_object(raw)
        return {'parsed':parsed,'raw':raw,'model':response.model,'usage':usage,'request_id':rid,'latency_s':time.monotonic()-started}


class JevClient:
    model='jev-1.13.0'
    endpoint='https://api.typesafe.ai/v1/systemone'

    def __init__(self, admission, ledger, *, transport=None):
        self.admission=admission;self.ledger=ledger;self.transport=transport

    def ask(self,sample_id,payload,questions):
        clean=bind(self.admission,sample_id,payload)
        qs=validate_questions(questions)
        if len(json.dumps({'model':self.model,'state':clean,'questions':qs},ensure_ascii=False).encode())+4096>65536:
            raise ValueError('Jev input exceeds conservative reservation bound')
        if not clean['evidence']:
            return {'status':'MISSING_INPUT','model':self.model,'answers':{},'usage':None}
        key=os.environ.get('TYPESAFE_API_KEY')
        if self.transport is None and not key: raise ProviderFailure('TYPESAFE_API_KEY unavailable')
        # Published ~32k request token limit; conservative reservation of 64k input tokens.
        rid=self.ledger.reserve('typesafe:'+self.model,Decimal(65536)*Decimal('.042')/1000000)
        try:
            if self.transport is None:
                import httpx
                with httpx.Client(timeout=30,follow_redirects=False,trust_env=False) as client:
                    response=client.post(self.endpoint,headers={'Authorization':'Bearer '+key},json={'model':self.model,'state':clean,'questions':qs})
                    if response.status_code in (402,403):self.ledger.halt('Jev billing/access failure')
                    response.raise_for_status()
                    raw=response.json()
            else: raw=self.transport(self.endpoint,{'model':self.model,'state':clean,'questions':qs})
        except Exception as exc: raise ProviderFailure(type(exc).__name__+'; reservation retained') from None
        n_in,n_out=usage_counts(raw.get('usage'),'input_tokens','output_tokens')
        self.ledger.settle(rid,Decimal(n_in)*Decimal('.042')/1000000,{'input_tokens':n_in,'output_tokens':n_out})
        if raw.get('model')!=self.model: raise ProviderFailure('Unexpected Jev model version')
        answers=parse_answers(qs,raw.get('answers'))
        return {'status':'OK','model':self.model,'answers':answers,'raw':raw,'usage':raw['usage'],'request_id':rid}


def validate_questions(questions):
    if type(questions) is not dict or not questions:raise ValueError('Questions required')
    result={}
    for ident,q in questions.items():
        if type(ident) is not str or not ident or type(q) is not dict:raise ValueError('Malformed question')
        typ=q.get('type');instr=payloads.text(q.get('instructions'),required=True)
        if typ not in ('noul','choice','score'):raise ValueError('Unknown question type')
        obj={'type':typ,'instructions':instr}
        if typ=='choice':
            criteria=q.get('criteria')
            if type(criteria) is not dict or len(criteria)<2 or any(type(k) is not str or type(v) is not str for k,v in criteria.items()):raise ValueError('Invalid choices')
            obj['criteria']=dict(criteria)
        elif typ=='score':
            criteria=q.get('criteria')
            if type(criteria) is not list or len(criteria)<2 or any(type(v) is not str for v in criteria):raise ValueError('Invalid levels')
            obj['criteria']=list(criteria)
        result[ident]=obj
    return result


def parse_answers(questions,answers):
    if type(answers) is not dict or set(answers)!=set(questions):raise ProviderFailure('Missing/extra typed answers')
    result={}
    for ident,q in questions.items():
        a=answers[ident]
        if type(a) is not dict or a.get('type')!=q['type']:raise ProviderFailure('Wrong answer type')
        if q['type']=='noul':result[ident]={'type':'noul','noul':payloads.probability(a.get('noul'))}
        else:
            keys=set(q['criteria']) if q['type']=='choice' else {str(i) for i in range(len(q['criteria']))}
            probs=a.get('probabilities')
            if type(probs) is not dict or set(probs)!=keys:raise ProviderFailure('Wrong distribution alternatives')
            probs={k:payloads.probability(v) for k,v in probs.items()}
            if abs(sum(probs.values())-1)>1e-5:raise ProviderFailure('Probabilities do not sum to one')
            out={'type':q['type'],'probabilities':probs,'confidence':payloads.probability(a.get('confidence'))}
            if q['type']=='choice':
                if a.get('choice') not in keys:raise ProviderFailure('Invalid selected choice')
                out['choice']=a['choice']
            else:
                score=a.get('score')
                if type(score) not in (float,int) or not 0<=score<=len(keys)-1:raise ProviderFailure('Invalid score')
                legend={str(i):v for i,v in enumerate(q['criteria'])}
                if a.get('legend')!=legend:raise ProviderFailure('Wrong score legend')
                out.update(score=score,legend=legend)
            result[ident]=out
    return result
