import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from PIL import Image
from scripts.redesign.clients import GLMClient,GPTMiniClient,JevClient,ProviderFailure,parse_answers,validate_questions,structured_object
from scripts.redesign.ledger import Ledger,BudgetBlocked
from scripts.redesign_registry import claim_sha256

HEAD='This is a synthetic claim.'
CONFIG={'provider':'baseten','base_url':'https://inference.baseten.co/v1','model':'zai-org/GLM-5.3-Flash','api_key_env':'BASETEN_API_KEY'}

class FakeAdmission:
    def __init__(self,image=None):self.image=image
    def check(self,ident):
        if ident!='synthetic':raise ValueError('Forbidden sample')
        return {'claim_sha256':claim_sha256(HEAD),'image_byte_sha256':hashlib.sha256(self.image or b'').hexdigest()}

class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.ledger=Ledger(Path(self.tmp.name)/'spend',15,applicable_remaining=15)
        self.sdk=Mock()
        self.sdk.chat.completions.create.return_value=NS(model=CONFIG['model'],usage=NS(model_dump=lambda:{'prompt_tokens':20,'completion_tokens':10}),choices=[NS(finish_reason='stop',message=NS(content='{"ok":true}'))])
        self.payload={'headline':HEAD,'evidence':[{'id':'e1','claim_id':'c1','text':'The source says AI-generated illustrations can be real artwork.','kind':'passage','snapshot_id':'s1','private':{'label':'SECRET'}}],'relevancy':{'aligned':True,'explanation':{'secret':'NO'},'raw':'SECRET'},'visual_veracity':{'ai_generated':True,'anomalies':['legitimate',{'SECRET':'NO'}]},'image_path':'/fake/SECRET','label':'SECRET'}
    def client(self,image=None):return GLMClient(CONFIG,FakeAdmission(image),self.ledger,client=self.sdk)
    def test_pinned_mini_boundary_parameters_and_distinct_pricing(self):
        config={'provider':'openai','base_url':'https://api.openai.com/v1','model':'gpt-4o-mini-2024-07-18','api_key_env':'OPENAI_API_KEY'}
        b=io.BytesIO();Image.new('RGB',(8,8),'red').save(b,format='PNG');data=b.getvalue()
        self.sdk.chat.completions.create.return_value.model=config['model']
        client=GPTMiniClient(config,FakeAdmission(data),self.ledger,client=self.sdk)
        client.complete('synthetic',self.payload,'Return JSON',data)
        req=self.sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(req['model'],config['model']);self.assertNotIn('reasoning_effort',req)
        self.assertNotIn('SECRET',json.dumps(req));self.assertNotIn('image_path',json.dumps(req))
        self.assertEqual(base64.b64decode(req['messages'][1]['content'][1]['image_url']['url'].split(',')[1]),data)
        self.assertEqual(self.ledger.summary()['spent_usd'],'0.000009')
        events=[json.loads(line) for line in self.ledger.path.read_text().splitlines()]
        reserve=next(e for e in events if e['event']=='reserve')
        self.assertEqual(reserve['maximum_usd'],'0.0198144')
        with self.assertRaises(ValueError):GPTMiniClient(CONFIG,FakeAdmission(),self.ledger,client=self.sdk)
        with self.assertRaises(ValueError):GPTMiniClient(dict(config,model='gpt-4o-mini'),FakeAdmission(),self.ledger,client=self.sdk)
        self.sdk.reset_mock()
        with self.assertRaises(ValueError):client.complete('reserved',self.payload,'JSON',data)
        self.sdk.chat.completions.create.assert_not_called()
    def test_image_bytes_and_typed_signal_payload_at_sdk_boundary(self):
        b=io.BytesIO();Image.new('RGB',(8,8),'red').save(b,format='PNG');data=b.getvalue()
        result=self.client(data).complete('synthetic',self.payload,'Return JSON',data)
        req=self.sdk.chat.completions.create.call_args.kwargs
        content=req['messages'][1]['content']
        self.assertEqual(base64.b64decode(content[1]['image_url']['url'].split(',')[1]),data)
        self.assertNotIn('SECRET',json.dumps(req))
        self.assertNotIn('image_path',json.dumps(req))
        self.assertIn('AI-generated illustrations',json.dumps(req))
        self.assertTrue(result['parsed']['ok'])
        self.assertEqual(self.ledger.summary()['pending_ids'],[])
    def test_reserved_unknown_claim_and_image_mismatch_rejected_before_sdk(self):
        for ident,payload,image in [('reserved',self.payload,None),('unknown',self.payload,None),('synthetic',{'headline':'Different claim'},None),('synthetic',self.payload,b'wrong')]:
            with self.assertRaises(ValueError):self.client().complete(ident,payload,'JSON',image)
        self.sdk.chat.completions.create.assert_not_called()
    def test_malformed_nested_fields_never_stringified(self):
        for field,value in [('evidence',[{'text':{'SECRET':'NO'}}]),('claims',[['SECRET']]),('checks',[{'probability':{'SECRET':'NO'}}])]:
            with self.assertRaises(ValueError):self.client().complete('synthetic',dict(self.payload,**{field:value}),'JSON')
        self.sdk.chat.completions.create.assert_not_called()
    def test_timeout_keeps_reservation(self):
        self.sdk.chat.completions.create.side_effect=TimeoutError('private credentials should not print')
        with self.assertRaisesRegex(ProviderFailure,'TimeoutError'):self.client().complete('synthetic',self.payload,'JSON')
        self.assertEqual(len(self.ledger.summary()['pending_ids']),1)
    def test_usage_missing_keeps_reservation(self):
        self.sdk.chat.completions.create.return_value.usage=None
        with self.assertRaises(ProviderFailure):self.client().complete('synthetic',self.payload,'JSON')
        self.assertEqual(len(self.ledger.summary()['pending_ids']),1)
    def test_budget_denied_before_sdk_and_no_fallback(self):
        l=Ledger(Path(self.tmp.name)/'unknown',15)
        c=GLMClient(CONFIG,FakeAdmission(),l,client=self.sdk)
        with self.assertRaises(BudgetBlocked):c.complete('synthetic',self.payload,'JSON')
        self.sdk.chat.completions.create.assert_not_called()
        with self.assertRaises(ValueError):GLMClient(dict(CONFIG,model='another'),FakeAdmission(),l)
    def test_wrong_model_or_truncated_response_is_not_success(self):
        self.sdk.chat.completions.create.return_value.model='wrong'
        with self.assertRaises(ProviderFailure):self.client().complete('synthetic',self.payload,'JSON')
        self.sdk.chat.completions.create.return_value.model=CONFIG['model']
        self.sdk.chat.completions.create.return_value.choices[0].finish_reason='length'
        with self.assertRaises(ProviderFailure):self.client().complete('synthetic',self.payload,'JSON')
    def test_jev_typed_boundary_retains_probabilities_without_private_fields(self):
        transport=Mock(return_value={'model':'jev-1.13.0','answers':{'support':{'type':'noul','noul':.13}},'usage':{'input_tokens':100,'output_tokens':4}})
        c=JevClient(FakeAdmission(),self.ledger,transport=transport)
        result=c.ask('synthetic',self.payload,{'support':{'type':'noul','instructions':'Does passage support claim?','raw':'SECRET'}})
        self.assertEqual(result['answers']['support']['noul'],.13)
        self.assertNotIn('SECRET',json.dumps(transport.call_args.args))
        self.assertEqual(result['model'],'jev-1.13.0')
    def test_jev_missing_input_does_not_dispatch(self):
        transport=Mock();c=JevClient(FakeAdmission(),self.ledger,transport=transport)
        result=c.ask('synthetic',{'headline':HEAD},{'x':{'type':'noul','instructions':'Support?'}})
        self.assertEqual(result['status'],'MISSING_INPUT');transport.assert_not_called()
    def test_choice_score_and_malformed_probabilities(self):
        qs=validate_questions({'a':{'type':'choice','instructions':'Choose','criteria':{'yes':'yes','no':'no'}},'b':{'type':'score','instructions':'Rate','criteria':['low','high']}})
        good={'a':{'type':'choice','choice':'yes','probabilities':{'yes':.7,'no':.3},'confidence':.7},'b':{'type':'score','score':.4,'legend':{'0':'low','1':'high'},'probabilities':{'0':.6,'1':.4},'confidence':.6}}
        self.assertEqual(parse_answers(qs,good)['b']['score'],.4)
        good['a']['probabilities']['yes']={'bad':1}
        with self.assertRaises(ValueError):parse_answers(qs,good)

    def test_explicit_json_fence_only_and_failed_response_audited(self):
        self.assertEqual(structured_object('```json\n{"x":1}\n```'),{'x':1})
        for bad in ['Here is my guess {"x":1}','```json\n{}\n``` trailing','[1]']:
            with self.assertRaises(ProviderFailure):structured_object(bad)
        audit=Path(self.tmp.name)/'audit'
        self.sdk.chat.completions.create.return_value.choices[0].message.content='not JSON'
        c=GLMClient(CONFIG,FakeAdmission(),self.ledger,client=self.sdk,audit_dir=audit)
        with self.assertRaises(ProviderFailure):c.complete('synthetic',self.payload,'JSON')
        record=json.loads(next(audit.glob('*.json')).read_text())
        self.assertEqual(record['raw'],'not JSON');self.assertNotIn('SECRET',json.dumps(record))
        self.assertEqual(self.ledger.summary()['pending_ids'],[])

    def test_jev_oversized_payload_rejected_before_reservation_or_dispatch(self):
        transport=Mock();c=JevClient(FakeAdmission(),self.ledger,transport=transport)
        payload=dict(self.payload,evidence=[dict(self.payload['evidence'][0],text='x'*65536)])
        with self.assertRaises(ValueError):c.ask('synthetic',payload,{'support':{'type':'noul','instructions':'Support?'}})
        transport.assert_not_called();self.assertEqual(self.ledger.summary()['pending_ids'],[])
