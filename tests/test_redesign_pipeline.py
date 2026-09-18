import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from scripts.redesign.pipeline import Features,run_frozen,decide,select_evidence,extract_claims,validate_verdict
from scripts.redesign.baseline import historical_baseline
from scripts.redesign.clients import ProviderFailure

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.sample={'sample_id':'core','headline':'A red square exists.','image_bytes':b'image','image_path':'synthetic.png'}
        self.extracted={'claims':[{'id':'c1','text':'A red square exists.','source_span':'A red square exists.'}],'image_role':'illustration'}
        self.evidence=[{'id':'e1','claim_id':'c1','text':'A source describes a red square.','kind':'passage','url':'https://example.com','snapshot_id':'snapshot'}]
        self.client=Mock()
        self.client.complete.return_value={'parsed':{'prediction':'Not Misinformation','confidence':.8,'source_ids':['e1@c1'],'evidence_status':'SUPPORTS','explanation':'Source e1 supports c1.'}}
        self.bank={'templates':[{'id':'support','type':'noul','instructions':'Support?'},{'id':'contradiction','type':'noul','instructions':'Contradiction?'}]}
    def test_evidence_reaches_final_request_with_traceable_citation(self):
        r=run_frozen(self.sample,self.extracted,self.evidence,[],self.client)
        payload=self.client.complete.call_args.args[1]
        self.assertEqual(payload['evidence'][0]['text'],self.evidence[0]['text'])
        self.assertEqual(r['verdict']['source_ids'],['e1@c1']);self.assertEqual(r['payload']['evidence'][0]['snapshot_id'],'snapshot')
    def test_missing_and_failed_evidence_defer_not_contradict(self):
        r=run_frozen(self.sample,self.extracted,[],[{'tool':'search','status':'TOOL_ERROR'}],self.client)
        self.assertEqual(r['prediction'],'ABSTAIN');self.assertEqual(r['verdict']['evidence_status'],'UNRESOLVED');self.client.complete.assert_not_called()
    def test_flags_keep_unavailable_tools_explicit_and_budget_stops(self):
        r=run_frozen(self.sample,self.extracted,self.evidence,[],self.client,features=Features(image_role=False,detector=True,image_origin=True))
        self.assertIsNone(r['payload']['image_role']);self.assertEqual([s['status'] for s in r['payload']['tool_status']],['UNAVAILABLE','UNAVAILABLE'])
        self.assertEqual(decide(r['payload'],4,Features(),searched=True,can_search=True),'DEFER_UNRESOLVED')
    def test_jev_probabilities_not_multiplied_or_inverted(self):
        j=Mock();j.ask.return_value={'status':'OK','answers':{'support':{'noul':.1},'contradiction':{'noul':.2}}}
        r=run_frozen(self.sample,self.extracted,self.evidence,[],self.client,features=Features(evidence_checker='jev'),jev=j,bank=self.bank)
        self.assertEqual([x['probability'] for x in r['payload']['checks']],[.1,.2])
        self.assertEqual(decide(r['payload'],0,Features(),searched=True,can_search=True),'REFORMULATE_QUERY')
        with self.assertRaises(ProviderFailure):run_frozen(self.sample,self.extracted,self.evidence,[],self.client,features=Features(evidence_checker='jev'),bank=self.bank)
    def test_malformed_extraction_and_unknown_citation_fail(self):
        self.client.complete.return_value={'parsed':{'claims':[{'id':'c','text':'Invented','source_span':'Invented'}],'query':'q'}}
        with self.assertRaises(ProviderFailure):extract_claims(self.sample,self.client)
        self.client.complete.return_value={'parsed':{'prediction':'Misinformation','confidence':.8,'explanation':'Computed rationale.','source_ids':['invented'],'evidence_status':'CONTRADICTS'}}
        with self.assertRaises(ProviderFailure):run_frozen(self.sample,self.extracted,self.evidence,[],self.client)
    def test_evidence_cap_records_omissions(self):
        selected,omitted=select_evidence(self.evidence,1)
        self.assertEqual(selected,[]);self.assertEqual(omitted,['e1'])

    def test_binary_verdict_with_unresolved_evidence_or_no_citation_is_rejected(self):
        # Synthetic counterpart of the observed A2 replay grounding failure.
        for status,cites in [('UNRESOLVED',[]),('UNRESOLVED',['e1']),('SUPPORTS',[])]:
            v={'prediction':'Misinformation','confidence':.85,'explanation':'Prior knowledge, not these sources.',
               'source_ids':cites,'evidence_status':status}
            with self.assertRaisesRegex(ProviderFailure,'Unsupported binary verdict'):validate_verdict(v,self.evidence)
        valid={'prediction':'ABSTAIN','confidence':0,'explanation':'Unresolved.','source_ids':[],'evidence_status':'UNRESOLVED'}
        self.assertEqual(validate_verdict(valid,self.evidence),valid)

class HistoricalBridgeTests(unittest.TestCase):
    def test_original_three_chains_and_best_qa_run_with_clean_final_payload(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fake_label.png';Image.new('RGB',(8,8),'red').save(path)
            sample={'sample_id':'core','headline':'A red square exists.','image_path':path}
            outputs=[{'aligned':True,'confidence':.9,'explanation':'aligned'}, {'ai_generated':False,'confidence':.9,'explanation':'photo','anomalies':[]}]
            for chain in range(3):
                outputs.append([f'Question {chain}-{i}?' for i in range(3)])
                outputs.extend([{'answer':'source says yes','confidence':.8,'citations':[{'url':'https://example.com'}],'rationale':'source'} for _ in range(3)])
                outputs.append({'index':1})
                outputs.append([f'Unused followup {chain}-{i}?' for i in range(3)])
            outputs.append({'label':'Not Misinformation','confidence':.8,'rationale':'support'})
            native=Mock(side_effect=[{'raw':json.dumps(x)} for x in outputs]);client=Mock();client.native=native
            evidence=[{'url':'https://example.com','text':'Source explicitly supports red square.'}]
            collect=Mock(return_value=(evidence,[{'tool':'search','status':'SUCCESS'}]))
            result=historical_baseline(sample,client,collect)
            self.assertEqual(collect.call_count,9);self.assertEqual(native.call_count,21)
            self.assertEqual(result['prediction'],'Not Misinformation')
            self.assertEqual(len(result['signals']['best_qa_per_chain']),3)
            self.assertNotIn('fake_label',json.dumps(native.call_args.args))
