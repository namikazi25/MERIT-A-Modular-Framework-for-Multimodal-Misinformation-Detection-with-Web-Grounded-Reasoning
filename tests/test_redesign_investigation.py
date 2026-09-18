import unittest
from unittest.mock import Mock
from scripts.redesign.investigation import investigate
from scripts.redesign.pipeline import Features,link_evidence


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.sample={'sample_id':'core','headline':'Red and blue squares exist.','image_bytes':b'synthetic'}
        self.q={'claims':[{'id':'red','text':'Red squares exist.','source_span':'Red'},
                          {'id':'blue','text':'Blue squares exist.','source_span':'blue'}],
                'query':'red blue square evidence','image_role':'illustration'}
        self.bank={'templates':[{'id':'support','type':'noul','instructions':'Support?'},{'id':'contradiction','type':'noul','instructions':'Contradict?'}]}
        self.e=[{'id':'p1','claim_id':'old','text':'No red square exists.','kind':'passage','snapshot_id':'snap','url':'https://example.org'},
                {'id':'p2','claim_id':'old','text':'A blue square exists.','kind':'passage','snapshot_id':'snap','url':'https://example.org'}]
        self.client=Mock()
        def verdict(sid,payload,prompt,*args):
            return {'parsed':{'prediction':'ABSTAIN','confidence':.2,'explanation':'Unresolved source conflict.',
                     'source_ids':[p['id'] for p in payload['evidence']],'evidence_status':'UNRESOLVED'}}
        self.client.complete.side_effect=verdict
        self.jev=Mock();self.jev.ask.return_value={'status':'OK','answers':{'support':{'noul':.1},'contradiction':{'noul':.2}}}

    def test_checks_inside_loop_trigger_bounded_reformulation_and_reach_judge(self):
        collect=Mock(return_value=(self.e,[{'tool':'search','status':'SUCCESS'}]))
        r=investigate(self.sample,self.q,collect,self.client,jev=self.jev,bank=self.bank)
        self.assertEqual(collect.call_count,2)
        self.assertEqual([x['action'] for x in r['actions']],['SEARCH_TEXT','REFORMULATE_QUERY','FINALISE_WITH_GLM'])
        self.assertEqual(len(r['rounds']),3);self.assertEqual(r['payload']['checks'][0]['probability'],.1)
        self.assertEqual({p['claim_id'] for p in r['payload']['evidence']},{'red','blue'})
        self.assertTrue(all(p['snapshot_id']=='snap' for p in r['payload']['evidence']))
        self.assertEqual(self.client.complete.call_args.args[1]['checks'],r['payload']['checks'])

    def test_no_evidence_and_tool_failures_never_create_contradiction(self):
        collect=Mock(return_value=([],[{'tool':'search','status':'TOOL_ERROR'}]))
        r=investigate(self.sample,self.q,collect,self.client,jev=self.jev,bank=self.bank)
        self.assertEqual(r['prediction'],'ABSTAIN');self.assertEqual(r['verdict']['evidence_status'],'UNRESOLVED')
        self.assertEqual(collect.call_count,2);self.jev.ask.assert_not_called();self.client.complete.assert_not_called()

    def test_features_isolate_checking_role_and_unavailable_origin(self):
        collect=Mock()
        r=investigate(self.sample,self.q,collect,self.client,initial_evidence=self.e,
            features=Features(claim_linking=False,image_role=False,evidence_checker='none',image_origin=True,detector=True))
        collect.assert_not_called();self.jev.ask.assert_not_called()
        self.assertIsNone(r['payload']['image_role']);self.assertEqual(r['payload']['claims'][0]['text'],self.sample['headline'])
        self.assertEqual([s['status'] for s in r['payload']['tool_status']],['UNAVAILABLE','UNAVAILABLE'])

    def test_step_ceiling_and_duplicate_query_do_not_loop(self):
        collect=Mock(return_value=([],[]))
        r=investigate(self.sample,self.q,collect,self.client,features=Features(max_steps=1))
        collect.assert_not_called();self.assertEqual(r['prediction'],'ABSTAIN')
        q=dict(self.q,query=self.q['claims'][0]['text'])
        r=investigate(self.sample,q,collect,self.client,jev=self.jev,bank=self.bank)
        self.assertEqual(collect.call_count,1);self.assertEqual(r['actions'][-1]['action'],'DEFER_DUPLICATE_QUERY')

    def test_lexical_candidate_links_preserve_contradiction_and_distinct_claims(self):
        linked=link_evidence(self.q['claims'],self.e)
        self.assertEqual(linked[0]['text'],'No red square exists.')
        self.assertEqual({x['claim_id'] for x in linked},{'red','blue'})
        self.assertEqual(len({x['id'] for x in linked}),len(linked))

if __name__=='__main__':unittest.main()
