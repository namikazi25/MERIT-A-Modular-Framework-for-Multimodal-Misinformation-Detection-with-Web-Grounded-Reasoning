import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from scripts.redesign.retrieval import public_url,deduplicate,SnapshotStore,Discovery,Extraction,searx_result,extraction_result,passages,RetrievalFailure
from scripts.redesign.ledger import Ledger
from scripts.redesign.traffic import Traffic

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=SnapshotStore(Path(self.tmp.name)/'snapshots')
        self.ledger=Ledger(Path(self.tmp.name)/'spend',1,applicable_remaining=1)
        self.admission=Mock()
        self.now=1000.0
        self.traffic=Traffic(Path(self.tmp.name)/"traffic.json",clock=lambda:self.now,sleep=self.advance)
    def advance(self,seconds):
        self.now+=seconds
    def test_private_schemes_dns_credentials_and_redirects_fail(self):
        for u in ['file:///etc/passwd','http://127.0.0.1','http://[::1]/','http://localhost','https://user:secret@example.com','http://10.0.0.1','http://example.com:8080']:
            with self.assertRaises(ValueError):public_url(u,resolve=False)
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('192.168.1.5',443))]):
            with self.assertRaises(ValueError):public_url('https://example.com')
        r=extraction_result({'success':True,'data':{'markdown':'a','metadata':{'url':'http://127.0.0.1','statusCode':200}}},200,'https://example.com')
        self.assertEqual(r['status'],'UNSAFE_REDIRECT')
    def test_partial_search_json_disabled_and_deduplication(self):
        raw={'results':[{'url':'https://example.com','content':'one','engines':['bing']},{'url':'https://example.com/#x','content':'two'}],'unresponsive_engines':[['duckduckgo','timeout']]}
        r=searx_result(raw,200,['duckduckgo','bing'],5)
        self.assertEqual(r['status'],'PARTIAL_FAILURE');self.assertEqual(len(r['results']),1)
        self.assertEqual(r['responding_engines'],['bing']);self.assertIsNone(r['upstream_requests'])
        self.assertEqual(searx_result({},403,[],5)['detail'],'JSON_DISABLED_OR_ACCESS_DENIED')
        self.assertEqual(searx_result('html',200,[],5)['status'],'TOOL_ERROR')
    def test_target_status_empty_blocked_truncation_unknown_dates(self):
        def raw(body,status=200):return {'success':True,'data':{'markdown':body,'metadata':{'statusCode':status}}}
        for body,status,wanted in [('Article',404,'TARGET_ERROR'),('',200,'EMPTY_OR_TRUNCATED'),('Please verify you are human',200,'BLOCKED_OR_LOGIN'),('x'*20,200,'EMPTY_OR_TRUNCATED')]:
            self.assertEqual(extraction_result(raw(body,status),200,'https://example.com',max_chars=10)['status'],wanted)
        r=extraction_result(raw('Complete article'),200,'https://example.com')
        self.assertEqual(r['status'],'USABLE');self.assertEqual(r['date_origin'],'unknown');self.assertIsNone(r['publication_date'])
    def test_snapshot_replay_verifies_config_and_hash_without_dispatch(self):
        transport=Mock(return_value=({'results':[{'url':'https://example.com','content':'Passage','engines':['duckduckgo']}]},200))
        d=Discovery(self.admission,self.ledger,self.store,traffic=self.traffic,backend='searxng',endpoint='http://127.0.0.1:8080',transport=transport)
        r=d.search('core','query');replay=d.search('core','query',replay=r['snapshot_id'])
        self.assertEqual(replay['mode'],'REPLAY');self.assertEqual(transport.call_count,1)
        with self.assertRaises(RetrievalFailure):d.search('core','changed',replay=r['snapshot_id'])
        p=self.store.path/(r['snapshot_id']+'.json');p.write_text('{}')
        with self.assertRaises(ValueError):self.store.get(r['snapshot_id'])
    def test_passage_traceability_and_api_request_no_generated_formats(self):
        transport=Mock(return_value=({'success':True,'data':{'markdown':'First passage.\n\nSecond passage.','metadata':{'statusCode':200,'url':'https://example.com/next'}}},200))
        e=Extraction(self.admission,self.ledger,self.store,traffic=self.traffic,endpoint='http://127.0.0.1:3002/v2/scrape',transport=transport,resolver=lambda u:public_url(u,resolve=False))
        r=e.scrape('core','https://example.com/');ps=passages(r,'c1')
        self.assertEqual(len(ps),2);self.assertEqual(ps[0]['claim_id'],'c1');self.assertEqual(ps[0]['snapshot_id'],r['snapshot_id'])
        self.assertEqual(transport.call_args.args[1]['formats'],['markdown'])
        replay=e.scrape('core','https://example.com/',replay=r['snapshot_id']);self.assertEqual(replay['mode'],'REPLAY');self.assertEqual(transport.call_count,1)
    def test_guard_runs_before_discovery_or_extraction_dispatch(self):
        self.admission.check.side_effect=ValueError('reserved')
        transport=Mock()
        d=Discovery(self.admission,self.ledger,self.store,traffic=self.traffic,backend='duckduckgo',transport=transport)
        e=Extraction(self.admission,self.ledger,self.store,traffic=self.traffic,endpoint='http://127.0.0.1:3002/v2/scrape',transport=transport)
        with self.assertRaises(ValueError):d.search('reserved','query')
        with self.assertRaises(ValueError):e.scrape('reserved','https://example.com')
        transport.assert_not_called()
    def test_failed_discovery_is_visible_and_never_relabelled_empty_success(self):
        d=Discovery(self.admission,self.ledger,self.store,traffic=self.traffic,backend='duckduckgo',transport=Mock(side_effect=TimeoutError()))
        r=d.search('core','query');self.assertEqual(r['status'],'TOOL_ERROR');self.assertEqual(r['retries'],0)

    def test_rate_limit_blocks_later_queries_across_instances_but_replay_works(self):
        transport=Mock(return_value=({'results':[],'unresponsive_engines':[['brave','too many requests']]},200))
        args=dict(backend='searxng',endpoint='http://127.0.0.1:8080',engines=('brave',),transport=transport)
        first=Discovery(self.admission,self.ledger,self.store,traffic=self.traffic,**args).search('core','first')
        second=Discovery(self.admission,self.ledger,self.store,traffic=self.traffic,**args).search('core','second')
        self.assertEqual(second['mode'],'NO_DISPATCH');self.assertEqual(second['logical_queries'],0)
        self.assertEqual(transport.call_count,1)
        replay=Discovery(self.admission,self.ledger,self.store,traffic=self.traffic,**args).search('core','first',replay=first['snapshot_id'])
        self.assertEqual(replay['mode'],'REPLAY');self.assertEqual(transport.call_count,1)

    def test_consent_pages_are_not_usable_evidence(self):
        raw={'success':True,'data':{'markdown':'Before you continue to YouTube\n\nAccept all cookies','metadata':{'statusCode':200}}}
        self.assertEqual(extraction_result(raw,200,'https://example.com')['status'],'BLOCKED_OR_LOGIN')
