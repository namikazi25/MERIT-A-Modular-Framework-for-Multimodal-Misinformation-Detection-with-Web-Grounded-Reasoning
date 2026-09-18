import concurrent.futures
from datetime import datetime, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.redesign.traffic import Traffic, TrafficBlocked, retry_after_seconds
from scripts.redesign.retrieval import Discovery, Extraction, SnapshotStore, public_url
from scripts.redesign.ledger import Ledger


def contend(path):
    try:
        return Traffic(path).start('search', ['search:all', 'engine:duckduckgo'], max_wait=0)
    except TrafficBlocked:
        return None


class TrafficTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = 1000.0
        self.waits = []
        self.traffic = Traffic(self.root / 'traffic.json', clock=lambda: self.now, sleep=self.advance)

    def advance(self, seconds):
        self.waits.append(seconds)
        self.now += seconds

    def test_search_spacing_survives_new_adapter_instance(self):
        token = self.traffic.start('search', ['search:all', 'engine:duckduckgo'])
        self.traffic.finish(token)
        second = Traffic(self.traffic.path, clock=lambda: self.now, sleep=self.advance)
        second.finish(second.start('search', ['search:all', 'engine:brave']))
        self.assertEqual(sum(self.waits), 30)
        self.assertLessEqual(max(self.waits), 10)

    def test_global_scrape_spacing_and_per_host_spacing(self):
        for host in ('a.example', 'b.example', 'a.example'):
            token = self.traffic.start('extract', ['extract:all', 'host:' + host])
            self.traffic.finish(token)
        self.assertEqual(sum(self.waits), 30)
        self.assertEqual(self.waits[0], 6)

    def test_parallel_processes_cannot_both_dispatch(self):
        with concurrent.futures.ProcessPoolExecutor(2) as pool:
            results = list(pool.map(contend, [str(self.traffic.path)] * 2))
        self.assertEqual(sum(r is not None for r in results), 1)
        # Simulated interrupted lease stays blocked even after elapsed time.
        with self.assertRaises(TrafficBlocked):
            Traffic(self.traffic.path).start('search', ['search:all'], max_wait=0)

    def test_retry_after_and_access_block_persist_without_auto_reopening(self):
        token = self.traffic.start('search', ['search:all', 'engine:duckduckgo'])
        self.traffic.finish(token, blocked_scopes=['engine:duckduckgo'], access=True, retry_after='100000')
        with self.assertRaises(TrafficBlocked) as raised:
            self.traffic.start('search', ['search:all', 'engine:duckduckgo'])
        self.assertEqual(raised.exception.retry_at, 101000)
        self.now += 200000
        with self.assertRaises(TrafficBlocked):
            self.traffic.start('search', ['search:all', 'engine:duckduckgo'])
        # Unrelated engine is not declared blocked; global spacing still applies.
        self.traffic.finish(self.traffic.start('search', ['search:all', 'engine:bing']))

    def test_retry_after_http_date_invalid_and_longer_values(self):
        date = format_datetime(datetime.fromtimestamp(1100, timezone.utc), usegmt=True)
        self.assertEqual(retry_after_seconds(date, 1000), 100)
        self.assertEqual(retry_after_seconds('999999', 1000), 999999)
        for v in (None, 'NaN', '-1', 'garbage'):
            self.assertEqual(retry_after_seconds(v, 1000), 0)

    def test_corrupt_state_and_deadline_do_not_dispatch(self):
        self.traffic.path.write_text('{}')
        with self.assertRaises(TrafficBlocked):
            self.traffic.start('search', ['search:all'])
        other = Traffic(self.root / 'other.json', clock=lambda: self.now, sleep=self.advance)
        with self.assertRaises(TrafficBlocked):
            other.start('search', ['search:all'], deadline='1970-01-01T00:00:01Z')

    def test_invalid_timestamp_policy_and_backwards_clock_fail_closed(self):
        self.traffic.finish(self.traffic.start('search', ['search:all']))
        saved = self.traffic.path.read_text()
        for part in ('policy', 'last_time', 'services'):
            state = json.loads(saved)
            if part == 'policy':
                state['policy']['search_interval_s'] = 0
            elif part == 'last_time':
                state['last_time'] = float('nan')
            else:
                state['services']['search']['next_at'] = -1
            self.traffic.path.write_text(json.dumps(state))
            with self.subTest(part=part), self.assertRaises(TrafficBlocked):
                self.traffic.start('search', ['search:all'])
        self.traffic.path.write_text(saved)
        self.now -= 1
        with self.assertRaises(TrafficBlocked):
            self.traffic.start('search', ['search:all'])

    def test_prior_observed_failure_blocks_fresh_instance(self):
        self.traffic.record_observed_block('engine:duckduckgo', 999, reason='Recorded challenge')
        fresh = Traffic(self.traffic.path, clock=lambda: self.now, sleep=self.advance)
        with self.assertRaisesRegex(TrafficBlocked, 'Recorded challenge'):
            fresh.start('search', ['search:all', 'engine:duckduckgo'])

    def test_direct_ddgs_remains_blocked_independently_of_searxng(self):
        self.traffic.record_observed_block('route:direct_ddgs',999,reason='Unrepaired direct route')
        guard,ledger,store=self.adapters()
        transport=Mock()
        result=Discovery(guard,ledger,store,backend='duckduckgo',transport=transport,
                         traffic=self.traffic).search('core','query')
        self.assertEqual(result['mode'],'NO_DISPATCH')
        transport.assert_not_called()
        good=Mock(return_value=({'results':[]},200))
        Discovery(guard,ledger,store,backend='searxng',endpoint='http://127.0.0.1:8080',
                  transport=good,traffic=self.traffic).search('core','query')
        good.assert_called_once()

    def adapters(self):
        return Mock(), Ledger(self.root / 'spend.jsonl', 1, applicable_remaining=1), SnapshotStore(self.root / 'snapshots')

    def test_local_http_429_stops_later_search_without_transport_call(self):
        guard, ledger, store = self.adapters()
        transport = Mock(return_value=({}, 429))
        kwargs = dict(backend='searxng', endpoint='http://127.0.0.1:8080', transport=transport, traffic=self.traffic)
        first = Discovery(guard, ledger, store, **kwargs).search('core', 'first')
        second = Discovery(guard, ledger, store, **kwargs).search('core', 'next')
        self.assertEqual(first['api_status'], 429)
        self.assertEqual(second['mode'], 'NO_DISPATCH')
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(ledger.summary()['reserved_usd'], '0')

    def test_admission_rechecked_after_pacing_before_dispatch(self):
        self.traffic.finish(self.traffic.start('search', ['search:all']))
        guard, ledger, store = self.adapters()
        guard.check.side_effect = [None, ValueError('Registry changed while waiting')]
        transport = Mock()
        with self.assertRaisesRegex(ValueError, 'Registry changed'):
            Discovery(guard, ledger, store, backend='duckduckgo', transport=transport,
                      traffic=self.traffic).search('core', 'query')
        self.assertEqual(sum(self.waits), 30)
        transport.assert_not_called()
        self.assertEqual(ledger.summary()['pending_ids'], [])
        self.assertIsNone(json.loads(self.traffic.path.read_text())['services']['search']['active'])

    def test_target_429_blocks_only_that_host_and_replay_still_works(self):
        guard, ledger, store = self.adapters()
        transport = Mock(return_value=({'success': True, 'data': {'markdown': '', 'metadata': {'statusCode': 429}}}, 200))
        e = Extraction(guard, ledger, store, endpoint='http://127.0.0.1:3002/v2/scrape',
                       transport=transport, resolver=lambda u: public_url(u, resolve=False), traffic=self.traffic)
        first = e.scrape('core', 'https://a.example/')
        self.assertEqual(e.scrape('core', 'https://a.example/')['mode'], 'NO_DISPATCH')
        self.assertEqual(e.scrape('core', 'https://a.example/', replay=first['snapshot_id'])['mode'], 'REPLAY')
        e.scrape('core', 'https://b.example/')
        self.assertEqual(transport.call_count, 2)

    def test_api_backoff_header_is_observed_at_http_boundary(self):
        from unittest.mock import patch
        guard, ledger, store = self.adapters()
        response = Mock(status_code=429, headers={'Retry-After': '100000'})
        response.json.return_value = {}
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.get.return_value = response
        with patch('httpx.Client', return_value=client):
            Discovery(guard, ledger, store, backend='searxng', endpoint='http://127.0.0.1:8080', traffic=self.traffic).search('core', 'query')
        state = json.loads(self.traffic.path.read_text())
        self.assertEqual(state['scopes']['search:all']['retry_at'], 101000)
