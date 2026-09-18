import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.redesign.evaluation import evaluate
from scripts.redesign.ledger import Ledger, BudgetBlocked
from scripts.redesign.guard import Admission
from scripts.redesign_registry_validate import RegistryValidationError


def contend(args):
    path, n = args
    ledger = Ledger(path, '1', applicable_remaining='1')
    try:
        return ledger.reserve('synthetic', '.75')
    except BudgetBlocked:
        return None


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'spend.jsonl'

    def test_processes_cannot_double_spend_balance(self):
        Ledger(self.path, '1', applicable_remaining='1')
        with concurrent.futures.ProcessPoolExecutor(2) as pool:
            ids = list(pool.map(contend, [(str(self.path), n) for n in range(2)]))
        self.assertEqual(sum(x is not None for x in ids), 1)

    def test_unknown_budget_blocks_paid_but_allows_zero(self):
        l = Ledger(self.path, 15)
        with self.assertRaises(BudgetBlocked): l.reserve('glm', '.001')
        ident = l.reserve('local', 0)
        l.settle(ident, 0)

    def test_retries_reserved_and_timeout_stays_reserved_on_resume(self):
        l = Ledger(self.path, 1, applicable_remaining=1)
        l.reserve('glm', '.25', retries=2)
        resumed = Ledger(self.path, 1, applicable_remaining=1)
        self.assertEqual(resumed.summary()['reserved_usd'], '0.75')
        with self.assertRaises(BudgetBlocked): resumed.reserve('glm', '.3')

    def test_policy_cannot_be_reset_and_settlement_is_single_use(self):
        l = Ledger(self.path, 1, applicable_remaining=1)
        with self.assertRaises(BudgetBlocked): Ledger(self.path, 15, applicable_remaining=15)
        ident = l.reserve('glm', '.5')
        l.settle(ident, '.1')
        with self.assertRaises(BudgetBlocked): l.settle(ident, '.1')
        self.assertEqual(l.summary()['spent_usd'], '0.1')

    def test_deadline_concurrency_billing_and_invalid_cost(self):
        l = Ledger(self.path, 1, applicable_remaining=1, deadline='2000-01-01T00:00:00Z')
        with self.assertRaises(BudgetBlocked): l.reserve('x', 0)
        other = Ledger(Path(self.tmp.name)/'other', 1, applicable_remaining=1)
        a = other.reserve('x', 0)
        other.reserve('x', 0)
        with self.assertRaises(BudgetBlocked): other.reserve('x', 0)
        with self.assertRaises(BudgetBlocked): other.settle(a, '.2')
        self.assertTrue(other.summary()['halted'])
        for bad in ('NaN', '-1', 'Infinity'):
            with self.assertRaises(ValueError): other.reserve('x', bad)
        with self.assertRaises(ValueError): other.reserve('x', 0, retries=3)


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_selective_metrics(self):
        truth = {'a':'True','b':'True','c':'Fake','d':'Fake','e':'Fake','f':'True'}
        rows = [{'sample_id':i,'prediction':p} for i,p in [('a','Misinformation'),('b','Not Misinformation'),('c','Misinformation'),('d','ABSTAIN'),('e',{})]]
        m = evaluate(truth, rows)
        self.assertEqual(m['confusion_answered'], {'TP':1,'FP':1,'TN':1,'FN':0})
        self.assertAlmostEqual(m['accuracy_all'], 2/6)
        self.assertAlmostEqual(m['coverage'], .5)
        self.assertAlmostEqual(m['macro_f1_answered'], 2/3)
        self.assertAlmostEqual(m['false_flags_all_authentic'], 1/3)
        self.assertAlmostEqual(m['false_positives_answered_authentic'], 1/2)
        self.assertAlmostEqual(m['misinformation_detected_all'], 1/3)
        self.assertEqual(m['statuses'], {'answered':3,'abstain':1,'malformed':1,'missing':1})

    def test_mismatches_unknown_truth_duplicates_and_zero_match_fail(self):
        for truth, rows in [({},[]),({'a':'Unknown'},[]),({'a':'Fake'},[]),
                            ({'a':'Fake'},[{'sample_id':'aa','prediction':'Fake'}]),
                            ({'a':'Fake'},[{'sample_id':'a'},{'sample_id':'a'}])]:
            with self.assertRaises(ValueError): evaluate(truth, rows)

    def test_failure_and_abstention_are_not_authentic(self):
        m=evaluate({'a':'True','b':'True'},[{'sample_id':'a','prediction':'True','status':'TIMEOUT'}, {'sample_id':'b','prediction':'ABSTAIN'}])
        self.assertEqual(m['accuracy_all'],0)
        self.assertEqual(m['coverage'],0)
        self.assertIsNone(m['false_positives_answered_authentic'])


class GuardTests(unittest.TestCase):
    def test_only_verified_development_core_admitted(self):
        # Synthetic validator makes this independent of protected sample contents.
        class V:
            roles={'development_core':{'core'}}
            by_id={'core':{'canonical_id':'core'}}
            def __init__(self,*a): pass
            def validate_request(self,purpose,ids):
                if ids[0] in ('reserved','unknown'):
                    raise RegistryValidationError('Rejected')
        with patch('scripts.redesign.guard.RegistryValidator',V):
            guard=Admission('.')
            self.assertEqual(guard.check('core')['canonical_id'],'core')
            for ident in ('reserved','unknown','historical','holdout',''):
                with self.assertRaises(RegistryValidationError): guard.check(ident)
