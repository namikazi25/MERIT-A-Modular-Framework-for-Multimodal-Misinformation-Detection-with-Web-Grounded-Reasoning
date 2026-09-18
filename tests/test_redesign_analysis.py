import unittest
from scripts.redesign_analysis import paired

class PairedTests(unittest.TestCase):
    def test_fixed_new_errors_and_nonanswers_are_separate(self):
        truth={'a':'Fake','b':'True','c':'Fake'};groups={'a':'g1','b':'g2','c':'g1'}
        def row(pred):return {'status':'OK','prediction':pred}
        left={'a':row('Not Misinformation'),'b':row('Not Misinformation'),'c':row('ABSTAIN')}
        right={'a':row('Misinformation'),'b':row('Misinformation'),'c':row('Misinformation')}
        r=paired(truth,left,right,groups,draws=100)
        self.assertEqual(r['errors_fixed'],1);self.assertEqual(r['new_errors'],1)
        self.assertEqual(r['correct_after_previous_nonanswer'],1);self.assertEqual(r['n_verified_groups'],2)
        self.assertAlmostEqual(r['all_case_accuracy_delta'],1/3)
        self.assertEqual(r,paired(truth,left,right,groups,draws=100))
        with self.assertRaises(ValueError):paired(truth,{'wrong':{}},right,groups)
