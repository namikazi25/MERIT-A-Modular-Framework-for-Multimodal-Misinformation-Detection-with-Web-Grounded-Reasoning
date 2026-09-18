import unittest
from scripts.redesign_expansion import averitec,averimatec,classification_diagnostic,AVERITEC_LABELS

class ExpansionTests(unittest.TestCase):
    def test_text_fixture_separates_gold_and_provenance(self):
        r=averitec({'claim':'Synthetic red square.','label':'Supported','questions':[{'answer':'PRIVATE GOLD'}],
                    'fact_checking_article':'https://example.org/label-bearing','debug':'PRIVATE'},split='synthetic',index=0)
        self.assertEqual(r['model_input'],{'claim':'Synthetic red square.','image_references':[]})
        self.assertFalse(r['execution_authorized']);self.assertEqual(r['evaluation_only']['label'],'Supported')

    def test_image_fixture_preserves_distinct_references_without_loading_them(self):
        r=averimatec({'claim_text':'Synthetic blue square.','claim_images':['a/photo.png','b/photo.png'],
                      'article':'https://example.org/factcheck','date':'2026','location':'GB'},split='synthetic',index=2)
        self.assertEqual(r['model_input']['image_references'],['a/photo.png','b/photo.png'])
        self.assertIsNone(r['evaluation_only']['label']);self.assertFalse(r['execution_authorized'])
        with self.assertRaises(ValueError):averimatec({'claim_text':'x','claim_images':[{}]},split='dev',index=0)

    def test_four_class_nei_is_answer_not_abstention_and_identity_errors_fail(self):
        truth={'a':'Not Enough Evidence','b':'Refuted','c':'Supported'}
        rows=[{'sample_id':'a','prediction':'Not Enough Evidence','status':'OK'},
              {'sample_id':'b','prediction':'ABSTAIN','status':'OK'}]
        r=classification_diagnostic(truth,rows,AVERITEC_LABELS)
        self.assertEqual(r['accuracy_all'],1/3);self.assertEqual(r['statuses'],{'answered':1,'abstain':1,'missing':1})
        for bad in ([],[{'sample_id':'foreign'}],rows+rows):
            with self.assertRaises(ValueError):classification_diagnostic(truth,bad,AVERITEC_LABELS)
        self.assertIsNone(r['official_evidence_score'])
