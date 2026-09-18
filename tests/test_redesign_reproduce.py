import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.redesign.reproduce import digest, make_bundle, read_json, verify_bundle


def fixture():
    return {'truth': {'a': 'Fake', 'b': 'True', 'c': 'True', 'd': 'Fake', 'e': 'True'},
            'groups': {'a': 'g1', 'b': 'g2', 'c': 'g3', 'd': 'g1', 'e': 'g4'},
            'batches': {'synthetic': {'baseline': [
                {'sample_id': 'a', 'status': 'OK', 'prediction': 'Misinformation'},
                {'sample_id': 'b', 'status': 'OK', 'prediction': 'Misinformation'},
                {'sample_id': 'c', 'status': 'OK', 'prediction': 'ABSTAIN'},
                {'sample_id': 'd', 'status': 'ERROR', 'prediction': None}]}}}


class ReproduceTests(unittest.TestCase):
    def test_hand_calculated_selective_metrics_keep_all_denominators(self):
        result = verify_bundle(make_bundle(fixture()))
        metrics = result['metrics']['synthetic']['baseline']
        self.assertEqual(metrics['statuses'], {'answered': 2, 'abstain': 1, 'failure': 1, 'missing': 1})
        self.assertEqual(metrics['confusion_answered'], {'TP': 1, 'TN': 0, 'FP': 1, 'FN': 0})
        self.assertEqual(metrics['accuracy_all'], .2)
        self.assertEqual(metrics['coverage'], .4)
        self.assertEqual(metrics['accuracy_answered'], .5)
        self.assertEqual(metrics['false_flags_all_authentic'], 1 / 3)
        self.assertEqual(metrics['misinformation_detected_all'], .5)

    def test_member_or_prediction_mutation_invalidates_checksum(self):
        original = make_bundle(fixture())
        for key in ('groups', 'batches'):
            changed = copy.deepcopy(original)
            if key == 'groups':
                changed['payload']['groups']['a'] = 'g2'
            else:
                changed['payload']['batches']['synthetic']['baseline'][0]['prediction'] = 'True'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify_bundle(changed)

    def test_rehashed_wrong_metrics_and_wrong_evaluator_still_fail(self):
        for key in ('expected_metrics', 'evaluator_sha256'):
            bundle = make_bundle(fixture())
            if key == 'expected_metrics':
                bundle[key]['synthetic']['baseline']['coverage'] = 1
            else:
                bundle[key] = '0' * 64
            bundle['bundle_sha256'] = digest({k: v for k, v in bundle.items() if k != 'bundle_sha256'})
            with self.assertRaises(ValueError):
                verify_bundle(bundle)

    def test_unknown_ids_duplicates_and_extra_payload_metadata_fail(self):
        for defect in ('unknown', 'duplicate', 'metadata', 'empty', 'group'):
            payload = fixture()
            rows = payload['batches']['synthetic']['baseline']
            if defect == 'unknown': rows[0]['sample_id'] = 'unknown'
            elif defect == 'duplicate': rows[1]['sample_id'] = rows[0]['sample_id']
            elif defect == 'metadata': rows[0]['raw'] = {'private_path': '/fake/label.png'}
            elif defect == 'empty': rows.clear()
            else: del payload['groups']['a']
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                make_bundle(payload)

    def test_json_duplicate_keys_and_nonfinite_values_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.json'
            for text in ('{"id":1,"id":2}', '{"id":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_json(path)
            bundle = make_bundle(fixture())
            path.write_text(json.dumps(bundle))
            self.assertEqual(verify_bundle(read_json(path))['status'], 'VERIFIED')
