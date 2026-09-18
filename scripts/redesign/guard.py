"""Admission is enforced by each redesign service before any external dispatch."""
from pathlib import Path
from scripts.redesign_registry_validate import RegistryValidator, RegistryValidationError


class Admission:
    def __init__(self, root):
        self.root = Path(root)

    def check(self, sample_id):
        if type(sample_id) is not str or not sample_id:
            raise RegistryValidationError('Canonical sample ID required')
        v = RegistryValidator(self.root / 'manifests/redesign_data_registry_v2.json',
                              self.root / 'manifests/redesign_sample_registry_v2.jsonl', self.root)
        v.validate_request('development', [sample_id])
        if sample_id not in v.roles['development_core']:
            raise RegistryValidationError('Only the approved 100-example development core is authorised')
        return v.by_id[sample_id].copy()
