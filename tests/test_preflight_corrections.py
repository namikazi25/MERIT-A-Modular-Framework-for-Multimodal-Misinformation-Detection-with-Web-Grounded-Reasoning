"""Focused regression tests for the Task 0.2.1 preflight corrections (offline).

Covers the four reported findings and the coordinator review items:
  F1 integrity binding (membership, dependencies, exposure, aliases, provenance),
  F2 complete exclusion propagation,
  F3 alias ambiguity handling,
  F4 nested model-input boundary,
plus overwrite protection, refusal of caller-declared dependency subsets, and refusal of
mutated in-memory state.

Run with:  python -m unittest tests.test_preflight_corrections -v
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# No credential loading and no network access from this module.
if "dotenv" not in sys.modules:
    _stub = types.ModuleType("dotenv")
    _stub.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = _stub

import scripts.redesign_registry as rr
import scripts.redesign_registry_validate as rv
from scripts.redesign_registry_validate import RegistryValidationError, RegistryValidator


class NetworkGuard:
    def __enter__(self):
        self._orig = socket.socket.connect

        def _blocked(*a, **k):
            raise AssertionError("OFFLINE TEST VIOLATION: network access attempted")

        socket.socket.connect = _blocked
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._orig
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_rows(specs):
    """specs: (tag, rel, byte_sha, pixel_sha, claim_text[, overrides])."""
    rows = []
    for index, spec in enumerate(specs):
        tag, rel, byte_sha, pixel_sha, text = spec[:5]
        overrides = spec[5] if len(spec) > 5 else {}
        row = {
            "canonical_id": rr.canonical_sample_id(rel, text),
            "dataset_index": index,
            "image_path_rel": rel,
            "image_byte_sha256": byte_sha,
            "image_pixel_sha256": pixel_sha,
            "claim_sha256": rr.claim_sha256(text),
            "dhash64": "0" * 16,
            "reference_status": "resolved",
            "group_id": f"g{index:05d}",
            "role": rr.ROLE_RESERVE,
            "aliases": [str(index)],
            "exposure": {f: False for f in rr.EXPOSURE_FLAGS
                          if f != "no_recorded_use_in_scanned_artifacts"} | {
                "no_recorded_use_in_scanned_artifacts": True},
            "exposure_evidence": [],
            "exclusion_reason": None,
            "eval_only": {"gt_answers": "True", "fake_cls": "original"},
            "_tag": tag,
            "_text": text,
        }
        row.update(overrides)
        if overrides.get("group_id"):
            row["group_id"] = overrides["group_id"]
        rows.append(row)
    return rows


def ids_of(rows):
    return {r["_tag"]: r["canonical_id"] for r in rows}


def reserve_ids(rows, dev_tag="a"):
    """Canonical ids of the rows assigned to the reserve by default_roles."""
    ident = ids_of(rows)
    return [c for tag, c in ident.items() if tag != dev_tag]


def default_roles(rows, dev_tag="a"):
    """Roles that are consistent with the manifests: one development member, the rest reserve."""
    ident = ids_of(rows)
    return {rr.ROLE_DEVELOPMENT_CORE: [ident[dev_tag]],
            rr.ROLE_RESERVE: [c for tag, c in ident.items() if tag != dev_tag]}


def make_fixture_repo(base: Path, rows, manifests=None, artifact_line: str = ""):
    manifests = manifests or {"dev": [], "holdout": [], "dev_core": [], "selection": []}
    (base / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "scripts" / "redesign_registry.py", base / "scripts" / "redesign_registry.py")
    src = base / "data" / "MMFakeBench_test" / "source"
    src.mkdir(parents=True, exist_ok=True)
    (src / "MMFakeBench_test.json").write_text(json.dumps([
        {"text": r["_text"], "image_path": r["image_path_rel"], "text_source": "x",
         "image_source": "x", "gt_answers": "True", "fake_cls": "original"} for r in rows]))
    cache = base / "_audit" / "preflight_02"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "image_hashes.jsonl").write_text("\n".join(json.dumps({
        "image_path": r["image_path_rel"], "byte_sha256": r["image_byte_sha256"],
        "pixel_sha256": r["image_pixel_sha256"]}) for r in rows))
    man = base / "manifests"
    man.mkdir(exist_ok=True)
    (man / "split-500-v1.json").write_text(json.dumps(
        {"dev": manifests.get("dev", []), "holdout": manifests.get("holdout", [])}))
    (man / "devcore-100-v1.json").write_text(json.dumps({"dev_core": manifests.get("dev_core", [])}))
    (man / "a1-select-300-v1.json").write_text(json.dumps({"samples": manifests.get("selection", [])}))
    (base / "results").mkdir(exist_ok=True)
    (base / "results" / "dev.jsonl").write_text(artifact_line)
    evidence = base / "evidence" / "fixture-run"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "1000.json").write_text("{}")
    (evidence / "manifest.json").write_text("{}")      # declared non-bundle: excluded by the rule
    return base


def build_registry_doc(repo: Path, rows, roles_sets, manifest_sets=None, artifacts=("results/dev.jsonl",)):
    lookup, unresolved, _checked = rr.build_alias_map(rows, repo / "manifests")
    group_members = {}
    for r in rows:
        group_members.setdefault(r["group_id"], set()).add(r["canonical_id"])
    exposure_artifacts = [
        {"path": rel, "sha256": rr.sha256_file(repo / rel), "kind": "results_jsonl"} for rel in artifacts
    ]
    exposure_directories = []
    for rel_dir in ("evidence/fixture-run",):
        stems = rr.directory_inventory(repo / rel_dir)
        exposure_directories.append({"path": rel_dir, "inventory_sha256": rr.inventory_hash(stems),
                                     "n_bundles": len(stems), "kind": "evidence_bundle_dir"})
    manifest_sets = manifest_sets or {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []}
    return {
        "registry_version": rr.MODULE_VERSION,
        "schema_version": rr.SCHEMA_VERSION,
        "grouping_policy_version": rr.GROUPING_POLICY_VERSION,
        "manifests_dir": "manifests",
        "warning": "ADMIN/EVALUATION-ONLY test fixture",
        "conventions": {},
        "fingerprints": {
            "dataset_index": rr.dataset_index_fingerprint(rows),
            "group_binding": rr.group_binding_hash(group_members),
            "sample_identity": rr.identity_hash(rows),
            "role_assignment": rr.role_assignment_hash(rows),
            "row_payload": rr.row_payload_hash(rows),
            "alias_mapping": rr.alias_mapping_hash(lookup),
            "alias_ambiguity": rr.alias_ambiguity_hash(lookup.ambiguous),
            "schema_version": rr.SCHEMA_VERSION,
            "grouping_policy_version": rr.GROUPING_POLICY_VERSION,
        },
        "dependencies": rr.collect_dependencies(repo),
        "counts": {"dataset_records": len(rows)},
        "alias_ambiguities": lookup.ambiguous,
        "alias_unresolved_rows": unresolved[:10],
        "blocking_issues": [],
        "n_blocking_issues": 0,
        "exposure_artifacts": exposure_artifacts,
        "exposure_directories": exposure_directories,
        "non_bundle_filenames": sorted(rr.NON_BUNDLE_FILENAMES),
        "roles": {k: sorted(set(v)) for k, v in roles_sets.items()},
        "manifest_sets": {k: sorted(set(v)) for k, v in manifest_sets.items()},
        "membership_hashes": {k: rr.membership_hash(v) for k, v in roles_sets.items()},
        "manifest_set_hashes": {k: rr.membership_hash(v) for k, v in manifest_sets.items()},
        "group_members": {g: sorted(m) for g, m in sorted(group_members.items())},
        "seed": None,
    }


DEV_CORE_ENTRY = {"dev": [{"sample_id": "0", "image_path": None}],
                  "dev_core": [{"sample_id": "0", "image_path": None}],
                  "holdout": [], "selection": []}


class FixtureBase(unittest.TestCase):
    def setUp(self) -> None:
        self._guard = NetworkGuard()
        self._guard.__enter__()
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        self._guard.__exit__(None, None, None)

    def write_fixture(self, rows, roles_sets, name="case", manifests=None, artifact_line="", dev_tag="a"):
        # Apply the declared roles to the rows, so role sets and rows agree by construction and
        # a fixture cannot accidentally declare an exposed manifest member as reserve.
        for role, cids in roles_sets.items():
            for r in rows:
                if r["canonical_id"] in cids:
                    r["role"] = role
        for r in rows:
            if r["role"] != rr.ROLE_RESERVE:
                # a record assigned to any non-reserve role is not "no recorded use"
                r["exposure"]["no_recorded_use_in_scanned_artifacts"] = False
        dev_row = next((r for r in rows if r["_tag"] == dev_tag), rows[0])
        manifests = manifests or {
            "dev": [{"sample_id": "0", "image_path": dev_row["image_path_rel"]}],
            "dev_core": [{"sample_id": "0", "image_path": dev_row["image_path_rel"]}],
            "holdout": [], "selection": []}
        repo = make_fixture_repo(self.base / name, rows, manifests, artifact_line)
        manifest_sets = {
            "dev": [dev_row["canonical_id"]],
            "dev_core": [dev_row["canonical_id"]],
            "holdout": [],
            "selection_pool": [],
        }
        doc = build_registry_doc(repo, rows, roles_sets, manifest_sets)
        samples = repo / "samples.jsonl"
        samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        reg = repo / "registry.json"
        reg.write_text(json.dumps(doc))
        return repo, reg, samples


def basic_rows():
    return make_rows([
        ("a", "/real/x/a.png", "b1", "p1", "claim a"),
        ("b", "/real/x/b.png", "b2", "p2", "claim b"),
        ("c", "/real/x/c.png", "b3", "p3", "claim c"),
        ("d", "/real/x/d.png", "b4", "p4", "claim d"),
    ])


class IntegrityBindingTest(FixtureBase):
    def test_valid_fixture_passes(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "valid")
        report = RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertEqual(report["samples"], 4)
        self.assertEqual(report["identity_sources"]["identities_verified_from_sources"], 4)

    def test_swapped_group_members_are_detected(self) -> None:
        rows = basic_rows()
        for r in rows:
            r["group_id"] = "gA" if r["_tag"] in ("a", "b") else "gB"
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "swap", manifests={
            "dev": [{"sample_id": "0", "image_path": rows[0]["image_path_rel"]}],
            "dev_core": [{"sample_id": "0", "image_path": rows[0]["image_path_rel"]}],
            "holdout": [], "selection": []})
        doc = json.loads(reg.read_text())
        # record the ORIGINAL grouping, then swap the members inside the rows
        doc["fingerprints"]["group_binding"] = rr.group_binding_hash({"gA": ["a", "b"], "gB": ["c", "d"]})
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("group-binding", str(ctx.exception))

    def test_group_binding_hash_separates_swaps_and_additions(self) -> None:
        self.assertNotEqual(rr.group_binding_hash({"g1": ["a", "b"], "g2": ["c", "d"]}),
                            rr.group_binding_hash({"g1": ["c", "d"], "g2": ["a", "b"]}))
        self.assertNotEqual(rr.group_binding_hash({"g1": ["a", "b"]}),
                            rr.group_binding_hash({"g1": ["a"], "g2": ["b"]}))

    def test_declared_group_members_table_swap_is_detected(self) -> None:
        """Swapping two member lists in the declared table alone must fail, rows untouched."""
        rows = basic_rows()
        for r in rows:
            r["group_id"] = "gA" if r["_tag"] in ("a", "b") else "gB"
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "declaredgroups")
        doc = json.loads(reg.read_text())
        before = doc["group_members"]
        # swap the two member lists; rows, fingerprint and hashes are left unchanged
        doc["group_members"] = {gid: sorted(before[other]) for gid, other in
                                zip(sorted(before), reversed(sorted(before)))}
        reg.write_text(json.dumps(doc))
        v = RegistryValidator(reg, samples, repo)
        with self.assertRaises(RegistryValidationError) as ctx:
            v.verify_integrity()
        self.assertIn("declared group_members", str(ctx.exception))
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_declared_group_members_are_required(self) -> None:
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "nogroups")
        doc = json.loads(reg.read_text())
        doc.pop("group_members")
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("group_members", str(ctx.exception))

    def test_recorded_dependency_fingerprint_change_is_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "depfp")
        doc = json.loads(reg.read_text())
        doc["dependencies"]["dataset_source_file"]["sha256"] = "0" * 64
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("does not match its recorded fingerprint", str(ctx.exception))

    def test_changed_dependency_contents_are_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "depcontent")
        target = repo / "manifests" / "split-500-v1.json"
        target.write_text(json.dumps({"dev": [], "holdout": []}))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("does not match its recorded fingerprint", str(ctx.exception))

    def test_altered_development_manifest_is_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "manifest")
        (repo / "manifests" / "devcore-100-v1.json").write_text(
            json.dumps({"dev_core": [{"sample_id": "1", "image_path": rows[1]["image_path_rel"]}]}))
        # contents changed but the recorded fingerprint is unchanged
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertTrue("does not match its recorded fingerprint" in str(ctx.exception)
                        or "manifest set" in str(ctx.exception))

    def test_manifest_membership_must_match_the_actual_manifest(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "membership")
        doc = json.loads(reg.read_text())
        doc["manifest_sets"]["dev"] = []                     # claim no dev membership
        doc["manifest_set_hashes"]["dev"] = rr.membership_hash([])
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError):
            RegistryValidator(reg, samples, repo).verify_integrity()

    def test_changed_exposure_artifact_is_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "artifact")
        (repo / "results" / "dev.jsonl").write_text('{"image_path": "/real/x/a.png"}\n')
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("exposure artifact", str(ctx.exception))

    def test_evidence_directory_inventory_change_is_detected(self) -> None:
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "inventory")
        v = RegistryValidator(reg, samples, repo)
        v.verify_integrity()                       # baseline is valid
        (repo / "evidence" / "fixture-run" / "9999.json").write_text("{}")
        with self.assertRaises(RegistryValidationError) as ctx:
            v.verify_integrity()
        self.assertIn("inventory changed", str(ctx.exception))

    def test_missing_evidence_directory_is_detected(self) -> None:
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "inventorymissing")
        shutil.rmtree(repo / "evidence" / "fixture-run")
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("evidence directory", str(ctx.exception))

    def test_non_bundle_files_do_not_change_the_inventory(self) -> None:
        """The declared non-bundle names are excluded, so adding one must not fail validation."""
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "nonbundle")
        v = RegistryValidator(reg, samples, repo)
        v.verify_integrity()
        (repo / "evidence" / "fixture-run" / "refetch_manifest.json").write_text("{}")
        report = v.verify_integrity()
        self.assertEqual(report["exposure_directories"]["directories_verified"], 1)

    def test_empty_evidence_directory_is_rejected(self) -> None:
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "inventoryempty")
        for p in (repo / "evidence" / "fixture-run").glob("*.json"):
            p.unlink()
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("now empty", str(ctx.exception))

    def test_declared_non_bundle_rule_must_match(self) -> None:
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "nonbundlerule")
        doc = json.loads(reg.read_text())
        doc["non_bundle_filenames"] = ["manifest.json"]
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("non_bundle_filenames", str(ctx.exception))

    def test_directory_inventory_change_after_approval_is_rejected(self) -> None:
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "inventoryafter")
        v = RegistryValidator(reg, samples, repo)
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        (repo / "evidence" / "fixture-run" / "4242.json").write_text("{}")
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_incomplete_dependency_set_is_refused(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "deps")
        doc = json.loads(reg.read_text())
        doc["dependencies"].pop("image_hash_cache")
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("complete dependency set", str(ctx.exception))

    def test_unsupported_schema_version_is_rejected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "schema")
        doc = json.loads(reg.read_text())
        doc["schema_version"] = "registry_schema_v1"
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("schema_version", str(ctx.exception))

    def test_row_identity_disagreeing_with_the_pinned_source_is_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "identity")
        tampered = [dict(r) for r in rows]
        tampered[1]["claim_sha256"] = "f" * 64
        samples.write_text("\n".join(json.dumps(r) for r in tampered) + "\n")
        with self.assertRaises(RegistryValidationError):
            RegistryValidator(reg, samples, repo).verify_integrity()

    def test_blocking_issues_refuse_admission(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "blocking")
        doc = json.loads(reg.read_text())
        doc["blocking_issues"] = [{"kind": "unresolved_manifest_reference", "sample_id": "999"}]
        doc["n_blocking_issues"] = 1
        reg.write_text(json.dumps(doc))
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("blocking issue", str(ctx.exception))

    def test_validate_request_rejects_invalid_artifacts_directly(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "direct")
        doc = json.loads(reg.read_text())
        doc["fingerprints"]["group_binding"] = "0" * 64
        doc["dependencies"]["grouping_code"]["sha256"] = "1" * 64
        reg.write_text(json.dumps(doc))
        v = RegistryValidator(reg, samples, repo)
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_DEVELOPMENT, [ids_of(rows)["a"]])

    def test_reserve_declared_over_a_manifest_member_is_rejected(self) -> None:
        """A registry that declares every row reserve while manifests list a member must fail."""
        rows = basic_rows()
        repo, reg, samples = self.write_fixture(rows, default_roles(rows), "reserveoverlap")
        all_ids = [r["canonical_id"] for r in rows]
        for r in rows:                       # declare everything reserve, including the dev member
            r["role"] = rr.ROLE_RESERVE
            r["exposure"]["no_recorded_use_in_scanned_artifacts"] = True
        doc = json.loads(reg.read_text())
        doc["roles"] = {rr.ROLE_RESERVE: sorted(all_ids)}
        doc["membership_hashes"] = {rr.ROLE_RESERVE: rr.membership_hash(all_ids)}
        doc["fingerprints"]["role_assignment"] = rr.role_assignment_hash(rows)
        doc["fingerprints"]["row_payload"] = rr.row_payload_hash(rows)
        reg.write_text(json.dumps(doc))
        samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        v = RegistryValidator(reg, samples, repo)
        with self.assertRaises(RegistryValidationError) as ctx:
            v.verify_integrity()
        self.assertIn("manifest set", str(ctx.exception))
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, sorted(all_ids))

    def test_reserve_containing_an_exposed_row_is_rejected(self) -> None:
        """A reserve row carrying an exposure flag must be rejected."""
        rows = basic_rows()
        roles = default_roles(rows)                 # 'a' is the development member, others reserve
        repo, reg, samples = self.write_fixture(rows, roles, "exposedreserve")
        exposed = next(r for r in rows if r["role"] == rr.ROLE_RESERVE)
        exposed["exposure"]["evaluated_previously"] = True
        exposed["exposure"]["no_recorded_use_in_scanned_artifacts"] = False
        doc = json.loads(reg.read_text())
        doc["fingerprints"]["role_assignment"] = rr.role_assignment_hash(rows)
        doc["fingerprints"]["row_payload"] = rr.row_payload_hash(rows)
        reg.write_text(json.dumps(doc))
        samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("not eligible for the reserve", str(ctx.exception))

    def test_exposure_flag_change_with_unchanged_role_is_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "exposureflag")
        tampered = [dict(r) for r in rows]
        tampered[2]["exposure"] = dict(tampered[2]["exposure"], evaluated_previously=True)
        samples.write_text("\n".join(json.dumps(r) for r in tampered) + "\n")
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("row-payload", str(ctx.exception))

    def test_exclusion_provenance_change_is_detected(self) -> None:
        rows = basic_rows()
        roles = default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, "provenance")
        tampered = [dict(r) for r in rows]
        tampered[0]["exposure_evidence"] = ["exclusion:path:g00000 -> possible_duplicate->g00001"]
        samples.write_text("\n".join(json.dumps(r) for r in tampered) + "\n")
        with self.assertRaises(RegistryValidationError) as ctx:
            RegistryValidator(reg, samples, repo).verify_integrity()
        self.assertIn("row-payload", str(ctx.exception))



class StateMutationTest(FixtureBase):
    """Section-2 acceptance: mutated state must never be approved."""

    def _validator(self, tag="mut", roles=None):
        rows = basic_rows()
        roles = roles or default_roles(rows)
        repo, reg, samples = self.write_fixture(rows, roles, tag)
        return repo, reg, samples, RegistryValidator(reg, samples, repo), rows

    def test_dependency_mutation_after_first_approval_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("depafter")
        report = v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        self.assertEqual(report["reserved_overlap"], len(reserve_ids(rows)))
        (repo / "manifests" / "split-500-v1.json").write_text(
            json.dumps({"dev": [], "holdout": []}))
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_second_approval_after_artifact_change_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("arta")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        (repo / "results" / "dev.jsonl").write_text('{"image_path": "/real/x/a.png"}\n')
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_disk_mutation_after_construction_is_detected(self) -> None:
        repo, reg, samples, v, rows = self._validator("disk")
        tampered = [json.loads(json.dumps(r)) for r in rows]
        # flip a reserved row to a development role on disk; the registry still says reserve
        reserve_row = next(r for r in tampered if r["role"] == rr.ROLE_RESERVE)
        reserve_row["role"] = rr.ROLE_DEVELOPMENT_CORE
        samples.write_text("\n".join(json.dumps(r) for r in tampered) + "\n")
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_row_group_id_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("rowgroup")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.rows[0]["group_id"] = "g-mutated"
        with self.assertRaises(RegistryValidationError) as ctx:
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        self.assertIn("in-memory", str(ctx.exception))

    def test_row_exposure_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("rowexp")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.rows[1]["exposure"]["evaluated_previously"] = True
        with self.assertRaises(RegistryValidationError) as ctx:
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        self.assertIn("in-memory", str(ctx.exception))

    def test_row_reference_status_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("rowref")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.rows[1]["reference_status"] = "unresolved_reference"
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_row_alias_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("rowalias")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.rows[0]["aliases"] = ["invented"]
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_row_role_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("rowrole")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        target = next(r for r in v.rows if r["role"] == rr.ROLE_RESERVE)
        target["role"] = rr.ROLE_DEVELOPMENT_CORE
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_in_memory_role_map_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("rolemap")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.roles[rr.ROLE_RESERVE] = set()
        with self.assertRaises(RegistryValidationError) as ctx:
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        self.assertIn("in-memory", str(ctx.exception))

    def test_in_memory_derived_group_map_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("groupmap")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.group_members["g00000"] = {"x"}
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))

    def test_in_memory_registry_document_mutation_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("regdoc")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        v.registry["dependencies"]["grouping_code"]["sha256"] = "9" * 64
        with self.assertRaises(RegistryValidationError) as ctx:
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        self.assertIn("in-memory", str(ctx.exception))


    def test_fresh_unresolved_manifest_after_approval_is_rejected(self) -> None:
        """A manifest added after a first approval that references an unbounded identity fails."""
        repo, reg, samples, v, rows = self._validator("freshmanifest")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))       # first approval
        (repo / "manifests" / "new_manifest.json").write_text(json.dumps(
            {"dev": [{"sample_id": "unbounded", "image_path": "/real/unknown/missing.png"}]}))
        with self.assertRaises(RegistryValidationError) as ctx:
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        self.assertIn("unresolved", str(ctx.exception))

    def test_unresolved_reference_in_a_declared_dependency_is_rejected(self) -> None:
        repo, reg, samples, v, rows = self._validator("unresdep")
        v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))
        (repo / "manifests" / "a1-select-300-v1.json").write_text(json.dumps(
            {"samples": [{"sample_id": "unbounded", "image_path": "/real/unknown/missing.png"}]}))
        with self.assertRaises(RegistryValidationError):
            v.validate_request(rv.PURPOSE_RESERVE, reserve_ids(rows))


class ExclusionPropagationTest(FixtureBase):
    def test_possible_duplicate_then_verified_group_companion(self) -> None:
        rows = basic_rows()[:3]                       # a (exposed), b, c
        ident = ids_of(rows)
        rows[2]["image_byte_sha256"] = rows[1]["image_byte_sha256"]   # c grouped with b
        repo = make_fixture_repo(self.base / "prop", rows, {
            "dev": [{"sample_id": "0", "image_path": rows[0]["image_path_rel"]}],
            "dev_core": [{"sample_id": "0", "image_path": rows[0]["image_path_rel"]}],
            "holdout": [], "selection": []},
            artifact_line=json.dumps({"image_path": rows[0]["image_path_rel"]}) + "\n")
        artifacts = [{"path": "results/dev.jsonl", "kind": "results_jsonl",
                      "resolved_by_field": {"resolved_by_path": [ident["a"]]}}]
        group_of, _ = rr.build_groups(rows)
        near = [{"a": ident["a"], "b": ident["b"], "hamming": 2}]
        flags, roles, _, _, excluded = rr.assign_roles(
            rows, artifacts, group_of, near,
            {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertEqual(roles[ident["b"]], rr.ROLE_QUARANTINE_POSSIBLE_DUPLICATE)
        self.assertEqual(roles[ident["c"]], rr.ROLE_QUARANTINE_POSSIBLE_DUPLICATE,
                         "C must not remain reserve when B is quarantined")
        self.assertNotEqual(roles[ident["b"]], rr.ROLE_RESERVE)
        self.assertNotEqual(roles[ident["c"]], rr.ROLE_RESERVE)
        self.assertTrue(flags[ident["c"]]["possible_or_unresolved_exposure"])
        self.assertFalse(flags[ident["c"]]["no_recorded_use_in_scanned_artifacts"])

    def test_longer_chain_propagates_fully(self) -> None:
        rows = make_rows([(t, f"/real/x/{t}.png", f"b{t}", f"p{t}", f"claim {t}")
                          for t in ("A", "B", "C", "D", "E")])
        ident = ids_of(rows)
        rows[4]["image_byte_sha256"] = rows[3]["image_byte_sha256"]     # E grouped with D
        artifacts = [{"path": "results/dev.jsonl", "kind": "results_jsonl",
                      "resolved_by_field": {"resolved_by_path": [ident["A"]]}}]
        group_of, _ = rr.build_groups(rows)
        near = [{"a": ident["A"], "b": ident["B"], "hamming": 2},
                {"a": ident["B"], "b": ident["C"], "hamming": 2},
                {"a": ident["C"], "b": ident["D"], "hamming": 2}]
        _, roles, _, _, _ = rr.assign_roles(rows, artifacts, group_of, near,
                                            {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        for tag in ("B", "C", "D", "E"):
            self.assertNotEqual(roles[ident[tag]], rr.ROLE_RESERVE, f"{tag} must not be reserved")

    def test_unresolved_member_excludes_its_verified_group(self) -> None:
        rows = basic_rows()[:2]
        ident = ids_of(rows)
        rows[0]["reference_status"] = "unresolved_reference"
        rows[1]["image_byte_sha256"] = rows[0]["image_byte_sha256"]
        group_of, _ = rr.build_groups(rows)
        _, roles, _, _, _ = rr.assign_roles(rows, [], group_of, [],
                                            {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertEqual(roles[ident["a"]], rr.ROLE_QUARANTINE)
        self.assertNotEqual(roles[ident["b"]], rr.ROLE_RESERVE)

    def test_ambiguous_alias_candidate_excludes_its_group(self) -> None:
        rows = basic_rows()[:2]
        ident = ids_of(rows)
        rows[1]["image_byte_sha256"] = rows[0]["image_byte_sha256"]
        group_of, _ = rr.build_groups(rows)
        _, roles, _, _, _ = rr.assign_roles(rows, [], group_of, [],
                                            {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []},
                                            ambiguous_ids=[ident["a"]])
        self.assertNotEqual(roles[ident["a"]], rr.ROLE_RESERVE)
        self.assertNotEqual(roles[ident["b"]], rr.ROLE_RESERVE)

    def test_unrelated_eligible_group_stays_reserve(self) -> None:
        rows = basic_rows()[:3]
        ident = ids_of(rows)
        artifacts = [{"path": "results/dev.jsonl", "kind": "results_jsonl",
                      "resolved_by_field": {"resolved_by_path": [ident["a"]]}}]
        group_of, _ = rr.build_groups(rows)
        near = [{"a": ident["a"], "b": ident["b"], "hamming": 2}]
        _, roles, _, _, _ = rr.assign_roles(rows, artifacts, group_of, near,
                                            {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertEqual(roles[ident["c"]], rr.ROLE_RESERVE, "unrelated group must stay eligible")

    def test_propagation_is_independent_of_input_order(self) -> None:
        rows = basic_rows()
        ident = ids_of(rows)
        rows[3]["image_byte_sha256"] = rows[2]["image_byte_sha256"]
        artifacts = [{"path": "results/dev.jsonl", "kind": "results_jsonl",
                      "resolved_by_field": {"resolved_by_path": [ident["a"]]}}]
        near = [{"a": ident["a"], "b": ident["b"], "hamming": 2},
                {"a": ident["b"], "b": ident["c"], "hamming": 2}]
        g1, _ = rr.build_groups(rows)
        _, r1, _, _, _ = rr.assign_roles(rows, artifacts, g1, near,
                                         {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        rev = list(reversed(rows))
        g2, _ = rr.build_groups(rev)
        _, r2, _, _, _ = rr.assign_roles(rev, artifacts, g2, list(reversed(near)),
                                         {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertEqual(r1, r2)


class AliasAmbiguityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.rows = make_rows([
            ("p", "/real/x/p.png", "bp", "pp", "claim p", {"dataset_index": 7}),
            ("q", "/real/x/q.png", "bq", "pq", "claim q", {"dataset_index": 9}),
        ])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _manifests(self, order):
        man = self.base / "manifests"
        man.mkdir(exist_ok=True)
        for name in ("a.json", "b.json"):
            p = man / name
            if p.exists():
                p.unlink()
        first, second = order
        text = {"p": "claim p", "q": "claim q"}
        (man / f"{first}.json").write_text(json.dumps(
            {"dev": [{"sample_id": "ds7", "image_path": "/real/x/p.png"}]}))
        (man / f"{second}.json").write_text(json.dumps(
            {"dev": [{"sample_id": "ds7", "image_path": "/real/x/q.png"}]}))
        return man

    def _context(self):
        return ({r["image_path_rel"]: r for r in self.rows},
                {r["dataset_index"]: r for r in self.rows})

    def test_conflicting_aliases_are_recorded_not_resolved(self) -> None:
        lookup, _, _ = rr.build_alias_map(self.rows, self._manifests(("a", "b")))
        self.assertIn("ds7", lookup.ambiguous, "conflict must be recorded as ambiguous")
        self.assertNotIn("ds7", lookup.mapping, "no identity may be chosen")
        self.assertEqual(len(lookup.ambiguous["ds7"]["candidates"]), 2)
        self.assertTrue(lookup.ambiguous["ds7"]["provenance"])

    def test_traversal_order_does_not_change_the_outcome(self) -> None:
        """Same manifest content, opposite traversal order: identical resolution and ambiguity."""
        import unittest.mock as mock
        man = self._manifests(("a", "b"))
        forward, _, _ = rr.build_alias_map(self.rows, man)
        real_glob = Path.glob

        def reversed_glob(self_path, pattern):
            return iter(sorted(real_glob(self_path, pattern), reverse=True))

        with mock.patch.object(Path, "glob", reversed_glob):
            backward, _, _ = rr.build_alias_map(self.rows, man)
        self.assertEqual(sorted(forward.mapping.items()), sorted(backward.mapping.items()))
        self.assertEqual(sorted(forward.ambiguous), sorted(backward.ambiguous))
        self.assertEqual(rr.alias_mapping_hash(forward), rr.alias_mapping_hash(backward))
        self.assertEqual(rr.alias_ambiguity_hash(forward.ambiguous),
                         rr.alias_ambiguity_hash(backward.ambiguous))

    def test_ambiguity_provenance_is_bound_by_the_hash(self) -> None:
        a, _, _ = rr.build_alias_map(self.rows, self._manifests(("a", "b")))
        stripped = json.loads(json.dumps(a.ambiguous))
        for info in stripped.values():
            info["provenance"] = {}
        self.assertNotEqual(rr.alias_ambiguity_hash(a.ambiguous), rr.alias_ambiguity_hash(stripped))

    def test_ambiguous_alias_fails_explicitly_including_numeric_fallback(self) -> None:
        lookup, _, _ = rr.build_alias_map(self.rows, self._manifests(("a", "b")))
        by_rel, by_index = self._context()
        for value in ("ds7", "7"):
            cid, status = rr.resolve_reference(value, "sample_id", by_rel, by_index, lookup)
            self.assertIsNone(cid, f"{value} must not resolve")
            self.assertEqual(status, "ambiguous_alias", f"{value} -> {status}")

    def test_plain_dict_lookup_is_refused(self) -> None:
        by_rel, by_index = self._context()
        with self.assertRaises(TypeError):
            rr.resolve_reference("7", "sample_id", by_rel, by_index, {"7": "mfb-p"})

    def test_repeated_alias_to_same_identity_remains_usable(self) -> None:
        man = self.base / "manifests2"
        man.mkdir(exist_ok=True)
        (man / "a.json").write_text(json.dumps(
            {"dev": [{"sample_id": "7", "image_path": "/real/x/p.png"}]}))
        (man / "b.json").write_text(json.dumps(
            {"dev_core": [{"sample_id": "7", "image_path": "/real/x/p.png"}]}))
        lookup, _, _ = rr.build_alias_map(self.rows, man)
        self.assertNotIn("7", lookup.ambiguous)
        by_rel, by_index = self._context()
        self.assertEqual(rr.resolve_reference("7", "sample_id", by_rel, by_index, lookup)[1],
                         "resolved_by_alias")

    def test_only_exact_supported_spellings_resolve(self) -> None:
        """A malformed identifier such as dsINVALID7 must never resolve to row 7."""
        lookup, _, _ = rr.build_alias_map(self.rows, self._manifests(("a", "b")))
        by_rel, by_index = self._context()
        for value in ("dsINVALID7", "ds7x", "7ds", "d s7", "dS", "7.0", "0x7", "-7"):
            cid, status = rr.resolve_reference(value, "sample_id", by_rel, by_index, lookup)
            self.assertIsNone(cid, f"{value!r} must not resolve, got {cid}")
            self.assertEqual(status, "unresolved_reference", f"{value!r} -> {status}")
        # surrounding whitespace is trimmed and the trimmed form is then judged by the same
        # rules, so a contested identity stays contested and is never silently resolved
        cid, status = rr.resolve_reference(" 7 ", "sample_id", by_rel, by_index, lookup)
        self.assertIsNone(cid)
        self.assertEqual(status, "ambiguous_alias")

    def test_exact_spellings_still_resolve(self) -> None:
        lookup = rr.AliasLookup()
        by_rel, by_index = self._context()
        for value in ("7", "ds7", "DS7"):
            cid, status = rr.resolve_reference(value, "sample_id", by_rel, by_index, lookup)
            self.assertEqual(cid, self.rows[0]["canonical_id"], f"{value!r} should resolve")
            self.assertEqual(status, "resolved_by_index_digits")

    def test_unknown_alias_does_not_become_a_new_sample(self) -> None:
        lookup, _, _ = rr.build_alias_map(self.rows, self._manifests(("a", "b")))
        by_rel, by_index = self._context()
        cid, status = rr.resolve_reference("ds4242", "sample_id", by_rel, by_index, lookup)
        self.assertIsNone(cid)
        self.assertEqual(status, "unresolved_reference")

    def test_ambiguous_candidates_are_never_reserved(self) -> None:
        lookup, _, _ = rr.build_alias_map(self.rows, self._manifests(("a", "b")))
        ambiguous_ids = sorted({c for info in lookup.ambiguous.values() for c in info["candidates"]})
        group_of, _ = rr.build_groups(self.rows)
        _, roles, _, _, _ = rr.assign_roles(self.rows, [], group_of, [],
                                            {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []},
                                            ambiguous_ids)
        for cid in ambiguous_ids:
            self.assertNotEqual(roles[cid], rr.ROLE_RESERVE)



class ModelInputBoundaryTest(unittest.TestCase):
    """F4: typed signal projection at the mocked SDK boundary."""

    def setUp(self) -> None:
        self._guard = NetworkGuard()
        self._guard.__enter__()
        import openai
        import scripts.ai_judge as ai_judge
        self.ai_judge = ai_judge
        self._openai = openai.OpenAI

        class Recorder:
            def __init__(self):
                self.calls = []

            def __call__(self, *a, **k):
                return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._create)))

            def _create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
                        {"label": "Not Misinformation", "confidence": 0.5, "rationale": "x",
                         "key_factors": []})), logprobs=None)],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        self.recorder = Recorder()
        openai.OpenAI = self.recorder

    def tearDown(self) -> None:
        import openai
        openai.OpenAI = self._openai
        self._guard.__exit__(None, None, None)

    def _text(self):
        def leaves(obj):
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                for v in obj.values():
                    yield from leaves(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    yield from leaves(v)
        return "\n".join(leaves(self.recorder.calls[-1]))

    def _judge(self, final_obj):
        from scripts.llm_loader import LLMModelLoader
        loader = LLMModelLoader({"provider": "openai", "model": "gpt-4o", "api_key": "dummy",
                                 "temperature": 0.0})
        self.ai_judge.judge_from_structured(final_obj, loader)
        return self._text()

    def test_nested_private_metadata_does_not_reach_the_boundary(self) -> None:
        text = self._judge({
            "headline": "A neutral claim",
            "relevancy": {"aligned": True, "confidence": 0.9, "explanation": "ok",
                          "benchmark_note": "SECRET_NESTED_MARKER", "nested": {"gt": "Fake"}},
            "visual_veracity": {"ai_generated": False, "confidence": 0.1, "explanation": "clean",
                                "debug": {"image_source": "AI-generated Image"},
                                "anomalies": ["ok", {"bad": "SECRET_IN_LIST"}]},
            "best_qa_per_chain": [],
        })
        self.assertNotIn("SECRET_NESTED_MARKER", text)
        self.assertNotIn("AI-generated Image", text)
        self.assertNotIn("SECRET_IN_LIST", text)

    def test_malformed_field_types_are_rejected_not_stringified(self) -> None:
        text = self._judge({
            "headline": "A neutral claim",
            "relevancy": {"aligned": {"bad": 1}, "confidence": "high",
                          "explanation": {"debug": "SECRET_DICT_EXPLANATION"}},
            "visual_veracity": {"ai_generated": "maybe", "confidence": None,
                                "explanation": ["SECRET_LIST_EXPLANATION"]},
            "best_qa_per_chain": [],
        })
        self.assertNotIn("SECRET_DICT_EXPLANATION", text)
        self.assertNotIn("SECRET_LIST_EXPLANATION", text)
        self.assertIn('"explanation": null', text)
        self.assertIn('"aligned": null', text)
        self.assertIn('"ai_generated": null', text)

    def test_valid_signals_are_preserved_including_long_values(self) -> None:
        long_explanation = "detail " * 2000
        many_anomalies = [f"anomaly-{i}-" + "y" * 300 for i in range(50)]
        text = self._judge({
            "headline": "A neutral claim",
            "relevancy": {"aligned": "partial", "confidence": 0.42, "explanation": long_explanation},
            "visual_veracity": {"ai_generated": True, "confidence": 0.97,
                                "explanation": "clean", "anomalies": many_anomalies},
            "best_qa_per_chain": [],
        })
        self.assertIn(long_explanation, text, "long explanation must not be truncated")
        for item in many_anomalies:
            self.assertIn(item, text, "anomaly list must not be truncated")
        self.assertIn('"aligned": "partial"', text)
        self.assertIn('"ai_generated": true', text)
        self.assertIn('"confidence": 0.42', text)

    def test_computed_ai_generated_signal_survives(self) -> None:
        text = self._judge({
            "headline": "Claim check: the photo is real, the caption is fake",
            "relevancy": {"aligned": False, "confidence": 0.8, "explanation": "mismatch"},
            "visual_veracity": {"ai_generated": True, "confidence": 0.91, "explanation": "artifacts"},
            "best_qa_per_chain": [],
        })
        self.assertIn("the photo is real, the caption is fake", text)
        self.assertIn('"ai_generated": true', text)

    def test_malformed_qa_entries_do_not_reach_the_boundary(self) -> None:
        payload = self.ai_judge._model_facing_payload({
            "headline": "h",
            "relevancy": {}, "visual_veracity": {},
            "best_qa_per_chain": [{
                "question": {"private": "SYNTHETIC_QA_MARKER"},
                "answer": {"private": "SYNTHETIC_QA_MARKER"},
                "confidence": {"private": "SYNTHETIC_CONF_MARKER"},
                "citations": [1, 2, 3],
            }],
        })
        entry = payload["best_qa_per_chain"][0]
        self.assertIsNone(entry["question"])
        self.assertIsNone(entry["answer"])
        self.assertIsNone(entry["confidence"])
        self.assertEqual(entry["citations_count"], 3, "valid citation count must be preserved")
        self.assertNotIn("SYNTHETIC", json.dumps(payload))

    def test_qa_projection_at_both_boundaries(self) -> None:
        text = self._judge({
            "headline": "A neutral claim",
            "relevancy": {"aligned": True, "confidence": 0.5, "explanation": "e"},
            "visual_veracity": {"ai_generated": False, "confidence": 0.1, "explanation": "c"},
            "best_qa_per_chain": [{"question": {"private": "SYNTHETIC_QA_MARKER"},
                                   "answer": {"private": "SYNTHETIC_QA_MARKER"},
                                   "confidence": {"private": "SYNTHETIC_CONF_MARKER"},
                                   "citations": [{"url": "https://example.org/x"}]}],
        })
        self.assertNotIn("SYNTHETIC", text)
        self.assertIn('"citations_count": 1', text)

        from scripts.judge_study.judges import judge_j1
        judge_j1({"image_path": "/data/MMFakeBench_test/real/x.png", "claim": "A neutral claim",
                  "relevancy": {"aligned": True, "confidence": 0.5, "explanation": "e"},
                  "visual_veracity": {"ai_generated": False, "confidence": 0.1, "explanation": "c"},
                  "answers": [{"question": {"private": "SYNTHETIC_QA_MARKER"},
                               "answer": {"private": "SYNTHETIC_QA_MARKER"},
                               "confidence": {"private": "SYNTHETIC_CONF_MARKER"},
                               "citations": [{"url": "https://example.org/x"}]}],
                  "documents": {}}, self._loader())
        text2 = self._text()
        self.assertNotIn("SYNTHETIC", text2)
        self.assertIn('"citations_count": 1', text2)

    def test_malformed_whole_signal_objects_keep_a_consistent_shape(self) -> None:
        payload = self.ai_judge._model_facing_payload({
            "headline": "h", "relevancy": None, "visual_veracity": "not-a-dict",
            "best_qa_per_chain": [],
        })
        self.assertEqual(tuple(payload["relevancy"]), self.ai_judge._RELEVANCY_FIELDS)
        self.assertEqual(tuple(payload["visual_veracity"]), self.ai_judge._VISUAL_FIELDS)
        self.assertEqual(payload["relevancy"]["aligned"], None)
        self.assertEqual(payload["visual_veracity"]["anomalies"], [])

    def test_projection_keys_match_the_declared_signal_schema(self) -> None:
        payload = self.ai_judge._model_facing_payload({
            "headline": "h", "relevancy": {"extra": 1}, "visual_veracity": {"extra": 1},
            "best_qa_per_chain": [],
        })
        self.assertEqual(tuple(payload["relevancy"]), self.ai_judge._RELEVANCY_FIELDS)
        self.assertEqual(tuple(payload["visual_veracity"]), self.ai_judge._VISUAL_FIELDS)

    def test_j1_entry_point_is_also_projected(self) -> None:
        from scripts.judge_study.judges import judge_j1
        bundle = {
            "image_path": "/data/MMFakeBench_test/fake/fam/x.png",
            "claim": "A neutral claim",
            "relevancy": {"aligned": True, "confidence": 0.9, "explanation": "ok",
                          "benchmark_note": "SECRET_NESTED_MARKER"},
            "visual_veracity": {"ai_generated": False, "confidence": 0.1, "explanation": "clean",
                                "debug": "SECRET_DEBUG"},
            "answers": [], "documents": {},
        }
        judge_j1(bundle, self._loader())
        text = self._text().lower()
        for token in ("secret_nested_marker", "secret_debug", "/fake/", "mmfakebench"):
            self.assertNotIn(token, text)

    def _loader(self):
        from scripts.llm_loader import LLMModelLoader
        return LLMModelLoader({"provider": "openai", "model": "gpt-4o", "api_key": "dummy",
                               "temperature": 0.0})


class OverwriteGuardTest(unittest.TestCase):
    def test_builder_refuses_to_overwrite_v1_output(self) -> None:
        out = subprocess.run(
            [sys.executable, "-m", "scripts.redesign_registry", "--repo", str(REPO), "--version", "v1"],
            cwd=str(REPO), capture_output=True, text=True)
        self.assertEqual(out.returncode, 3)
        self.assertIn("REFUSING to overwrite", out.stderr)
        self.assertIn("redesign_sample_registry_v1.jsonl", out.stderr)


class HistoricalPreservationTest(unittest.TestCase):
    def test_v1_registry_artifacts_and_development_core_are_unchanged(self) -> None:
        state = json.loads((REPO / "_audit" / "preflight_02_1" / "starting_state.json").read_text())
        protected = state["protected_artifact_sha256"]
        for rel in ("manifests/redesign_sample_registry_v1.jsonl",
                    "manifests/redesign_data_registry_v1.json",
                    "manifests/split-500-v1.json",
                    "manifests/devcore-100-v1.json",
                    "manifests/a1-select-300-v1.json"):
            if rel not in protected:
                continue
            self.assertEqual(rr.sha256_file(REPO / rel), protected[rel],
                             f"{rel} must be unchanged")

    def test_development_core_selection_is_preserved(self) -> None:
        core = json.loads((REPO / "manifests" / "devcore-100-v1.json").read_text())
        self.assertEqual(len(core["dev_core"]), 100)


class PreFixContrastTest(unittest.TestCase):
    """Demonstrates the reported defects against isolated pre-fix snapshots."""

    def setUp(self) -> None:
        self.pkg = REPO / "_audit" / "preflight_02_1"
        if not (self.pkg / "prefix_pkg" / "validate.py").exists():
            self.skipTest("pre-fix snapshot package not present")
        if str(self.pkg) not in sys.path:
            sys.path.insert(0, str(self.pkg))

    def test_prefix_has_no_member_binding_hash(self) -> None:
        import importlib
        pre = importlib.import_module("prefix_pkg.registry")
        self.assertTrue(hasattr(pre, "group_membership_hash"))
        self.assertFalse(hasattr(pre, "group_binding_hash"))
        self.assertEqual(pre.group_membership_hash({"g1": ["a"], "g2": ["b"]}.keys()),
                         pre.group_membership_hash({"g1": ["b"], "g2": ["a"]}.keys()))

    def test_prefix_resolves_an_ambiguous_alias_silently(self) -> None:
        import importlib
        pre = importlib.import_module("prefix_pkg.registry")
        rows = [{"image_path_rel": "/real/x/p.png", "dataset_index": 7, "canonical_id": "mfb-p"},
                {"image_path_rel": "/real/x/q.png", "dataset_index": 9, "canonical_id": "mfb-q"}]
        by_rel = {r["image_path_rel"]: r for r in rows}
        by_index = {r["dataset_index"]: r for r in rows}
        cid, status = pre.resolve_reference("7", "sample_id", by_rel, by_index, {"ds7": "mfb-p"})
        self.assertEqual(status, "resolved_by_index_digits",
                         "the pre-fix lookup resolves digits with no ambiguity state")
