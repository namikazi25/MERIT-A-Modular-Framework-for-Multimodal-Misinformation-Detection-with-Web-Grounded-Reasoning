"""Focused offline tests for the Task 0.2 redesign registry and validator.

All fixtures are synthetic. No network, no dotenv loading, no model calls.
Run with:  python -m unittest tests.test_redesign_registry -v
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image

# Guard: importing anything here must not read a real .env.
if "dotenv" not in sys.modules:
    _stub = types.ModuleType("dotenv")
    _stub.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = _stub

REPO_ROOT = Path(__file__).resolve().parents[1]

import scripts.redesign_registry as rr
from scripts.redesign_registry_validate import (
    RegistryValidationError,
    RegistryValidator,
    PURPOSE_DEVELOPMENT,
    PURPOSE_RESERVE,
)


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


def _make_row(cid, index, rel, byte_sha, pixel_sha, claim_sha, dhash="0" * 16, status="resolved",
              fake_cls="original", gt="True"):
    return {
        "canonical_id": cid,
        "dataset_index": index,
        "image_path_rel": rel,
        "image_byte_sha256": byte_sha,
        "image_pixel_sha256": pixel_sha,
        "dhash64": dhash,
        "claim_sha256": claim_sha,
        "reference_status": status,
        "eval_only": {"gt_answers": gt, "fake_cls": fake_cls},
    }


class ClaimNormalisationTest(unittest.TestCase):
    def test_preserves_negation_numbers_dates_and_entities(self) -> None:
        a = rr.normalize_claim("Trump did NOT win in 2020; 3,000 votes were recounted.")
        b = rr.normalize_claim("Trump DID win in 2020; 3,000 votes were recounted.")
        self.assertNotEqual(rr.claim_sha256(a), rr.claim_sha256(b), "negation must be preserved")

        c = rr.normalize_claim("The fire started on 12 March 2019 in Manchester.")
        d = rr.normalize_claim("The fire started on 12 March 2020 in Manchester.")
        self.assertNotEqual(rr.claim_sha256(c), rr.claim_sha256(d), "dates must be preserved")

        e = rr.normalize_claim("About 3,000 people attended.")
        f = rr.normalize_claim("About 30,000 people attended.")
        self.assertNotEqual(rr.claim_sha256(e), rr.claim_sha256(f), "numbers must be preserved")

    def test_absorbs_only_presentational_differences(self) -> None:
        a = rr.normalize_claim("  The council   approved the budget.  ")
        b = rr.normalize_claim("the council approved the budget.")
        c = rr.normalize_claim("The council\u00a0approved the budget.")
        d = rr.normalize_claim("The council \u2014 approved \u2018the budget\u2019.".replace("\u2014", "-"))
        self.assertEqual(rr.claim_sha256(a), rr.claim_sha256(b))
        self.assertEqual(rr.claim_sha256(a), rr.claim_sha256(c))
        self.assertNotEqual(rr.claim_sha256(a), rr.claim_sha256(d), "quotes are content, not whitespace")


class CanonicalIdTest(unittest.TestCase):
    def test_distinct_records_with_a_shared_image_stay_distinct(self) -> None:
        a = rr.canonical_sample_id("/fake/x.png", "claim one")
        b = rr.canonical_sample_id("/fake/x.png", "claim two")
        self.assertNotEqual(a, b)

    def test_id_is_deterministic_and_opaque(self) -> None:
        a = rr.canonical_sample_id("/fake/chatgpt_match_test_500/x.png", "c")
        self.assertEqual(a, rr.canonical_sample_id("/fake/chatgpt_match_test_500/x.png", "c"))
        self.assertTrue(a.startswith("mfb-"))
        for token in ("fake", "real", "chatgpt", "MMFakeBench"):
            self.assertNotIn(token, a.lower())


class ReferenceResolutionTest(unittest.TestCase):
    """Fixture updated for the v2 lookup type. Every assertion is preserved.

    ``resolve_reference`` now requires an ``AliasLookup`` so the ambiguity state always travels
    with the mapping; a bare dict is refused and raises ``TypeError``. The fixture therefore
    constructs the typed lookup instead of a plain dict.
    """

    def setUp(self) -> None:
        self.rows = [
            _make_row("mfb-a", 5, "/fake/fam/x.png", "b1", "p1", "c1"),
            _make_row("mfb-b", 9, "/real/fam/y.png", "b2", "p2", "c2"),
        ]
        self.by_rel = {r["image_path_rel"]: r for r in self.rows}
        self.by_index = {r["dataset_index"]: r for r in self.rows}
        self.alias_map = rr.AliasLookup(mapping={"ds5": "mfb-a", "9": "mfb-b"},
                                        provenance={"ds5": ["fixture"], "9": ["fixture"]})

    def test_resolves_by_recorded_path_first(self) -> None:
        cid, status = rr.resolve_reference("data/MMFakeBench_test/fake/fam/x.png", "image_path",
                                           self.by_rel, self.by_index, self.alias_map)
        self.assertEqual(cid, "mfb-a")
        self.assertEqual(status, "resolved_by_path")

    def test_resolves_historical_alias_namespaces(self) -> None:
        self.assertEqual(rr.resolve_reference("ds5", "sample_id", self.by_rel, self.by_index, self.alias_map)[0], "mfb-a")
        self.assertEqual(rr.resolve_reference("9", "sample_id", self.by_rel, self.by_index, self.alias_map)[0], "mfb-b")

    def test_digit_form_resolves_to_the_dataset_row_index(self) -> None:
        cid, status = rr.resolve_reference("5", "sample_id", self.by_rel, self.by_index, self.alias_map)
        self.assertEqual(cid, "mfb-a")
        self.assertEqual(status, "resolved_by_index_digits")

    def test_other_dataset_root_is_out_of_scope_not_resolved_by_digits(self) -> None:
        """MMFakeBench_val row 5 must NOT resolve to MMFakeBench_test row 5."""
        cid, status = rr.resolve_reference("data/MMFakeBench_val/fake/fam/x.png", "image_path",
                                           self.by_rel, self.by_index, self.alias_map)
        self.assertIsNone(cid)
        self.assertTrue(status.startswith("out_of_scope_dataset"))

    def test_unmapped_reference_is_never_silently_resolved(self) -> None:
        cid, status = rr.resolve_reference("ds9999", "sample_id", self.by_rel, self.by_index, self.alias_map)
        self.assertIsNone(cid)
        self.assertEqual(status, "unresolved_reference")
        cid2, status2 = rr.resolve_reference("data/MMFakeBench_test/fake/fam/missing.png", "image_path",
                                             self.by_rel, self.by_index, self.alias_map)
        self.assertIsNone(cid2)
        self.assertEqual(status2, "path_not_in_dataset")

    def test_positional_counters_are_not_treated_as_ids(self) -> None:
        """sample_index / iteration_index are counters; 5 happens to exist but 'sample_index=' must not map."""
        cid, status = rr.resolve_reference("sample_index=5", "sample_id", self.by_rel, self.by_index, self.alias_map)
        self.assertIsNone(cid)


class GroupingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.img = base / "a.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.img)
        self.copy = base / "a_copy.png"          # byte-identical
        self.copy.write_bytes(self.img.read_bytes())
        self.reencoded = base / "a_reencoded.png"  # same pixels, different bytes
        Image.open(self.img).save(self.reencoded, format="PNG", optimize=True, compress_level=9)

    def test_byte_and_decoded_identical_grouping(self) -> None:
        h1 = rr.image_hashes(self.img)
        h2 = rr.image_hashes(self.copy)
        h3 = rr.image_hashes(self.reencoded)
        self.assertEqual(h1["byte_sha256"], h2["byte_sha256"])
        self.assertEqual(h1["pixel_sha256"], h3["pixel_sha256"])
        self.assertEqual(h1["pixel_sha256"], h2["pixel_sha256"])

    def test_summary_never_groups_on_class_or_family(self) -> None:
        rows = [
            _make_row("mfb-1", 1, "/fake/chatgpt_match_test_500/a.png", "b1", "p1", "c1"),
            _make_row("mfb-2", 2, "/fake/chatgpt_match_test_500/b.png", "b2", "p2", "c2"),
            _make_row("mfb-3", 3, "/fake/gossipcop_midjourney_test_250/c.png", "b3", "p3", "c3"),
        ]
        group_of, edges = rr.build_groups(rows)
        self.assertEqual(len(edges), 0, "shared family or class must not create an edge")
        self.assertEqual(len(set(group_of.values())), 3)

    def test_transitive_closure(self) -> None:
        """A-B share bytes, B-C share a claim, so A-B-C must be one group."""
        rows = [
            _make_row("mfb-a", 1, "/fake/x/a.png", "same-bytes", "p-a", "c-a"),
            _make_row("mfb-b", 2, "/fake/x/b.png", "same-bytes", "p-b", "c-shared"),
            _make_row("mfb-c", 3, "/fake/x/c.png", "b-c", "p-c", "c-shared"),
        ]
        group_of, edges = rr.build_groups(rows)
        self.assertEqual(len({group_of[r["canonical_id"]] for r in rows}), 1)
        self.assertEqual({e["kind"] for e in edges}, {rr.EDGE_BYTE_IDENTICAL, rr.EDGE_CLAIM_IDENTICAL})

    def test_grouping_is_deterministic_under_reordered_input(self) -> None:
        rows = [
            _make_row(f"mfb-{n}", n, f"/fake/x/{n}.png", "same" if n in (2, 5) else f"b{n}",
                      f"p{n}", "cs" if n in (5, 7) else f"c{n}")
            for n in range(1, 9)
        ]
        g1, _ = rr.build_groups(rows)
        g2, _ = rr.build_groups(list(reversed(rows)))
        self.assertEqual(g1, g2, "group ids must not depend on input order")

    def test_membership_hash_ignores_order_and_duplicates(self) -> None:
        self.assertEqual(rr.membership_hash(["a", "b", "c"]), rr.membership_hash(["c", "a", "b", "a"]))
        self.assertNotEqual(rr.membership_hash(["a", "b"]), rr.membership_hash(["a", "b", "c"]))

    def test_order_hash_is_order_sensitive(self) -> None:
        self.assertNotEqual(rr.order_hash(["a", "b"]), rr.order_hash(["b", "a"]))


class ExposureAndRoleTest(unittest.TestCase):
    def _rows(self):
        return [
            _make_row("mfb-dev", 1, "/fake/x/dev.png", "b-dev", "p-dev", "c-dev"),
            _make_row("mfb-dup", 2, "/fake/x/dup.png", "b-dev", "p-dev", "c-dup"),   # shares bytes with dev
            _make_row("mfb-free", 3, "/real/x/free.png", "b-free", "p-free", "c-free"),
            _make_row("mfb-bad", 4, "/real/x/bad.png", "b-bad", "p-bad", "c-bad", status="unresolved_reference"),
        ]

    def test_exposure_propagates_across_verified_groups(self) -> None:
        rows = self._rows()
        group_of, _ = rr.build_groups(rows)
        artifacts = [{"path": "results/dev.jsonl", "kind": "results_jsonl",
                      "resolved_by_field": {"resolved_by_path": ["mfb-dev"]}}]
        flags, roles, _, propagated, _ = rr.assign_roles(rows, artifacts, group_of, [], {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertIn("mfb-dup", propagated, "a group mate of an evaluated sample must inherit exposure")
        self.assertTrue(flags["mfb-dup"]["used_for_development_or_selection"])
        self.assertEqual(roles["mfb-dup"], rr.ROLE_DEVELOPMENT_HISTORICAL)
        self.assertEqual(roles["mfb-free"], rr.ROLE_RESERVE)

    def test_unresolved_reference_is_quarantined_not_reserved(self) -> None:
        rows = self._rows()
        group_of, _ = rr.build_groups(rows)
        flags, roles, _, _, _ = rr.assign_roles(rows, [], group_of, [], {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertEqual(roles["mfb-bad"], rr.ROLE_QUARANTINE)
        self.assertTrue(flags["mfb-bad"]["possible_or_unresolved_exposure"])
        self.assertFalse(flags["mfb-bad"]["no_recorded_use_in_scanned_artifacts"])

    def test_unresolved_near_duplicate_candidate_is_quarantined(self) -> None:
        rows = self._rows()
        group_of, _ = rr.build_groups(rows)
        artifacts = [{"path": "results/dev.jsonl", "kind": "results_jsonl",
                      "resolved_by_field": {"resolved_by_path": ["mfb-dev"]}}]
        near_dups = [{"a": "mfb-dev", "b": "mfb-free", "hamming": 2}]
        flags, roles, _, _, conflicts = rr.assign_roles(rows, artifacts, group_of, near_dups,
                                                        {"dev": [], "holdout": [], "dev_core": [], "selection_pool": []})
        self.assertIn("mfb-free", conflicts)
        # F2 gives this path its own role: a possible-duplicate exclusion, not the generic one
        self.assertEqual(roles["mfb-free"], rr.ROLE_QUARANTINE_POSSIBLE_DUPLICATE)
        self.assertNotEqual(roles["mfb-free"], rr.ROLE_RESERVE)


def _v2_rows():
    """Rows whose canonical ids and claim hashes are derived, so source verification can pass."""
    specs = [("dev", "/fake/x/dev.png", "b1", "p1", "claim dev"),
             ("res", "/real/x/res.png", "b2", "p2", "claim res"),
             ("bad", "/real/x/bad.png", "b3", "p3", "claim bad")]
    rows = []
    for index, (tag, rel, b, p, text) in enumerate(specs):
        rows.append({
            "canonical_id": rr.canonical_sample_id(rel, text),
            "dataset_index": index,
            "image_path_rel": rel,
            "image_byte_sha256": b,
            "image_pixel_sha256": p,
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
        })
    return rows


def _v2_repo(base: Path, rows, manifests=None):
    manifests = manifests or {"dev": [{"sample_id": "0", "image_path": rows[0]["image_path_rel"]}],
                              "dev_core": [{"sample_id": "0", "image_path": rows[0]["image_path_rel"]}],
                              "holdout": [], "selection": []}
    (base / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "scripts" / "redesign_registry.py", base / "scripts" / "redesign_registry.py")
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
    (man / "split-500-v1.json").write_text(json.dumps({"dev": manifests.get("dev", []),
                                                       "holdout": manifests.get("holdout", [])}))
    (man / "devcore-100-v1.json").write_text(json.dumps({"dev_core": manifests.get("dev_core", [])}))
    (man / "a1-select-300-v1.json").write_text(json.dumps({"samples": manifests.get("selection", [])}))
    (base / "results").mkdir(exist_ok=True)
    (base / "results" / "dev.jsonl").write_text("")
    evidence = base / "evidence" / "fixture-run"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "1000.json").write_text("{}")
    (evidence / "manifest.json").write_text("{}")
    return base


def _v2_registry_doc(repo: Path, rows, roles):
    lookup, _, _ = rr.build_alias_map(rows, repo / "manifests")
    group_members = {}
    for r in rows:
        group_members.setdefault(r["group_id"], set()).add(r["canonical_id"])
    manifest_sets = {"dev": [rows[0]["canonical_id"]], "dev_core": [rows[0]["canonical_id"]],
                     "holdout": [], "selection_pool": []}
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
        "alias_unresolved_rows": [],
        "blocking_issues": [],
        "n_blocking_issues": 0,
        "exposure_artifacts": [{"path": "results/dev.jsonl",
                                "sha256": rr.sha256_file(repo / "results" / "dev.jsonl"),
                                "kind": "results_jsonl"}],
        "exposure_directories": [{
            "path": "evidence/fixture-run",
            "inventory_sha256": rr.inventory_hash(rr.directory_inventory(repo / "evidence" / "fixture-run")),
            "n_bundles": len(rr.directory_inventory(repo / "evidence" / "fixture-run")),
            "kind": "evidence_bundle_dir"}],
        "non_bundle_filenames": sorted(rr.NON_BUNDLE_FILENAMES),
        "roles": {k: sorted(set(v)) for k, v in roles.items()},
        "manifest_sets": manifest_sets,
        "membership_hashes": {k: rr.membership_hash(v) for k, v in roles.items()},
        "manifest_set_hashes": {k: rr.membership_hash(v) for k, v in manifest_sets.items()},
        "group_members": {g: sorted(m) for g, m in sorted(group_members.items())},
        "seed": None,
    }


class ValidatorTest(unittest.TestCase):
    """Fixtures were rebuilt for the v2 schema. Every assertion is preserved.

    The v2 schema requires a declared schema version, a complete dependency set, bound exposure
    artifacts, source-derived identities and a member-binding group hash. A v1-shaped fixture
    cannot satisfy those, so the fixture document is now generated by the production helpers.
    Two assertions were additionally strengthened: the valid fixture must verify source-derived
    identities and manifest memberships, and a mutated fixture must still be rejected.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.rows = _v2_rows()
        self.ids = {r["_tag"]: r["canonical_id"] for r in self.rows}
        self.group_of, _ = rr.build_groups(self.rows)
        for r in self.rows:
            r["group_id"] = self.group_of[r["canonical_id"]]
        self.rows[0]["role"] = rr.ROLE_DEVELOPMENT_CORE
        self.rows[1]["role"] = rr.ROLE_RESERVE
        self.rows[2]["role"] = rr.ROLE_QUARANTINE
        self.rows[2]["reference_status"] = "unresolved_reference"
        for r in self.rows:
            if r["role"] != rr.ROLE_RESERVE:
                # a record in any other role is not "no recorded use"
                r["exposure"]["no_recorded_use_in_scanned_artifacts"] = False
        self.roles = {rr.ROLE_DEVELOPMENT_CORE: [self.ids["dev"]],
                      rr.ROLE_RESERVE: [self.ids["res"]],
                      rr.ROLE_QUARANTINE: [self.ids["bad"]]}
        self.repo = _v2_repo(base, self.rows)
        self.samples = base / "samples.jsonl"
        self.samples.write_text("\n".join(json.dumps(r) for r in self.rows) + "\n")
        self.registry = base / "registry.json"
        self.registry.write_text(json.dumps(_v2_registry_doc(self.repo, self.rows, self.roles)))
        self.v = RegistryValidator(self.registry, self.samples, self.repo)

    def _rewrite(self, doc, rows=None):
        self.registry.write_text(json.dumps(doc))
        if rows is not None:
            self.samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_verify_all_passes(self) -> None:
        out = self.v.verify_all()
        self.assertEqual(out["samples"], 3)
        self.assertEqual(out["groups"], 3)
        self.assertEqual(out["identity_sources"]["identities_verified_from_sources"], 3)
        self.assertEqual(out["manifest_memberships"]["manifest_dev_core"], 1)

    def test_fingerprint_mismatch_is_rejected(self) -> None:
        bad = json.loads(self.registry.read_text())
        bad["fingerprints"]["dataset_index"] = "0" * 64
        self._rewrite(bad)
        with self.assertRaises(RegistryValidationError):
            RegistryValidator(self.registry, self.samples, self.repo).verify_all()

    def test_membership_hash_mismatch_is_rejected(self) -> None:
        bad = json.loads(self.registry.read_text())
        bad["membership_hashes"][rr.ROLE_RESERVE] = "0" * 64
        self._rewrite(bad)
        with self.assertRaises(RegistryValidationError):
            RegistryValidator(self.registry, self.samples, self.repo).verify_all()

    def test_development_request_with_reserve_id_is_rejected(self) -> None:
        with self.assertRaises(RegistryValidationError):
            self.v.validate_request(PURPOSE_DEVELOPMENT, [self.ids["dev"], self.ids["res"]])

    def test_unknown_id_is_rejected(self) -> None:
        with self.assertRaises(RegistryValidationError):
            self.v.validate_request(PURPOSE_DEVELOPMENT, ["mfb-nope"])

    def test_unresolved_id_is_rejected(self) -> None:
        with self.assertRaises(RegistryValidationError):
            self.v.validate_request(PURPOSE_DEVELOPMENT, [self.ids["bad"]])

    def test_development_request_with_group_conflict_is_rejected(self) -> None:
        """A development id that shares a verified group with a reserved id must be rejected."""
        for r in self.rows:
            r["image_byte_sha256"] = "shared"
        self.group_of, _ = rr.build_groups(self.rows)
        for r in self.rows:
            r["group_id"] = self.group_of[r["canonical_id"]]
        doc = _v2_registry_doc(self.repo, self.rows, self.roles)
        self._rewrite(doc, self.rows)
        v = RegistryValidator(self.registry, self.samples, self.repo)
        with self.assertRaises(RegistryValidationError):
            v.validate_request(PURPOSE_DEVELOPMENT, [self.ids["dev"]])

    def test_reserve_request_accepts_only_reserved(self) -> None:
        report = self.v.validate_request(PURPOSE_RESERVE, [self.ids["res"]])
        self.assertEqual(report["reserved_overlap"], 1)
        with self.assertRaises(RegistryValidationError):
            self.v.validate_request(PURPOSE_RESERVE, [self.ids["dev"]])

    def test_overlap_helper(self) -> None:
        self.assertEqual(self.v.overlap(["a", "b"], ["b", "c"]), {"b"})


class OfflineBoundaryTest(unittest.TestCase):
    def test_network_guard_blocks_connections(self) -> None:
        with NetworkGuard():
            with self.assertRaises(AssertionError):
                socket.socket().connect(("example.org", 443))

    def test_dotenv_is_stubbed(self) -> None:
        import dotenv
        self.assertFalse(dotenv.load_dotenv(), "dotenv must be stubbed in tests, not reading .env")


if __name__ == "__main__":
    unittest.main()
