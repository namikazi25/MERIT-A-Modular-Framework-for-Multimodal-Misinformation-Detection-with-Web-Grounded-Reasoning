"""Validator for the redesign sample registry (Task 0.2 / 0.2.1, offline).

Checks that a requested set of samples is admissible for a stated purpose. This is real
enforcement code: a flag such as ``sealed: true`` in a manifest is not treated as protection
anywhere in this project unless this validator (or equivalent code) actually runs.

Integrity is enforced by ``validate_request`` itself, on entry, with **no cached approval and
no compatibility bypass**. There is no caller-controlled ``already_verified`` flag. Before every
admission decision the validator refuses mutated in-memory state, reloads the registry and rows
if their bytes changed on disk, rebuilds every derived map, and re-hashes the dependencies.

What is checked
---------------
1.  Schema and grouping-policy versions are supported. A missing or unsupported version fails.
2.  The **complete expected dependency set** is declared, every dependency file exists, and each
    re-hashes from disk to its recorded digest. A caller-declared subset is refused.
3.  Every exposure artifact that produced the exposure flags is re-hashed from disk. A missing or
    changed artifact fails.
4.  The registry declares no blocking issue: an unresolved manifest or artifact reference means the
    affected identity cannot be bounded and admission is refused.
5.  The dataset-index fingerprint recomputes from the loaded rows.
6.  The sample-identity hash recomputes from the loaded rows.
7.  The group-binding hash recomputes from ``group_id -> sorted member ids``.
8.  The role-assignment hash recomputes from ``canonical_id -> role``.
9.  Membership hashes recompute for every role and manifest set.
10. The alias table is rebuilt from the hash-pinned manifests and compared, including each alias's
    recorded provenance, and no ambiguous candidate may sit in the reserve.
11. Role sets are consistent with the rows, and no reserved sample shares a verified group with an
    exposed, quarantined or conservatively excluded sample.
12. In-memory registry, rows, role set, manifest set, group-membership or by-id mutations after
    loading are rejected rather than silently approved.

What is NOT checked, and must not be claimed
--------------------------------------------
*   Checksums establish consistency against recorded references. They are **not** protection
    against an actor who can rewrite the artifacts and all trusted references together.
*   The validator does not re-run the exposure scan or re-derive grouping or near-duplicate links;
    it verifies the recorded results against hash-pinned inputs and internal consistency rules.
*   Exposure itself is limited to the artefacts that were scanned. Validating this registry does
    not establish that the reserve is contamination-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from scripts.redesign_registry import (
    EXPOSURE_FLAGS,
    NON_BUNDLE_FILENAMES,
    directory_inventory,
    inventory_hash,
    SUPPORTED_GROUPING_POLICIES,
    SUPPORTED_SCHEMA_VERSIONS,
    alias_ambiguity_hash,
    alias_mapping_hash,
    build_alias_map,
    dataset_index_fingerprint,
    group_binding_hash,
    identity_hash,
    canonical_sample_id,
    claim_sha256,
    membership_hash,
    role_assignment_hash,
    row_payload_hash,
    sha256_file,
    sha256_text,
)

NOT_YET_INTEGRATED = (
    "main.py, scripts/judge_study/* and the historical evaluation runners do not call this "
    "validator. Runner integration is a later task; until then the reserve is enforced only "
    "where this validator is called explicitly."
)

PURPOSE_DEVELOPMENT = "development"
PURPOSE_RESERVE = "evaluation_reserve"
PURPOSE_HISTORICAL = "historical_analysis"
VALID_PURPOSES = (PURPOSE_DEVELOPMENT, PURPOSE_RESERVE, PURPOSE_HISTORICAL)

RESERVED_ROLE = "reserved_not_evaluated"
QUARANTINE_ROLES = ("quarantined_unresolved", "quarantined_possible_duplicate")
UNRESOLVED_ROLES = set(QUARANTINE_ROLES) | {"unresolved_reference"}

# The complete dependency set a registry must declare. A caller-declared subset is refused.
EXPECTED_DEPENDENCIES = frozenset({
    "dataset_source_file",
    "image_hash_cache",
    "grouping_code",
    "development_manifest_split",
    "development_manifest_devcore",
    "development_manifest_selection",
})


class RegistryValidationError(Exception):
    """Raised when a request is not admissible."""


class RegistryValidator:
    """Validates a sample request against a registry, reloading and re-verifying every time.

    There is deliberately **no cached approval**. On every public request the validator:

    1. recomputes a digest of its in-memory registry document, rows and derived maps and
       **rejects** the request if any of them changed since they were loaded;
    2. re-reads the registry and samples files from disk and rebuilds every derived map if
       their bytes changed;
    3. re-hashes the complete required dependency set and every exposure artifact;
    4. re-runs the full integrity check against that freshly loaded state.

    A check on one object therefore cannot approve a subsequently changed object, whether the
    change happened on disk or in memory.
    """

    def __init__(self, registry_path: Path | str, samples_path: Path | str, repo_root: Path | str | None = None):
        self.registry_path = Path(registry_path)
        self.samples_path = Path(samples_path)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.registry: Dict[str, Any] = {}
        self.rows: List[Dict[str, Any]] = []
        self.by_id: Dict[str, Dict[str, Any]] = {}
        self.group_members: Dict[str, Set[str]] = {}
        self.roles: Dict[str, Set[str]] = {}
        self.manifest_sets: Dict[str, Set[str]] = {}
        self._loaded_disk_digest: Optional[str] = None
        self._loaded_content_digest: Optional[str] = None
        self._load()

    # -- loading and derived maps ------------------------------------------
    def _load(self) -> None:
        """Reload every input and rebuild every derived map from the current disk state."""
        self.registry = json.loads(self.registry_path.read_text())
        self.rows = [
            json.loads(line) for line in self.samples_path.read_text().splitlines() if line.strip()
        ]
        self.by_id = {r["canonical_id"]: r for r in self.rows}
        self.group_members = {}
        for row in self.rows:
            self.group_members.setdefault(row["group_id"], set()).add(row["canonical_id"])
        self.roles = {k: set(v) for k, v in (self.registry.get("roles") or {}).items()}
        self.manifest_sets = {k: set(v) for k, v in (self.registry.get("manifest_sets") or {}).items()}
        self._loaded_disk_digest = self._disk_digest()
        self._loaded_content_digest = self._content_digest()

    def _disk_digest(self) -> str:
        parts = []
        for path in (self.registry_path, self.samples_path):
            parts.append(sha256_file(path) if path.exists() else "MISSING")
        return sha256_text("|".join(parts))

    def _content_digest(self) -> str:
        """Digest of everything held in memory that admission decisions depend on."""
        rows_digest = row_payload_hash(self.rows)
        derived = sha256_text("|".join([
            sha256_text(json.dumps(self.registry, sort_keys=True, default=str)),
            sha256_text(json.dumps({k: sorted(map(str, v)) for k, v in self.roles.items()}, sort_keys=True)),
            sha256_text(json.dumps({k: sorted(map(str, v)) for k, v in self.manifest_sets.items()}, sort_keys=True)),
            sha256_text(json.dumps({k: sorted(map(str, v)) for k, v in self.group_members.items()}, sort_keys=True)),
            sha256_text(json.dumps(
                {k: [str(v.get("dataset_index")), str(v.get("group_id")), str(v.get("role"))]
                 for k, v in self.by_id.items()}, sort_keys=True)),
        ]))
        return sha256_text(rows_digest + "|" + derived)

    def sync(self) -> Dict[str, Any]:
        """Reject mutated in-memory state, then reload current disk state. Never cached."""
        if self._loaded_content_digest is not None:
            current = self._content_digest()
            if current != self._loaded_content_digest:
                raise RegistryValidationError(
                    "in-memory registry state changed after it was loaded (registry document, rows, "
                    "derived role/group/by-id maps or group_id/exposure/reference_status fields); "
                    "refusing to validate mutated state. Construct a new validator instead.")
        disk = self._disk_digest()
        reloaded = False
        if disk != self._loaded_disk_digest:
            self._load()
            reloaded = True
            if self._disk_digest() != self._loaded_disk_digest:
                raise RegistryValidationError("registry or samples file changed while reloading")
        return {"reloaded_from_disk": reloaded}

    # -- individual checks -------------------------------------------------
    def unknown_ids(self, ids: Iterable[str]) -> List[str]:
        return sorted({i for i in ids if i not in self.by_id})

    def unresolved_ids(self, ids: Iterable[str]) -> List[str]:
        return sorted({i for i in ids if self.by_id.get(i, {}).get("reference_status") != "resolved"})

    def verify_versions(self) -> None:
        schema = self.registry.get("schema_version")
        policy = self.registry.get("grouping_policy_version")
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            raise RegistryValidationError(
                f"unsupported or missing schema_version {schema!r}; supported: {SUPPORTED_SCHEMA_VERSIONS}")
        if policy not in SUPPORTED_GROUPING_POLICIES:
            raise RegistryValidationError(
                f"unsupported or missing grouping_policy_version {policy!r}; "
                f"supported: {SUPPORTED_GROUPING_POLICIES}")

    def verify_blocking_issues(self) -> None:
        """Unbounded unresolved references must fail explicitly, never warn and continue."""
        issues = self.registry.get("blocking_issues") or []
        if issues:
            kinds = sorted({i.get("kind", "unknown") for i in issues if isinstance(i, dict)})
            raise RegistryValidationError(
                f"registry declares {len(issues)} blocking issue(s) {kinds}: unresolved identities "
                "cannot be bounded, so this registry must not be used for admission")
        declared = self.registry.get("n_blocking_issues")
        if declared not in (0, None):
            raise RegistryValidationError(
                f"n_blocking_issues={declared} but no blocking issue list is present")

    def _dependency_path(self, name: str) -> Path:
        dep = (self.registry.get("dependencies") or {}).get(name)
        if not dep or not dep.get("path"):
            raise RegistryValidationError(f"registry does not declare dependency {name!r}")
        path = self.repo_root / dep["path"]
        if not path.exists():
            raise RegistryValidationError(f"dependency {name!r} is missing: {dep['path']}")
        return path

    def verify_dependencies(self) -> Dict[str, str]:
        """Require the complete expected dependency set and re-hash each file from disk."""
        deps = self.registry.get("dependencies") or {}
        declared_names = set(deps)
        missing_declared = sorted(EXPECTED_DEPENDENCIES - declared_names)
        if missing_declared:
            raise RegistryValidationError(
                f"registry does not declare the complete dependency set; missing {missing_declared}")
        unexpected = sorted(declared_names - EXPECTED_DEPENDENCIES)
        if unexpected:
            raise RegistryValidationError(
                f"registry declares unsupported dependency name(s) {unexpected}; expected exactly "
                f"{sorted(EXPECTED_DEPENDENCIES)}")
        seen: Dict[str, str] = {}
        for name in sorted(deps):
            dep = deps[name] or {}
            rel = dep.get("path")
            expected = dep.get("sha256")
            if not rel:
                raise RegistryValidationError(f"dependency {name!r} has no recorded path")
            path = self.repo_root / rel
            if not path.exists():
                raise RegistryValidationError(f"dependency {name!r} is missing: {rel}")
            actual = sha256_file(path)
            if expected != actual:
                raise RegistryValidationError(
                    f"dependency {name!r} ({rel}) does not match its recorded fingerprint: "
                    f"recorded={expected} actual={actual}")
            seen[name] = actual
        return seen

    def verify_exposure_artifacts(self) -> Dict[str, int]:
        """Re-hash every exposure artifact that produced the recorded exposure flags."""
        artifacts = self.registry.get("exposure_artifacts")
        if not artifacts:
            raise RegistryValidationError(
                "registry does not bind its exposure artifacts; exposure flags cannot be trusted")
        missing, mismatched = [], []
        for art in artifacts:
            rel = art.get("path")
            expected = art.get("sha256")
            if not rel:
                raise RegistryValidationError("exposure artifact entry has no path")
            path = self.repo_root / rel
            if not path.exists():
                missing.append(rel)
                continue
            if sha256_file(path) != expected:
                mismatched.append(rel)
        if missing:
            raise RegistryValidationError(f"exposure artifact(s) missing: {missing[:5]} (n={len(missing)})")
        if mismatched:
            raise RegistryValidationError(
                f"exposure artifact(s) changed since the registry was built: {mismatched[:5]} "
                f"(n={len(mismatched)})")
        return {"exposure_artifacts_verified": len(artifacts)}

    def verify_exposure_directories(self) -> Dict[str, int]:
        """Re-inventory every evidence run directory used by the exposure scan.

        The scan resolves exposure from the *set of bundle stems* in each directory, so that set
        is the dependency for those 8 directories, not a file digest. A missing directory, a
        changed inventory, or a directory that has become empty fails.
        """
        directories = self.registry.get("exposure_directories")
        if directories is None:
            raise RegistryValidationError(
                "registry does not bind its evidence-directory inventories; exposure flags cannot "
                "be trusted")
        if not directories:
            raise RegistryValidationError("registry declares no evidence-directory inventories")
        recorded_non_bundle = set(self.registry.get("non_bundle_filenames") or [])
        if recorded_non_bundle != set(NON_BUNDLE_FILENAMES):
            raise RegistryValidationError(
                "recorded non_bundle_filenames disagree with the inventory rule used by the builder: "
                f"recorded={sorted(recorded_non_bundle)} rule={sorted(NON_BUNDLE_FILENAMES)}")
        missing, changed, empty = [], [], []
        total = 0
        for entry in directories:
            rel = entry.get("path")
            if not rel:
                raise RegistryValidationError("evidence-directory entry has no path")
            path = self.repo_root / rel
            if not path.is_dir():
                missing.append(rel)
                continue
            stems = directory_inventory(path)
            total += len(stems)
            if not stems:
                empty.append(rel)
                continue
            if inventory_hash(stems) != entry.get("inventory_sha256"):
                changed.append(rel)
            elif entry.get("n_bundles") not in (None, len(stems)):
                changed.append(rel)
        if missing:
            raise RegistryValidationError(
                f"evidence directory(ies) missing: {missing[:5]} (n={len(missing)})")
        if empty:
            raise RegistryValidationError(
                f"evidence directory(ies) are now empty: {empty[:5]} (n={len(empty)})")
        if changed:
            raise RegistryValidationError(
                f"evidence directory inventory changed since the registry was built: {changed[:5]} "
                f"(n={len(changed)})")
        return {"directories_verified": len(directories), "bundles_inventoried": total}

    def verify_dataset_fingerprint(self) -> str:
        got = dataset_index_fingerprint(self.rows)
        expected = self.registry.get("fingerprints", {}).get("dataset_index")
        if expected != got:
            raise RegistryValidationError(
                f"dataset-index fingerprint mismatch: registry={expected} recomputed={got}")
        return got

    def verify_identity_hash(self) -> str:
        got = identity_hash(self.rows)
        expected = self.registry.get("fingerprints", {}).get("sample_identity")
        if expected != got:
            raise RegistryValidationError(
                f"sample-identity hash mismatch: registry={expected} recomputed={got}")
        return got

    def verify_group_binding(self) -> str:
        got = group_binding_hash(self.group_members)
        expected = self.registry.get("fingerprints", {}).get("group_binding")
        if expected != got:
            raise RegistryValidationError(
                "group-binding hash mismatch: the recorded grouping is not the grouping of the loaded "
                f"rows (registry={expected} recomputed={got})")
        return got

    def verify_declared_group_members(self) -> Dict[str, int]:
        """Require the declared group_members table to equal the row-derived grouping.

        The binding hash is computed from the rows. The registry also *declares* a
        ``group_members`` table; if that table were trusted or ignored, swapping two member
        lists in it would go unnoticed. Both are therefore compared exactly.
        """
        declared = self.registry.get("group_members")
        if not isinstance(declared, dict):
            raise RegistryValidationError(
                "registry does not declare a group_members table; declared grouping cannot be checked")
        derived = {gid: sorted(members) for gid, members in self.group_members.items()}
        declared_norm = {str(gid): sorted(members) for gid, members in declared.items()}
        if declared_norm != derived:
            diffs = []
            for gid in sorted(set(declared_norm) | set(derived)):
                if declared_norm.get(gid) != derived.get(gid):
                    diffs.append(f"{gid}: declared={declared_norm.get(gid)} derived={derived.get(gid)}")
            raise RegistryValidationError(
                f"declared group_members table disagrees with the row-derived grouping, "
                f"e.g. {diffs[:2]}")
        return {"declared_groups": len(derived)}

    def verify_role_assignment(self) -> str:
        got = role_assignment_hash(self.rows)
        expected = self.registry.get("fingerprints", {}).get("role_assignment")
        if expected != got:
            raise RegistryValidationError(
                f"role-assignment hash mismatch: registry={expected} recomputed={got}")
        return got

    def verify_row_payload(self) -> str:
        """Bind exposure flags, aliases, exclusion reasons and provenance, not only the role."""
        got = row_payload_hash(self.rows)
        expected = self.registry.get("fingerprints", {}).get("row_payload")
        if expected != got:
            raise RegistryValidationError(
                "row-payload hash mismatch: an identity, group, role, reference status, exposure "
                f"flag, alias, exclusion reason or provenance field changed (registry={expected} "
                f"recomputed={got})")
        return got

    def verify_identities_from_sources(self) -> Dict[str, int]:
        """Recompute canonical ids and content hashes from the pinned dataset source and cache.

        This does not hash the row table against its own recorded hash. It rebuilds the expected
        identity of every row from the dependency files and compares that with the loaded rows.
        """
        dataset_path = self._dependency_path("dataset_source_file")
        cache_path = self._dependency_path("image_hash_cache")
        records = json.loads(dataset_path.read_text())
        cache: Dict[str, Dict[str, Any]] = {}
        for line in cache_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item.get("image_path")] = item

        mismatches: List[str] = []
        for r in self.rows:
            index = r.get("dataset_index")
            if not isinstance(index, int) or index < 0 or index >= len(records):
                mismatches.append(f"{r.get('canonical_id')}: dataset_index out of range")
                continue
            rec = records[index]
            rel = rec.get("image_path")
            if rel != r.get("image_path_rel"):
                mismatches.append(f"{r.get('canonical_id')}: image_path_rel differs from the source record")
                continue
            expected_cid = canonical_sample_id(rel, str(rec.get("text", "")))
            if expected_cid != r.get("canonical_id"):
                mismatches.append(f"{r.get('canonical_id')}: canonical id does not match the source record")
            expected_claim = claim_sha256(str(rec.get("text", "")))
            if expected_claim != r.get("claim_sha256"):
                mismatches.append(f"{r.get('canonical_id')}: claim hash does not match the source record")
            cached = cache.get(rel)
            if cached is None:
                mismatches.append(f"{r.get('canonical_id')}: no cached image hash for {rel}")
            else:
                if cached.get("byte_sha256") and cached["byte_sha256"] != r.get("image_byte_sha256"):
                    mismatches.append(f"{r.get('canonical_id')}: byte hash differs from the cache")
                if cached.get("pixel_sha256") and cached["pixel_sha256"] != r.get("image_pixel_sha256"):
                    mismatches.append(f"{r.get('canonical_id')}: pixel hash differs from the cache")
        if mismatches:
            raise RegistryValidationError(
                f"{len(mismatches)} row identit(ies) disagree with the pinned sources, e.g. {mismatches[:3]}")
        return {"identities_verified_from_sources": len(self.rows)}

    def verify_manifest_memberships(self) -> Dict[str, int]:
        """Rebuild manifest memberships from the actual development manifests and compare."""
        manifests_dir = self.repo_root / self.registry.get("manifests_dir", "manifests")
        dataset_path = self._dependency_path("dataset_source_file")
        records = json.loads(dataset_path.read_text())
        by_rel = {rec.get("image_path"): i for i, rec in enumerate(records)}
        by_id = {r["canonical_id"]: r for r in self.rows}

        def members(entries) -> Set[str]:
            out: Set[str] = set()
            for item in entries or []:
                if not isinstance(item, dict):
                    continue
                rel = str(item.get("image_path", "")).lstrip("/")
                rel = "/" + rel if rel and not rel.startswith("/") else rel
                index = by_rel.get(rel)
                if index is None:
                    raise RegistryValidationError(
                        f"manifest entry {item.get('sample_id')!r} references {rel!r} which is not in "
                        "the pinned dataset source")
                cid = canonical_sample_id(rel, str(records[index].get("text", "")))
                if cid not in by_id:
                    raise RegistryValidationError(
                        f"manifest entry {item.get('sample_id')!r} resolves to {cid} which is not in "
                        "the registry rows")
                out.add(cid)
            return out

        expected_sets: Dict[str, Set[str]] = {}
        split = json.loads((manifests_dir / "split-500-v1.json").read_text())
        expected_sets["dev"] = members(split.get("dev"))
        expected_sets["holdout"] = members(split.get("holdout"))
        core = json.loads((manifests_dir / "devcore-100-v1.json").read_text())
        expected_sets["dev_core"] = members(core.get("dev_core"))
        selection = json.loads((manifests_dir / "a1-select-300-v1.json").read_text())
        expected_sets["selection_pool"] = members(selection.get("samples"))

        if not core.get("dev_core"):
            # guard against a silently empty dev-core manifest: the selection must be preserved
            raise RegistryValidationError("development-core manifest is empty; selection must be preserved")

        for name, expected in sorted(expected_sets.items()):
            recorded = self.manifest_sets.get(name)
            if recorded is None:
                raise RegistryValidationError(f"registry does not record manifest set {name!r}")
            if recorded != expected:
                raise RegistryValidationError(
                    f"manifest set {name!r} does not match the actual manifest "
                    f"(recorded={len(recorded)} manifest={len(expected)})")
        return {f"manifest_{k}": len(v) for k, v in sorted(expected_sets.items())}

    def verify_membership_hashes(self) -> None:
        for label, sets, hashes in (
            ("role", self.roles, self.registry.get("membership_hashes") or {}),
            ("manifest set", self.manifest_sets, self.registry.get("manifest_set_hashes") or {}),
        ):
            for name, ids in sorted(sets.items()):
                expected = hashes.get(name)
                if expected is None:
                    raise RegistryValidationError(f"no recorded membership hash for {label} {name!r}")
                got = membership_hash(ids)
                if got != expected:
                    raise RegistryValidationError(
                        f"membership hash mismatch for {label} {name!r}: recorded={expected} recomputed={got}")

    def verify_role_sets_match_rows(self) -> None:
        rows_reserve = {r["canonical_id"] for r in self.rows if r.get("role") == RESERVED_ROLE}
        declared_reserve = set(self.roles.get(RESERVED_ROLE, set()))
        if rows_reserve != declared_reserve:
            raise RegistryValidationError(
                "reserved role set does not match the reserved records in the samples file "
                f"(rows={len(rows_reserve)} declared={len(declared_reserve)})")
        known = set(self.roles)
        for row in self.rows:
            if row.get("role") not in known:
                raise RegistryValidationError(
                    f"record {row['canonical_id']} has role {row.get('role')!r} which is not declared "
                    "in the registry roles")

    def verify_alias_mapping(self) -> Dict[str, Any]:
        """Rebuild aliases from the hash-pinned manifests and compare with the recorded values."""
        manifests_dir = self.repo_root / self.registry.get("manifests_dir", "manifests")
        if not manifests_dir.exists():
            raise RegistryValidationError(f"manifests directory missing: {manifests_dir}")
        lookup, unresolved, _checked = build_alias_map(self.rows, manifests_dir)
        ambiguities = lookup.ambiguous
        # A fresh unresolved manifest or run reference is an unbounded identity. It must fail,
        # even when it does not change the alias mapping itself.
        if unresolved:
            sample = unresolved[:3]
            raise RegistryValidationError(
                f"{len(unresolved)} unresolved manifest reference(s) cannot be bounded: {sample}. "
                "Admission is refused while an identity is unresolved.")
        recorded_unresolved = self.registry.get("alias_unresolved_rows") or []
        if len(recorded_unresolved) != len(unresolved):
            raise RegistryValidationError(
                "recorded alias_unresolved_rows disagree with the rebuilt unresolved references "
                f"(recorded={len(recorded_unresolved)} rebuilt={len(unresolved)})")
        record = self.registry.get("fingerprints", {})
        if record.get("alias_mapping") != alias_mapping_hash(lookup):
            raise RegistryValidationError(
                "alias-mapping hash mismatch: the rebuilt alias table (mapping and provenance) does "
                "not match the recorded one")
        got_amb = alias_ambiguity_hash(ambiguities)
        if record.get("alias_ambiguity") != got_amb:
            raise RegistryValidationError(
                "alias-ambiguity hash mismatch: the rebuilt ambiguity set does not match the recorded one")
        declared = self.registry.get("alias_ambiguities") or {}
        if alias_ambiguity_hash(declared) != got_amb:
            raise RegistryValidationError(
                "recorded alias_ambiguities disagree with the rebuilt ambiguity set")
        reserved = self.roles.get(RESERVED_ROLE, set())
        for alias, info in sorted(declared.items()):
            candidates = info.get("candidates") or []
            if len(set(candidates)) < 2:
                raise RegistryValidationError(f"ambiguity {alias!r} records fewer than two candidates")
            unknown = [c for c in candidates if c not in self.by_id]
            if unknown:
                raise RegistryValidationError(
                    f"blocking ambiguity: alias {alias!r} references unknown sample ids {unknown}; "
                    "the affected identities cannot be bounded")
            bad = sorted(set(candidates) & reserved)
            if bad:
                raise RegistryValidationError(
                    f"ambiguous alias {alias!r} leaves candidate(s) {bad} in the reserve")
        return {"aliases": len(lookup), "ambiguities": len(ambiguities)}

    def verify_reserve_eligibility(self) -> Dict[str, int]:
        """A reserved record must be clean: no exposure flag, no manifest membership, resolved.

        This is the semantic counterpart to the hash checks. Without it a registry could declare
        every row reserved while the manifests and exposure records say otherwise, and admission
        would still succeed.
        """
        reserved = sorted(self.roles.get(RESERVED_ROLE, set()))
        manifest_members: Set[str] = set()
        for name, ids in self.manifest_sets.items():
            manifest_members |= set(ids)
        overlap = sorted(set(reserved) & manifest_members)
        if overlap:
            raise RegistryValidationError(
                f"{len(overlap)} reserved record(s) are also members of a recorded manifest set, "
                f"e.g. {overlap[:5]}")

        active_flags = [f for f in EXPOSURE_FLAGS if f != "no_recorded_use_in_scanned_artifacts"]
        problems: List[str] = []
        for cid in reserved:
            row = self.by_id.get(cid)
            if row is None:
                problems.append(f"{cid}: reserved id is not present in the rows")
                continue
            if row.get("reference_status") != "resolved":
                problems.append(f"{cid}: reference_status is {row.get('reference_status')!r}")
            flags = row.get("exposure") or {}
            raised = [f for f in active_flags if flags.get(f)]
            if raised:
                problems.append(f"{cid}: exposure flag(s) {raised}")
            if flags.get("no_recorded_use_in_scanned_artifacts") is not True:
                problems.append(f"{cid}: no_recorded_use_in_scanned_artifacts is not true")
            if row.get("exclusion_reason"):
                problems.append(f"{cid}: carries exclusion_reason {row['exclusion_reason']!r}")
            if row.get("exposure_evidence"):
                problems.append(f"{cid}: carries exclusion provenance {row['exposure_evidence'][:1]}")
        if problems:
            raise RegistryValidationError(
                f"{len(problems)} reserved record(s) are not eligible for the reserve, e.g. {problems[:3]}")
        return {"reserve_eligible": len(reserved)}

    def verify_reserve_disjointness(self) -> None:
        reserved = self.roles.get(RESERVED_ROLE, set())
        non_reserve = {r["canonical_id"] for r in self.rows if r.get("role") != RESERVED_ROLE}
        reserved_groups = {self.by_id[c]["group_id"] for c in reserved if c in self.by_id}
        other_groups = {self.by_id[c]["group_id"] for c in non_reserve}
        overlap = reserved_groups & other_groups
        if overlap:
            raise RegistryValidationError(
                f"{len(overlap)} verified group(s) contain both reserved and non-reserved samples, "
                f"e.g. {sorted(overlap)[:3]}")

    # -- aggregate ---------------------------------------------------------
    def verify_integrity(self) -> Dict[str, Any]:
        """Re-run every check against the currently loaded state. No result is cached."""
        self.verify_versions()
        self.verify_blocking_issues()
        deps = self.verify_dependencies()
        artifact_report = self.verify_exposure_artifacts()
        directory_report = self.verify_exposure_directories()
        report = {
            "schema_version": self.registry.get("schema_version"),
            "grouping_policy_version": self.registry.get("grouping_policy_version"),
            "dataset_fingerprint": "ok", "sample_identity": "ok", "group_binding": "ok",
            "declared_group_members": "ok", "role_assignment": "ok", "row_payload": "ok", "membership_hashes": "ok", "roles": "ok",
            "reserve_eligibility": "ok",
            "aliases": "ok", "reserve_disjointness": "ok", "blocking_issues": 0,
            "dependencies_verified": sorted(deps),
            "exposure_artifacts_verified": artifact_report["exposure_artifacts_verified"],
            "exposure_directories": directory_report,
            "groups": len(self.group_members), "samples": len(self.rows),
        }
        self.verify_dataset_fingerprint()
        self.verify_identity_hash()
        self.verify_group_binding()
        report["declared_groups"] = self.verify_declared_group_members()
        self.verify_role_assignment()
        self.verify_row_payload()
        self.verify_membership_hashes()
        self.verify_role_sets_match_rows()
        report["reserve_eligibility"] = self.verify_reserve_eligibility()
        report["identity_sources"] = self.verify_identities_from_sources()
        report["manifest_memberships"] = self.verify_manifest_memberships()
        report["alias_summary"] = self.verify_alias_mapping()
        self.verify_reserve_disjointness()
        return report

    # Public verification entry point. It also refuses mutated in-memory state.
    def verify_all(self) -> Dict[str, Any]:
        self.sync()
        return self.verify_integrity()

    # -- overlap -----------------------------------------------------------
    def overlap(self, set_a: Iterable[str], set_b: Iterable[str]) -> Set[str]:
        return set(set_a) & set(set_b)

    def group_conflicts(self, ids: Iterable[str], other_ids: Iterable[str]) -> Dict[str, Set[str]]:
        """Groups containing at least one id from each side."""
        groups_other: Dict[str, Set[str]] = {}
        for i in other_ids:
            row = self.by_id.get(i)
            if row:
                groups_other.setdefault(row["group_id"], set()).add(i)
        conflicts: Dict[str, Set[str]] = {}
        for i in ids:
            row = self.by_id.get(i)
            if row and row["group_id"] in groups_other:
                conflicts[row["group_id"]] = groups_other[row["group_id"]]
        return conflicts

    # -- request admission -------------------------------------------------
    def validate_request(self, purpose: str, ids: Sequence[str]) -> Dict[str, Any]:
        """Admit or reject a request. Integrity is enforced here, on entry, every time."""
        if purpose not in VALID_PURPOSES:
            raise RegistryValidationError(f"unknown purpose {purpose!r}; expected one of {VALID_PURPOSES}")

        sync_report = self.sync()
        integrity = self.verify_integrity()

        unknown = self.unknown_ids(ids)
        if unknown:
            raise RegistryValidationError(f"unknown canonical ids: {unknown[:10]} (n={len(unknown)})")

        unresolved = self.unresolved_ids(ids)
        if unresolved:
            raise RegistryValidationError(
                f"ids with unresolved reference mappings cannot be used: {unresolved[:10]} "
                f"(n={len(unresolved)})")

        reserved = sorted(self.roles.get(RESERVED_ROLE, set()))
        report: Dict[str, Any] = {
            "purpose": purpose, "n_requested": len(set(ids)),
            "schema_version": integrity.get("schema_version"),
            "reloaded_from_disk": sync_report["reloaded_from_disk"],
        }

        if purpose in (PURPOSE_DEVELOPMENT, PURPOSE_HISTORICAL):
            bad = sorted(set(ids) & set(reserved))
            if bad:
                raise RegistryValidationError(
                    f"{purpose} request contains {len(bad)} reserved samples, e.g. {bad[:5]}")
            conflicts = self.group_conflicts(ids, reserved)
            if conflicts:
                sample = list(conflicts.items())[:3]
                raise RegistryValidationError(
                    f"{purpose} request touches {len(conflicts)} group(s) that also contain reserved "
                    f"samples, e.g. {sample}")
            report["reserved_overlap"] = 0
            report["group_conflicts_with_reserve"] = 0

        if purpose == PURPOSE_RESERVE:
            outside = sorted(set(ids) - set(reserved))
            if outside:
                raise RegistryValidationError(
                    f"reserve request contains {len(outside)} non-reserved samples, e.g. {outside[:5]}")
            report["reserved_overlap"] = len(set(ids))
        return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a sample set against the redesign registry.")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--purpose", default=PURPOSE_DEVELOPMENT, choices=VALID_PURPOSES)
    ap.add_argument("--ids-file", help="file with one canonical id per line")
    ap.add_argument("--set-name", help="name of a membership set recorded in the registry")
    ap.add_argument("--self-check", action="store_true", help="verify integrity only")
    args = ap.parse_args(argv)

    v = RegistryValidator(args.registry, args.samples, args.repo_root)
    try:
        print(json.dumps(v.verify_all(), indent=2))
    except RegistryValidationError as exc:
        print(f"INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 1
    if args.self_check:
        return 0

    if args.ids_file:
        ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    elif args.set_name:
        ids = list(v.registry.get("roles", {}).get(args.set_name, [])) or \
              list(v.registry.get("manifest_sets", {}).get(args.set_name, []))
        if not ids:
            print(f"no such membership set: {args.set_name}", file=sys.stderr)
            return 2
    else:
        print("provide --ids-file or --set-name", file=sys.stderr)
        return 2

    try:
        report = v.validate_request(args.purpose, ids)
    except RegistryValidationError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
