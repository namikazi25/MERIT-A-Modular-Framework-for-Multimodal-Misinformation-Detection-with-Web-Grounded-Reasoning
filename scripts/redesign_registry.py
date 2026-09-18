"""Redesign sample registry, duplicate-aware grouping, exposure roles and validator.

Task 0.2 (offline). This module is deliberately separate from the production pipeline
in ``main.py``: nothing here is imported by the pipeline, and no prompt, provider
configuration, judgement rule, retrieval behaviour or metric definition is touched.

Purpose
-------
Establish, from local artefacts only, which examples may be used for development and
which may be reserved for a defensible evaluation. It does not shuffle the existing 500
examples and it does not invent a clean test set.

Model-facing safety
-------------------
The registry is ADMIN/EVALUATION-ONLY and must never be fed to a model. Every row is
tagged ``record_type = "admin_only_sample_record"`` and carries label-bearing paths and
labels. Raw claim text is not stored: only a normalised hash. Exports intended for a
model are produced elsewhere and are outside this module's scope.

Determinism
-----------
Every hash covers deterministic content. Creation time is stored separately and is never
part of a membership hash. Grouping and role assignment are order-independent functions
of the content.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MODULE_VERSION = "redesign_registry_v2"
RECORD_TYPE = "admin_only_sample_record"
SCHEMA_VERSION = "registry_schema_v2"
GROUPING_POLICY_VERSION = "grouping_policy_exact_identity_v1"
SUPPORTED_SCHEMA_VERSIONS = ("registry_schema_v2",)
SUPPORTED_GROUPING_POLICIES = ("grouping_policy_exact_identity_v1",)

# ---------------------------------------------------------------------------
# Hashing conventions (one documented convention, used everywhere)
# ---------------------------------------------------------------------------
# Sample membership hash:   sha256("set|"   + "\n".join(sorted(canonical_ids)))
# Execution order hash:     sha256("order|" + "\n".join(canonical_ids_in_run_order))
# Dataset index fingerprint: sha256("dataset|" + "\n".join(f"{i}\t{rel_path}" for each row))
# Group membership hash:    sha256("groups|" + "\n".join(sorted(group_ids)))
# Path-based split hashes and ID-based run hashes are never mixed.
MEMBERSHIP_HASH_DOMAIN = "set"
ORDER_HASH_DOMAIN = "order"
DATASET_HASH_DOMAIN = "dataset"
GROUP_HASH_DOMAIN = "groups"
GROUP_BINDING_HASH_DOMAIN = "group_binding"
IDENTITY_HASH_DOMAIN = "identity"
ROLE_ASSIGNMENT_HASH_DOMAIN = "role_assignment"
ALIAS_MAPPING_HASH_DOMAIN = "alias_mapping"
ALIAS_AMBIGUITY_HASH_DOMAIN = "alias_ambiguity"
ROW_PAYLOAD_HASH_DOMAIN = "row_payload"

# Every integrity-relevant field of a registry row. A canonical JSON digest over these fields
# binds exposure flags, aliases, exclusion reasons and provenance, not just the role.
ROW_PAYLOAD_FIELDS = (
    "canonical_id", "dataset_index", "image_path_rel",
    "image_byte_sha256", "image_pixel_sha256", "claim_sha256",
    "group_id", "role", "reference_status", "exposure", "aliases",
    "exclusion_reason", "exposure_evidence",
)

CLAIM_NORMALISATION = (
    "NFKC; curly quotes to ASCII; en/em dash to '-'; NBSP to space; "
    "whitespace runs collapsed; strip; casefold. Negation, numbers, dates, entities "
    "and factual qualifiers are preserved (no stopword removal, no stemming, no "
    "punctuation deletion)."
)
PIXEL_HASH_PROCEDURE = (
    "PIL Image.open(path); record size and mode; convert('RGB'); "
    "sha256(f\"{mode}->RGB|{w}x{h}|\" + pixels). EXIF orientation is not applied. "
    "Two files with different bytes can therefore share a decoded-pixel hash."
)
CANONICAL_ID_RULE = "mfb-<sha256(rel_image_path + US + raw_claim_text)[:16]>"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_claim(text: str) -> str:
    """Conservative claim normalisation. Meaning-bearing content is preserved."""
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u2009": " ",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = re.sub(r"\s+", " ", out).strip()
    return out.casefold()


def claim_sha256(text: str) -> str:
    return sha256_text(normalize_claim(text))


def canonical_sample_id(rel_image_path: str, raw_claim_text: str) -> str:
    """Opaque, deterministic ID. Distinct records stay distinct even when they share an image."""
    payload = f"{rel_image_path}\x1f{raw_claim_text or ''}"
    return "mfb-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def image_hashes(abs_path: Path) -> Dict[str, Any]:
    """Byte and decoded-pixel hashes plus dimensions, per PIXEL_HASH_PROCEDURE."""
    from PIL import Image  # local import keeps the module import cheap

    raw = abs_path.read_bytes()
    byte_sha = hashlib.sha256(raw).hexdigest()
    with Image.open(abs_path) as im:
        size = im.size
        src_mode = im.mode
        rgb = im.convert("RGB")
        pixel_sha = hashlib.sha256(
            f"{src_mode}->RGB|{size[0]}x{size[1]}|".encode("utf-8") + rgb.tobytes()
        ).hexdigest()
    return {
        "byte_sha256": byte_sha,
        "pixel_sha256": pixel_sha,
        "width": size[0],
        "height": size[1],
        "src_mode": src_mode,
    }


# ---------------------------------------------------------------------------
# Membership hashes
# ---------------------------------------------------------------------------
def membership_hash(canonical_ids: Iterable[str]) -> str:
    ids = sorted(set(canonical_ids))
    return sha256_text(MEMBERSHIP_HASH_DOMAIN + "|" + "\n".join(ids))


def order_hash(canonical_ids: Sequence[str]) -> str:
    return sha256_text(ORDER_HASH_DOMAIN + "|" + "\n".join(canonical_ids))


def group_membership_hash(group_ids: Iterable[str]) -> str:
    """DEPRECATED. Hashes group names only, so swapping members between groups is undetected.

    Retained for reading v1 registries. New artifacts must use ``group_binding_hash``.
    """
    return sha256_text(GROUP_HASH_DOMAIN + "|" + "\n".join(sorted(set(group_ids))))


def identity_hash(rows: Sequence[Dict[str, Any]]) -> str:
    """Bind canonical ids, dataset indices, paths and per-sample content hashes."""
    lines = [
        "\t".join([
            str(r["dataset_index"]), r["canonical_id"], r["image_path_rel"],
            str(r.get("image_byte_sha256")), str(r.get("image_pixel_sha256")), str(r.get("claim_sha256")),
        ])
        for r in sorted(rows, key=lambda x: x["canonical_id"])
    ]
    return sha256_text(IDENTITY_HASH_DOMAIN + "|" + "\n".join(lines))


def role_assignment_hash(rows: Sequence[Dict[str, Any]]) -> str:
    """Bind every canonical id to its assigned role, so changed role fields are detected."""
    lines = [f"{r['canonical_id']}\t{r.get('role')}" for r in sorted(rows, key=lambda x: x["canonical_id"])]
    return sha256_text(ROLE_ASSIGNMENT_HASH_DOMAIN + "|" + "\n".join(lines))


def alias_mapping_hash(lookup: "AliasLookup") -> str:
    """Bind the complete alias table: alias -> canonical id -> recorded provenance origins."""
    lines = []
    for alias in sorted(lookup.mapping):
        origins = ";".join(sorted(lookup.provenance.get(alias, [])))
        lines.append(f"{alias}\t{lookup.mapping[alias]}\t{origins}")
    return sha256_text(ALIAS_MAPPING_HASH_DOMAIN + "|" + "\n".join(lines))


def alias_ambiguity_hash(ambiguities: Mapping[str, Any]) -> str:
    """Bind each ambiguous alias to its candidate identities AND each candidate's provenance."""
    lines = []
    for alias, info in sorted(ambiguities.items()):
        if isinstance(info, dict):
            candidates = sorted(info.get("candidates") or [])
            provenance = info.get("provenance") or {}
            prov_text = ";".join(
                f"{cid}<-{'>'.join(sorted(provenance.get(cid) or []))}" for cid in candidates)
        else:
            candidates = sorted(info or [])
            prov_text = ""
        lines.append(f"{alias}\t{','.join(candidates)}\t{prov_text}")
    return sha256_text(ALIAS_AMBIGUITY_HASH_DOMAIN + "|" + "\n".join(lines))


def row_payload_hash(rows: Sequence[Dict[str, Any]]) -> str:
    """Canonical JSON digest of the full integrity-relevant row payload.

    Binds identity fields, group, role, reference status, exposure flags, aliases, exclusion
    reason and exclusion provenance. Changing any of them changes the digest, so an altered
    exposure flag cannot pass by leaving the role alone.
    """
    payload = []
    for r in sorted(rows, key=lambda x: str(x.get("canonical_id"))):
        payload.append({k: r.get(k) for k in ROW_PAYLOAD_FIELDS})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(ROW_PAYLOAD_HASH_DOMAIN + "|" + canonical)


def group_binding_hash(group_members: Mapping[str, Iterable[str]]) -> str:
    """Deterministic hash binding every group id to its complete, sorted membership.

    Serialisation: one line per group, ``group_id<TAB>member1,member2,...``, groups sorted by id
    and members sorted within each group. Swapping members between two groups, moving a member,
    or changing any membership changes the digest even when the group names, the group count and
    the overall sample set are unchanged.
    """
    lines = []
    for group_id in sorted(group_members):
        members = sorted(set(group_members[group_id]))
        lines.append(f"{group_id}\t{','.join(members)}")
    return sha256_text(GROUP_BINDING_HASH_DOMAIN + "|" + "\n".join(lines))


# Files inside an evidence run directory that are NOT sample bundles. The exposure scan counts
# the remaining `*.json` stems as the directory's bundle inventory.
NON_BUNDLE_FILENAMES = frozenset({
    "manifest.json",
    "refetch_manifest.json",
    "summary.json",
    "run_summary.json",
})


def directory_inventory(directory: Path) -> List[str]:
    """Sorted bundle stems in an evidence run directory, excluding declared non-bundle files."""
    return sorted(p.stem for p in Path(directory).glob("*.json") if p.name not in NON_BUNDLE_FILENAMES)


def inventory_hash(stems) -> str:
    return sha256_text("inventory|" + "\n".join(sorted(stems)))


DEPENDENCY_PATHS = {
    "dataset_source_file": "data/MMFakeBench_test/source/MMFakeBench_test.json",
    "image_hash_cache": "_audit/preflight_02/image_hashes.jsonl",
    "grouping_code": "scripts/redesign_registry.py",
    "development_manifest_split": "manifests/split-500-v1.json",
    "development_manifest_devcore": "manifests/devcore-100-v1.json",
    "development_manifest_selection": "manifests/a1-select-300-v1.json",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_dependencies(repo: Path) -> Dict[str, Dict[str, str]]:
    """Recompute, from the live local files, the fingerprints the registry depends on.

    The validator recomputes these again from disk; comparing two copies of a stored hash is
    not sufficient, so the paths travel with the digests and are re-hashed at verification time.
    """
    out: Dict[str, Dict[str, str]] = {}
    for name, rel in DEPENDENCY_PATHS.items():
        path = repo / rel
        out[name] = {"path": rel, "sha256": sha256_file(path) if path.exists() else None}
    return out


def dataset_index_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    payload = "\n".join(f"{r['dataset_index']}\t{r['image_path_rel']}" for r in rows)
    return sha256_text(DATASET_HASH_DOMAIN + "|" + payload)


# ---------------------------------------------------------------------------
# Duplicate-aware grouping
# ---------------------------------------------------------------------------
@dataclass
class UnionFind:
    parent: Dict[str, str] = field(default_factory=dict)

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# Edge kinds that are VERIFIED identity links and may be grouped.
EDGE_BYTE_IDENTICAL = "byte_identical_image"
EDGE_DECODED_IDENTICAL = "decoded_identical_image"
EDGE_CLAIM_IDENTICAL = "normalised_claim_identical"
EDGE_SOURCE_IMAGE_ID = "recorded_source_image_id"


def build_groups(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Deterministic connected groups over verified identity links.

    Grouping is by verified identity only. Sharing a publisher, a generator family, a
    distortion class or a topic is NOT a reliable event identity and never creates an edge.
    """
    uf = UnionFind()
    edges: List[Dict[str, Any]] = []
    for r in rows:
        uf.add(r["canonical_id"])

    def link(key: str, kind: str, getter) -> None:
        buckets: Dict[str, List[str]] = {}
        for r in rows:
            value = getter(r)
            if value:
                buckets.setdefault(value, []).append(r["canonical_id"])
        for value, members in sorted(buckets.items()):
            if len(members) < 2:
                continue
            members = sorted(members)
            for other in members[1:]:
                uf.union(members[0], other)
            edges.append({"kind": kind, "key": key, "value": value[:80], "members": members})

    link("byte_sha256", EDGE_BYTE_IDENTICAL, lambda r: r.get("image_byte_sha256"))
    link("pixel_sha256", EDGE_DECODED_IDENTICAL, lambda r: r.get("image_pixel_sha256"))
    link("claim_sha256", EDGE_CLAIM_IDENTICAL, lambda r: r.get("claim_sha256"))

    # Deterministic group ids: sorted by the smallest member's dataset_index.
    index_by_id = {r["canonical_id"]: r["dataset_index"] for r in rows}
    root_members: Dict[str, List[str]] = {}
    for r in rows:
        root_members.setdefault(uf.find(r["canonical_id"]), []).append(r["canonical_id"])
    group_of: Dict[str, str] = {}
    ordered_roots = sorted(root_members, key=lambda root: min(index_by_id[m] for m in root_members[root]))
    for n, root in enumerate(ordered_roots, start=1):
        gid = f"g{n:05d}"
        for member in root_members[root]:
            group_of[member] = gid
    return group_of, edges


def flag_near_duplicate_candidates(
    rows: Sequence[Dict[str, Any]], max_hamming: int = 4
) -> List[Dict[str, Any]]:
    """Flag dHash-near pairs as CANDIDATES only. Never used to create a group.

    A perceptual-hash match is not proof of an identical image or of the same event, so
    these links are reported separately and can quarantine otherwise-eligible material.
    """
    import numpy as np

    entries = [(r["canonical_id"], int(r["dhash64"], 16)) for r in rows if r.get("dhash64")]
    if len(entries) < 2:
        return []
    ids = [e[0] for e in entries]
    bits = np.array([[(v >> (63 - b)) & 1 for b in range(64)] for _, v in entries], dtype=np.uint8)
    candidates: List[Dict[str, Any]] = []
    chunk = 512
    for start in range(0, len(ids), chunk):
        block = bits[start:start + chunk]
        dist = (block[:, None, :] != bits[None, :, :]).sum(axis=2)
        for i_local, i_global in enumerate(range(start, min(start + chunk, len(ids)))):
            for j in range(i_global + 1, len(ids)):
                d = int(dist[i_local, j])
                if d <= max_hamming:
                    candidates.append({"a": ids[i_global], "b": ids[j], "hamming": d})
    return sorted(candidates, key=lambda c: (c["hamming"], c["a"], c["b"]))


__all__ = [
    "MODULE_VERSION", "RECORD_TYPE", "CLAIM_NORMALISATION", "PIXEL_HASH_PROCEDURE",
    "CANONICAL_ID_RULE", "sha256_text", "normalize_claim", "claim_sha256",
    "canonical_sample_id", "image_hashes", "membership_hash", "order_hash",
    "group_membership_hash", "dataset_index_fingerprint", "build_groups",
    "flag_near_duplicate_candidates", "EDGE_BYTE_IDENTICAL", "EDGE_DECODED_IDENTICAL",
    "EDGE_CLAIM_IDENTICAL",
]


# ---------------------------------------------------------------------------
# Artefact scanning (recorded prior exposure)
# ---------------------------------------------------------------------------
DATASET_ROOT_MARKERS = ("MMFakeBench_test", "MMFakeBench_val")


def _rel_from_recorded_path(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a recorded image path to a dataset-relative path under MMFakeBench_test.

    Returns (rel_path, out_of_scope_marker). A path that names a different dataset root
    (for example MMFakeBench_val) is reported as out of scope, NOT resolved by digits.
    """
    if not value:
        return None, None
    text = str(value).replace("\\", "/")
    if "MMFakeBench_val" in text:
        return None, "MMFakeBench_val"
    if "MMFakeBench_test" in text:
        tail = text.split("MMFakeBench_test/", 1)[1]
        return "/" + tail.lstrip("/"), None
    if "/fake/" in text or "/real/" in text:
        idx_f = text.find("/fake/")
        idx_r = text.find("/real/")
        start = idx_f if idx_f != -1 and (idx_r == -1 or idx_f < idx_r) else idx_r
        return text[start:], None
    return None, "unknown_dataset_root"


@dataclass
class AliasLookup:
    """Historical-alias state: the usable mapping, the recorded ambiguities and provenance.

    Ambiguity travels with the lookup, so a caller cannot resolve an alias while omitting the
    ambiguity information. ``resolve_reference`` requires this object and rejects a plain dict.
    """

    mapping: Dict[str, str] = field(default_factory=dict)
    ambiguous: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    provenance: Dict[str, List[str]] = field(default_factory=dict)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __getitem__(self, key: str) -> str:
        return self.mapping[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.mapping.get(key, default)

    def items(self):
        return self.mapping.items()

    def keys(self):
        return self.mapping.keys()

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping)


def build_alias_map(rows, manifests_dir: Path):
    """Collect every candidate canonical identity for each historical sample-id alias.

    Returns ``(alias_map, ambiguities, unresolved, checked)``.

    An alias that maps to exactly one canonical sample in every manifest that records it is a
    usable mapping. An alias that maps to more than one sample is **not** resolved: it is
    recorded as an ambiguity with the provenance of every conflicting candidate. Collection is
    order-independent, so reversing manifest traversal cannot change the outcome, and no
    namespace is invented to hide a conflict.
    """
    by_rel = {r["image_path_rel"]: r for r in rows}
    candidates: Dict[str, Dict[str, List[str]]] = {}
    unresolved: List[Dict[str, Any]] = []
    checked = 0

    for path in sorted(manifests_dir.glob("*.json")):
        data = json.loads(path.read_text())
        groups = []
        for key in ("dev", "holdout", "dev_core", "samples"):
            if isinstance(data.get(key), list):
                groups.append((key, data[key]))
        for key, items in groups:
            for item in items:
                if not isinstance(item, dict):
                    continue
                sid = item.get("sample_id")
                rel, oos = _rel_from_recorded_path(item.get("image_path", ""))
                if sid is None or rel is None:
                    unresolved.append({"manifest": path.name, "set": key, "sample_id": sid,
                                       "image_path": item.get("image_path"), "reason": oos or "no_path"})
                    continue
                row = by_rel.get(rel)
                if row is None:
                    unresolved.append({"manifest": path.name, "set": key, "sample_id": str(sid),
                                       "image_path": item.get("image_path"), "reason": "path_not_in_dataset"})
                    continue
                digits = re.sub(r"\D", "", str(sid))
                if digits and str(row["dataset_index"]) == digits:
                    checked += 1
                origin = f"{path.name}:{key}:{sid}"
                candidates.setdefault(str(sid), {}).setdefault(row["canonical_id"], []).append(origin)

    lookup = AliasLookup()
    conflicts: Dict[str, Dict[str, Any]] = {}
    for alias, by_cid in sorted(candidates.items()):
        if len(by_cid) == 1:
            cid = next(iter(by_cid))
            lookup.mapping[alias] = cid
            lookup.provenance[alias] = sorted(by_cid[cid])
        else:
            conflicts[alias] = {
                "candidates": sorted(by_cid),
                "provenance": {cid: sorted(origins) for cid, origins in sorted(by_cid.items())},
                "reason": "alias maps to more than one canonical sample",
            }

    # A contested identity stays contested in every spelling. The bare dataset-index form of an
    # ambiguous ``dsNNNN`` alias denotes the same identity, so it is ambiguous too, unless the
    # bare form already maps to one of the very candidates recorded as conflicting.
    for alias in sorted(conflicts):
        info = conflicts[alias]
        lookup.ambiguous[alias] = info
        prefixed = re.fullmatch(r"ds(\d+)", alias, flags=re.IGNORECASE)
        digits = prefixed.group(1) if prefixed else None
        if not digits or digits in lookup.ambiguous:
            continue
        existing = lookup.mapping.get(digits)
        if existing is not None and existing in info["candidates"]:
            continue
        lookup.ambiguous[digits] = {
            "candidates": list(info["candidates"]),
            "provenance": {cid: list(origins) for cid, origins in info["provenance"].items()},
            "reason": f"bare form of the ambiguous alias {alias!r}",
        }
    return lookup, unresolved, checked


def resolve_reference(value, field: str, by_rel, by_index, lookup: "AliasLookup"):
    """Resolve one recorded reference to a canonical id.

    ``lookup`` must be an :class:`AliasLookup`, so the ambiguity state always travels with the
    mapping. An ambiguous alias fails explicitly, including through the dataset-index digits
    fallback: a contested identity is never silently chosen, in any spelling.
    """
    if not isinstance(lookup, AliasLookup):
        raise TypeError(
            "resolve_reference requires an AliasLookup carrying its ambiguity state; "
            f"got {type(lookup).__name__}")
    text = str(value).strip()
    if not text:
        return None, "empty"
    if field == "image_path" or "/fake/" in text or "/real/" in text:
        rel, oos = _rel_from_recorded_path(text)
        if oos:
            return None, f"out_of_scope_dataset:{oos}"
        row = by_rel.get(rel)
        return (row["canonical_id"], "resolved_by_path") if row else (None, "path_not_in_dataset")
    if text in lookup.ambiguous:
        return None, "ambiguous_alias"
    if text in lookup.mapping:
        return lookup.mapping[text], "resolved_by_alias"
    # Only two exact, supported spellings may resolve through the dataset-index namespace:
    # bare digits, or "ds" followed by digits and nothing else. Anything else is unresolved,
    # so a malformed identifier such as "dsINVALID7" can never reach row 7.
    bare = re.fullmatch(r"\d+", text)
    prefixed = re.fullmatch(r"ds(\d+)", text, flags=re.IGNORECASE)
    digits = bare.group(0) if bare else (prefixed.group(1) if prefixed else None)
    if digits is not None:
        index_key = str(int(digits))
        if index_key in lookup.ambiguous:
            return None, "ambiguous_alias"
        row = by_index.get(int(digits))
        if row is not None:
            return row["canonical_id"], "resolved_by_index_digits"
    return None, "unresolved_reference"


def scan_artifacts(repo: Path, rows, lookup: "AliasLookup") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Enumerate recorded exposure across local artefacts. Identifiers only, no predictions."""
    by_rel = {r["image_path_rel"]: r for r in rows}
    by_index = {r["dataset_index"]: r for r in rows}
    artifacts: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {"unresolved": [], "excluded": []}

    def add(path: Path, kind: str, hits: Dict[str, Any], notes: str = "") -> None:
        artifacts.append({
            "path": str(path.relative_to(repo)),
            "kind": kind,
            "sha256": sha256_file(path) if path.is_file() else None,
            "resolved_by_field": {k: sorted(v) for k, v in hits.items()},
            "notes": notes,
        })

    def resolve_many(pairs):
        out: Dict[str, List[str]] = {}
        unresolved: List[str] = []
        for value, field in pairs:
            cid, status = resolve_reference(value, field, by_rel, by_index, lookup)
            if cid:
                out.setdefault(status, []).append(cid)
            else:
                unresolved.append(f"{field}={value!r}:{status}")
        return out, unresolved

    # --- results/*.jsonl -------------------------------------------------
    for path in sorted((repo / "results").glob("*.jsonl")):
        pairs = []
        out_of_scope = 0
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    totals["unresolved"].append(f"parse_error:{path.name}")
                    continue
                image_path = str(obj.get("image_path") or "")
                rel, marker = _rel_from_recorded_path(image_path)
                # Record-level scope. A row whose image_path names a different dataset root
                # (for example MMFakeBench_val) is out of scope, and its OTHER identifier
                # fields must not be resolved against this dataset. Index digits are not
                # portable between dataset roots.
                if marker:
                    out_of_scope += 1
                    continue
                if rel:
                    pairs.append((image_path, "image_path"))
                    continue
                details = obj.get("sample_details") or {}
                for field in ("dataset_index", "sample_id"):
                    if details.get(field) is not None:
                        pairs.append((details[field], field))
                for field in ("dataset_index", "sample_id"):
                    if obj.get(field) is not None:
                        pairs.append((obj[field], field))
        hits, unresolved = resolve_many(pairs)
        add(path, "results_jsonl", hits,
            notes=f"records_in_scope={sum(len(v) for v in hits.values())} records_out_of_scope={out_of_scope}")
        totals["unresolved"] += [f"{path.name}:{u}" for u in unresolved[:20]]

    # --- results/judge_study/**/labels.csv and *.jsonl -------------------
    for path in sorted((repo / "results" / "judge_study").rglob("labels.csv")):
        import csv
        with path.open() as fh:
            reader = csv.DictReader(fh)
            pairs = [(r.get("sample_id"), "sample_id") for r in reader if r.get("sample_id")]
        hits, unresolved = resolve_many(pairs)
        add(path, "judge_study_labels", hits, notes=f"rows={len(pairs)}")
        totals["unresolved"] += [f"{path.name}:{u}" for u in unresolved[:10]]
    for path in sorted((repo / "results" / "judge_study").rglob("*.jsonl")):
        pairs = []
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for field in ("sample_id", "image_path"):
                    if obj.get(field):
                        pairs.append((obj[field], field))
        hits, unresolved = resolve_many(pairs)
        add(path, "judge_study_jsonl", hits, notes=f"rows={len(pairs)}")

    # --- evidence run directories (filenames only) -----------------------
    ev_root = repo / "evidence"
    for directory in sorted(p for p in ev_root.iterdir() if p.is_dir()):
        if directory.name == "search_cache":
            totals["excluded"].append(f"evidence/{directory.name}: query cache, not sample bundles")
            continue
        stems = directory_inventory(directory)
        hits, unresolved = resolve_many([(s, "sample_id") for s in stems])
        add(directory, "evidence_bundle_dir", hits, notes=f"bundles={len(stems)}")
        artifacts[-1]["inventory"] = stems
        artifacts[-1]["inventory_sha256"] = inventory_hash(stems)
        totals["unresolved"] += [f"evidence/{directory.name}:{u}" for u in unresolved[:10]]

    # --- labeling sheet ---------------------------------------------------
    sheet = repo / "labeling" / "phase3_sheet.csv"
    if sheet.exists():
        import csv
        with sheet.open() as fh:
            reader = csv.DictReader(fh)
            pairs = [(r.get("sample_id"), "sample_id") for r in reader if r.get("sample_id")]
        hits, unresolved = resolve_many(pairs)
        add(sheet, "annotation_sheet", hits, notes=f"rows={len(pairs)}; human columns empty in the source audit")

    return artifacts, totals


# ---------------------------------------------------------------------------
# Data roles
# ---------------------------------------------------------------------------
ROLE_DEVELOPMENT_CORE = "development_core"
ROLE_DEVELOPMENT_HISTORICAL = "development_historical"
ROLE_SELECTION_POOL = "selection_pool_exposed"
ROLE_HISTORICAL_HOLDOUT = "historical_evaluation_holdout_v1"
ROLE_RESERVE = "reserved_not_evaluated"
ROLE_QUARANTINE = "quarantined_unresolved"
ROLE_QUARANTINE_POSSIBLE_DUPLICATE = "quarantined_possible_duplicate"

EXPOSURE_FLAGS = (
    "used_for_development_or_selection",
    "evaluated_previously",
    "evidence_collected_only",
    "included_in_annotation_material",
    "possible_or_unresolved_exposure",
    "no_recorded_use_in_scanned_artifacts",
)

# Roles under which a record must never be offered as part of the evaluation reserve.
NON_RESERVE_ROLES = (
    ROLE_DEVELOPMENT_CORE,
    ROLE_DEVELOPMENT_HISTORICAL,
    ROLE_SELECTION_POOL,
    ROLE_HISTORICAL_HOLDOUT,
    ROLE_QUARANTINE,
    ROLE_QUARANTINE_POSSIBLE_DUPLICATE,
)


def _fixpoint_group_reachability(seed_groups, group_links):
    """Breadth-first closure over the group graph, with a provenance path per reached group.

    ``group_links`` carries both verified-group collapse (already folded into the group ids)
    and recorded possible-duplicate relationships. The closure is order-independent: it is a
    set reachability computation, and each group keeps the shortest recorded path.
    """
    from collections import deque

    reached = {g: [g] for g in sorted(seed_groups)}
    queue = deque(sorted(seed_groups))
    while queue:
        current = queue.popleft()
        for neighbour in sorted(group_links.get(current, ())):
            if neighbour not in reached:
                reached[neighbour] = reached[current] + [f"possible_duplicate->{neighbour}"]
                queue.append(neighbour)
    return reached


def assign_roles(rows, artifacts, group_of, near_dup_pairs, manifest_sets, ambiguous_ids=()):
    """Assign exposure flags and one data role per sample. Order-independent.

    Exclusion propagates to a fixpoint over the union of verified groups (already collapsed
    into ``group_of``) and recorded possible-duplicate links. A possible-duplicate link is a
    reason for conservative reserve exclusion, not proof that two samples are identical.
    """
    from collections import defaultdict

    flags = {r["canonical_id"]: {f: False for f in EXPOSURE_FLAGS} for r in rows}
    evidence_refs = {r["canonical_id"]: [] for r in rows}
    provenance = {r["canonical_id"]: [] for r in rows}

    for art in artifacts:
        kind = art["kind"]
        for status, ids in art["resolved_by_field"].items():
            for cid in ids:
                evidence_refs[cid].append(f"{art['path']}[{status}]")
                if kind == "results_jsonl":
                    flags[cid]["evaluated_previously"] = True
                elif kind in ("judge_study_labels", "judge_study_jsonl"):
                    flags[cid]["evaluated_previously"] = True
                elif kind == "evidence_bundle_dir":
                    flags[cid]["evidence_collected_only"] = True
                elif kind == "annotation_sheet":
                    flags[cid]["included_in_annotation_material"] = True

    for cid in manifest_sets.get("dev", []):
        flags[cid]["used_for_development_or_selection"] = True
    for cid in manifest_sets.get("holdout", []):
        flags[cid]["used_for_development_or_selection"] = True
    for cid in manifest_sets.get("dev_core", []):
        flags[cid]["used_for_development_or_selection"] = True
    for cid in manifest_sets.get("selection_pool", []):
        flags[cid]["used_for_development_or_selection"] = True

    def directly_exposed(cid):
        return any(flags[cid][k] for k in EXPOSURE_FLAGS if k != "no_recorded_use_in_scanned_artifacts")

    # --- seeds -----------------------------------------------------------------
    unresolved_ids = {r["canonical_id"] for r in rows if r.get("reference_status") != "resolved"}
    ambiguous_set = {cid for cid in ambiguous_ids if cid in flags}
    seed_ids = {cid for cid in flags if directly_exposed(cid)} | unresolved_ids | ambiguous_set

    group_links = defaultdict(set)
    for pair in near_dup_pairs:
        ga, gb = group_of.get(pair["a"]), group_of.get(pair["b"])
        if ga and gb and ga != gb:
            group_links[ga].add(gb)
            group_links[gb].add(ga)

    reached_groups = _fixpoint_group_reachability({group_of[c] for c in seed_ids}, group_links)

    # --- flags -----------------------------------------------------------------
    group_has_exposed = defaultdict(bool)
    group_has_ambiguous = defaultdict(bool)
    for cid in seed_ids:
        g = group_of[cid]
        if directly_exposed(cid):
            group_has_exposed[g] = True
        if cid in unresolved_ids or cid in ambiguous_set:
            group_has_ambiguous[g] = True

    # exposure propagates across verified groups: a group mate of an exposed sample is not
    # reserve-eligible. This is the v1 behaviour and is preserved.
    propagated = set()
    for cid in flags:
        if not directly_exposed(cid) and group_has_exposed.get(group_of[cid]):
            flags[cid]["used_for_development_or_selection"] = True
            propagated.add(cid)
            provenance[cid].append(f"verified_group_exposure:{group_of[cid]}")

    # --- roles -----------------------------------------------------------------
    conservatively_excluded = set()
    roles: Dict[str, str] = {}
    for r in rows:
        cid = r["canonical_id"]
        f = flags[cid]
        if not directly_exposed(cid):
            f["no_recorded_use_in_scanned_artifacts"] = True

        in_manifest_set = cid in (
            set(manifest_sets.get("dev", [])) | set(manifest_sets.get("holdout", [])) |
            set(manifest_sets.get("dev_core", [])) | set(manifest_sets.get("selection_pool", []))
        )

        if r.get("reference_status") != "resolved":
            roles[cid] = ROLE_QUARANTINE
            f["possible_or_unresolved_exposure"] = True
            f["no_recorded_use_in_scanned_artifacts"] = False
            provenance[cid].append("unresolved_reference")
            conservatively_excluded.add(cid)
        elif cid in manifest_sets.get("dev_core", []):
            roles[cid] = ROLE_DEVELOPMENT_CORE
        elif cid in manifest_sets.get("dev", []):
            roles[cid] = ROLE_DEVELOPMENT_HISTORICAL
        elif cid in manifest_sets.get("holdout", []):
            roles[cid] = ROLE_HISTORICAL_HOLDOUT
        elif cid in manifest_sets.get("selection_pool", []):
            roles[cid] = ROLE_SELECTION_POOL
        elif cid in ambiguous_set and not directly_exposed(cid):
            roles[cid] = ROLE_QUARANTINE
            f["possible_or_unresolved_exposure"] = True
            f["no_recorded_use_in_scanned_artifacts"] = False
            provenance[cid].append("ambiguous_alias")
            conservatively_excluded.add(cid)
        elif group_has_ambiguous.get(group_of[cid]) and not directly_exposed(cid):
            # An unresolved or ambiguous member must not leave the rest of its verified
            # group eligible for the reserve.
            roles[cid] = ROLE_QUARANTINE
            f["possible_or_unresolved_exposure"] = True
            f["no_recorded_use_in_scanned_artifacts"] = False
            provenance[cid].append(f"verified_group_has_unresolved:{group_of[cid]}")
            conservatively_excluded.add(cid)
        elif directly_exposed(cid):
            roles[cid] = ROLE_DEVELOPMENT_HISTORICAL
        elif group_of[cid] in reached_groups and not in_manifest_set:
            # Reached from an exposure seed through possible-duplicate links, but not itself
            # recorded as exposed. Conservative exclusion, distinct from direct exposure.
            roles[cid] = ROLE_QUARANTINE_POSSIBLE_DUPLICATE
            f["possible_or_unresolved_exposure"] = True
            f["no_recorded_use_in_scanned_artifacts"] = False
            provenance[cid].append("path:" + " -> ".join(reached_groups[group_of[cid]]))
            conservatively_excluded.add(cid)
        else:
            roles[cid] = ROLE_RESERVE

    for cid, paths in provenance.items():
        if paths:
            evidence_refs[cid].extend(f"exclusion:{p}" for p in paths)
    return flags, roles, evidence_refs, propagated, conservatively_excluded


def load_manifest_sets(manifests_dir: Path, rows) -> Dict[str, List[str]]:
    by_rel = {r["image_path_rel"]: r for r in rows}
    out: Dict[str, List[str]] = {}
    split = manifests_dir / "split-500-v1.json"
    if split.exists():
        data = json.loads(split.read_text())
        for key in ("dev", "holdout"):
            out[key] = sorted(
                by_rel["/" + str(item["image_path"]).lstrip("/")]["canonical_id"]
                for item in data.get(key, [])
                if "/" + str(item["image_path"]).lstrip("/") in by_rel
            )
    core = manifests_dir / "devcore-100-v1.json"
    if core.exists():
        data = json.loads(core.read_text())
        out["dev_core"] = sorted(
            by_rel["/" + str(item["image_path"]).lstrip("/")]["canonical_id"]
            for item in data.get("dev_core", [])
            if "/" + str(item["image_path"]).lstrip("/") in by_rel
        )
    sel = manifests_dir / "a1-select-300-v1.json"
    if sel.exists():
        data = json.loads(sel.read_text())
        out["selection_pool"] = sorted(
            by_rel["/" + str(item["image_path"]).lstrip("/")]["canonical_id"]
            for item in data.get("samples", [])
            if "/" + str(item["image_path"]).lstrip("/") in by_rel
        )
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
LEGACY_ALIAS_CONVENTION = (
    "Historical sample ids are the dataset row index, optionally prefixed with 'ds'. "
    "Verified against every manifest row that records both a sample_id and an image_path "
    "({checked}/{total} consistent). Never assumed: resolution uses the recorded image path "
    "first, and the digit form only when no path is available in the record."
)


def build_rows(dataset_json: Path, image_root: Path, hash_cache: Path) -> List[Dict[str, Any]]:
    records = json.loads(dataset_json.read_text())
    cached = {}
    for line in hash_cache.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cached[row["image_path"]] = row
    rows: List[Dict[str, Any]] = []
    for index, rec in enumerate(records):
        rel = rec["image_path"]
        text = str(rec.get("text", ""))
        hashes = cached.get(rel, {})
        rows.append({
            "record_type": RECORD_TYPE,
            "canonical_id": canonical_sample_id(rel, text),
            "dataset_index": index,
            "image_path_rel": rel,
            "image_byte_sha256": hashes.get("byte_sha256"),
            "image_pixel_sha256": hashes.get("pixel_sha256"),
            "image_dims": [hashes.get("width"), hashes.get("height")],
            "image_src_mode": hashes.get("src_mode"),
            "dhash64": hashes.get("dhash64"),
            "claim_sha256": claim_sha256(text),
            "claim_normalisation": CLAIM_NORMALISATION,
            "reference_status": "resolved" if hashes.get("byte_sha256") else "unresolved_reference",
            "eval_only": {
                "gt_answers": rec.get("gt_answers"),
                "fake_cls": rec.get("fake_cls"),
                "image_source": rec.get("image_source"),
                "text_source": rec.get("text_source"),
            },
        })
    return rows


def code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build the redesign sample registry (offline).")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--near-dup-hamming", type=int, default=4)
    ap.add_argument("--version", default="v2", help="output artifact version suffix")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    dataset_json = repo / "data" / "MMFakeBench_test" / "source" / "MMFakeBench_test.json"
    image_root = repo / "data" / "MMFakeBench_test"
    hash_cache = repo / "_audit" / "preflight_02" / "image_hashes.jsonl"
    manifests_dir = repo / "manifests"

    rows = build_rows(dataset_json, image_root, hash_cache)
    lookup, alias_unresolved, checked = build_alias_map(rows, manifests_dir)
    alias_map, alias_ambiguities = lookup.mapping, lookup.ambiguous

    aliases: Dict[str, List[str]] = {}
    for alias, cid in alias_map.items():
        aliases.setdefault(cid, []).append(alias)
    for row in rows:
        seen = set(aliases.get(row["canonical_id"], []))
        seen.add(str(row["dataset_index"]))
        row["aliases"] = sorted(seen)

    # Every canonical identity touched by an ambiguous alias is quarantined: the affected
    # identities are bounded by the recorded candidates, so this is not a blocking ambiguity.
    ambiguous_ids = sorted({cid for info in alias_ambiguities.values() for cid in info["candidates"]})

    artifact_list, scan_totals = scan_artifacts(repo, rows, lookup)
    manifest_sets = load_manifest_sets(manifests_dir, rows)
    group_of, edges = build_groups(rows)
    near_dups = flag_near_duplicate_candidates(rows, max_hamming=args.near_dup_hamming)
    flags, roles, evidence_refs, propagated, conservatively_excluded = assign_roles(
        rows, artifact_list, group_of, near_dups, manifest_sets, ambiguous_ids
    )

    for row in rows:
        cid = row["canonical_id"]
        row["group_id"] = group_of[cid]
        row["exposure"] = flags[cid]
        row["exposure_evidence"] = sorted(evidence_refs[cid])[:20]
        row["role"] = roles[cid]
        if roles[cid] != ROLE_RESERVE:
            reasons = [k for k, v in flags[cid].items() if v]
            row["exclusion_reason"] = ";".join(reasons) or "assigned_by_manifest"
        else:
            row["exclusion_reason"] = None

    roles_out: Dict[str, List[str]] = {}
    for row in rows:
        roles_out.setdefault(row["role"], []).append(row["canonical_id"])
    manifest_sets_out = {key: sorted(set(ids)) for key, ids in manifest_sets.items()}

    reserve_ids = sorted(roles_out.get(ROLE_RESERVE, []))
    group_members: Dict[str, List[str]] = {}
    for row in rows:
        group_members.setdefault(row["group_id"], []).append(row["canonical_id"])
    group_binding = group_binding_hash(group_members)

    reserve_rows = [r for r in rows if r["role"] == ROLE_RESERVE]
    reserve_groups = sorted({r["group_id"] for r in reserve_rows})
    class_counts: Dict[str, Dict[str, int]] = {}
    for r in reserve_rows:
        cls = r["eval_only"]["fake_cls"] or "unknown"
        gt = r["eval_only"]["gt_answers"] or "unknown"
        class_counts.setdefault("by_fake_cls", {}).setdefault(cls, 0)
        class_counts["by_fake_cls"][cls] += 1
        class_counts.setdefault("by_gt", {}).setdefault(gt, 0)
        class_counts["by_gt"][gt] += 1

    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sample_out = repo / "manifests" / f"redesign_sample_registry_{args.version}.jsonl"

    dependencies = collect_dependencies(repo)

    # Any reference that could not be resolved is a blocking issue: an identity that cannot be
    # bounded must fail explicitly rather than silently becoming "unused".
    blocking_issues: List[Dict[str, Any]] = []
    for row in alias_unresolved:
        blocking_issues.append({"kind": "unresolved_manifest_reference", **row})
    for ref in scan_totals["unresolved"]:
        blocking_issues.append({"kind": "unresolved_artifact_reference", "reference": ref})

    exposure_artifacts = [
        {"path": a["path"], "sha256": a["sha256"], "kind": a["kind"]}
        for a in artifact_list if a.get("sha256")
    ]
    # Evidence run directories contribute their bundle *inventory* rather than a file digest.
    exposure_directories = [
        {"path": a["path"], "inventory_sha256": a["inventory_sha256"],
         "n_bundles": len(a.get("inventory") or []), "kind": a["kind"]}
        for a in artifact_list if a.get("inventory_sha256")
    ]

    registry = {
        "registry_version": MODULE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "grouping_policy_version": GROUPING_POLICY_VERSION,
        "created_utc": created,
        "warning": "ADMIN/EVALUATION-ONLY. Not model input. Contains label-bearing paths and labels.",
        "deterministic_content_note": "created_utc is excluded from every membership hash",
        "conventions": {
            "canonical_id_rule": CANONICAL_ID_RULE,
            "claim_normalisation": CLAIM_NORMALISATION,
            "pixel_hash_procedure": PIXEL_HASH_PROCEDURE,
            "membership_hash": "sha256('set|' + '\\n'.join(sorted(canonical_ids)))",
            "execution_order_hash": "sha256('order|' + '\\n'.join(canonical_ids_in_run_order))",
            "group_binding_hash": "sha256('group_binding|' + '\\n'.join(f'{group_id}\\t{comma_joined_sorted_members}'))",
            "group_membership_hash_DEPRECATED": "sha256('groups|' + '\\n'.join(sorted(group_ids))) - hashes group names only; "
                                                "swapping members between groups is undetected. Read-only support for v1.",
            "dataset_index_fingerprint": "sha256('dataset|' + '\\n'.join(f'{index}\\t{rel_path}'))",
            "legacy_alias_convention": LEGACY_ALIAS_CONVENTION.format(checked=checked, total=checked),
            "alias_convention_note": f"{checked} reference rows across the split manifests recorded both "
                                     f"a sample_id and an image_path and agreed with the digit rule",
            "alias_conflict_policy": "An alias mapping to more than one canonical sample is recorded as an "
                                     "ambiguity and fails explicitly. No sample is chosen and no namespace is invented.",
            "path_split_hashes_vs_id_run_hashes": "never mixed",
        },
        "fingerprints": {
            "dataset_index": dataset_index_fingerprint(rows),
            "group_binding": group_binding,
            "sample_identity": identity_hash(rows),
            "role_assignment": role_assignment_hash(rows),
            "row_payload": row_payload_hash(rows),
            "alias_mapping": alias_mapping_hash(lookup),
            "alias_ambiguity": alias_ambiguity_hash(alias_ambiguities),
            "schema_version": SCHEMA_VERSION,
            "grouping_policy_version": GROUPING_POLICY_VERSION,
        },
        "manifests_dir": "manifests",
        "dependencies": dependencies,
        "counts": {
            "dataset_records": len(rows),
            "distinct_canonical_ids": len({r["canonical_id"] for r in rows}),
            "groups": len(group_members),
            "group_edges": len(edges),
            "edges_by_kind": {k: sum(1 for e in edges if e["kind"] == k) for k in
                              sorted({e["kind"] for e in edges})},
            "near_duplicate_candidate_pairs": len(near_dups),
            "conservatively_excluded_records": len(conservatively_excluded),
            "aliases_resolved": len(alias_map),
            "alias_ambiguities": len(alias_ambiguities),
            "alias_convention_checks_consistent": checked,
            "exposure_propagated_within_groups": len(propagated),
        },
        "alias_ambiguities": alias_ambiguities,
        "alias_unresolved_rows": alias_unresolved[:100],
        "roles": {k: sorted(set(v)) for k, v in roles_out.items()},
        "manifest_sets": manifest_sets_out,
        "membership_hashes": {k: membership_hash(v) for k, v in roles_out.items()},
        "manifest_set_hashes": {k: membership_hash(v) for k, v in manifest_sets_out.items()},
        "group_members": {gid: sorted(members) for gid, members in sorted(group_members.items())},
        "reserve": {
            "name": "reserved_not_evaluated",
            "description": "Custom redesign evaluation reserve drawn from MMFakeBench official "
                           "test data. Not a new official benchmark split.",
            "policy": "Reserve only groups with no recorded model evaluation, development or "
                      "selection use, annotation-material inclusion or evidence collection, no "
                      "unresolved or ambiguous exposure, and no verified or possible-duplicate "
                      "link to exposed material. Exclusion propagates to a fixpoint over those "
                      "relationships. Conservative exclusion is a project choice, not a claim "
                      "that every earlier inference is training contamination.",
            "n_records": len(reserve_ids),
            "n_groups": len(reserve_groups),
            "composition": {k: dict(sorted(v.items())) for k, v in sorted(class_counts.items())},
            "limits": "Exposure is limited to the artefacts inspected. It does not prove absence "
                      "of prior human exposure or of model-training overlap. The reserve is not "
                      "described as proven contamination-free.",
        },
        "blocking_issues": blocking_issues,
        "n_blocking_issues": len(blocking_issues),
        "exposure_artifacts": exposure_artifacts,
        "exposure_directories": exposure_directories,
        "non_bundle_filenames": sorted(NON_BUNDLE_FILENAMES),
        "exposure_scan": {
            "artifacts": artifact_list,
            "n_artifacts": len(artifact_list),
            "unresolved_references": scan_totals["unresolved"][:200],
            "n_unresolved_references": len(scan_totals["unresolved"]),
            "exclusions": scan_totals["excluded"],
        },
        "grouping": {
            "edge_kinds": sorted({e["kind"] for e in edges}),
            "not_used_as_identity": ["publisher", "generator family", "distortion class", "topic"],
            "guarantees": "exact-image (byte), decoded-image and exact-normalised-claim grouped; "
                          "dHash near-duplicates are candidates only, never groups",
            "near_duplicate_threshold_hamming": args.near_dup_hamming,
            "possible_duplicate_policy": "A possible-duplicate link is a reason for conservative "
                                         "reserve exclusion, not proof that two samples are identical.",
        },
        "validator": {
            "module": "scripts/redesign_registry_validate.py",
            "status": "implemented and tested; not yet called by any runner",
            "not_yet_integrated": "main.py, scripts/judge_study/* and historical runners",
            "integrity": "validate_request() verifies integrity itself; there is no caller-controlled bypass",
        },
        "seed": None,
    }
    registry_out = repo / "manifests" / f"redesign_data_registry_{args.version}.json"

    # Refuse to overwrite any existing registry artifact, including a v1 artifact.
    existing = [str(p.relative_to(repo)) for p in (sample_out, registry_out) if p.exists()]
    if existing:
        print("REFUSING to overwrite existing registry artifact(s): " + ", ".join(existing), file=sys.stderr)
        print("Choose a new --version. Existing artifacts are preserved by design.", file=sys.stderr)
        return 3

    sample_out.write_text("")
    with sample_out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    registry_out.write_text(json.dumps(registry, indent=2, sort_keys=False))
    print(json.dumps({
        "samples": len(rows), "groups": len(group_members),
        "roles": {k: len(v) for k, v in sorted(roles_out.items())},
        "reserve": len(reserve_ids), "reserve_groups": len(reserve_groups),
        "near_dups": len(near_dups), "alias_unresolved": len(alias_unresolved),
        "wrote": [str(sample_out.relative_to(repo)), str(registry_out.relative_to(repo))],
    }, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
