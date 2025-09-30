import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.utils.dataset import (
    coerce_index,
    load_dataset_records,
    normalize_record_path,
    resolve_image_candidates,
)
from scripts.utils.json_utils import extract_json_array, extract_json_object
from scripts.utils.media import guess_mime_type, image_to_data_url
from scripts.utils.search import normalize_search_payload, normalize_search_results


class MediaUtilsTest(unittest.TestCase):
    def test_guess_mime_type_known_extension(self) -> None:
        self.assertEqual(guess_mime_type("sample.JPG"), "image/jpeg")
        self.assertEqual(guess_mime_type("sample.unknown", default="application/octet-stream"), "application/octet-stream")

    def test_image_to_data_url_encodes_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            handle.write(b"binary-data")
            temp_path = handle.name
        try:
            data_url = image_to_data_url(temp_path)
            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            encoded = data_url.split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), b"binary-data")
        finally:
            os.remove(temp_path)


class JsonUtilsTest(unittest.TestCase):
    def test_extract_json_object_with_padding(self) -> None:
        text = "Result: {\"value\": 7} -- end"
        parsed = extract_json_object(text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["value"], 7)

    def test_extract_json_object_fallback(self) -> None:
        parsed = extract_json_object("not json", fallback=lambda raw: {"raw": raw.strip()})
        self.assertEqual(parsed, {"raw": "not json"})

    def test_extract_json_array_variants(self) -> None:
        text = "prefix [\"a\", \"b\"] suffix"
        parsed = extract_json_array(text)
        self.assertEqual(parsed, ["a", "b"])
        fallback = extract_json_array("- item one\n- item two", fallback=lambda raw: raw.split())
        self.assertEqual(fallback, ["-", "item", "one", "-", "item", "two"])


class SearchUtilsTest(unittest.TestCase):
    def test_normalize_search_results(self) -> None:
        items = [
            {"href": "https://example.com", "title": "Title", "body": "Summary"},
            {"url": "", "title": "Missing"},
        ]
        normalized = normalize_search_results(items)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["url"], "https://example.com")
        self.assertEqual(normalized[0]["description"], "Summary")

    def test_normalize_search_payload_with_legacy(self) -> None:
        payload = {"provider": "legacy"}

        def legacy(data: dict) -> list:
            return [{"url": "https://legacy", "title": "Legacy", "description": "Info"}]

        normalized = normalize_search_payload(payload, legacy_extractor=legacy)
        self.assertEqual(normalized[0]["url"], "https://legacy")


class DatasetUtilsTest(unittest.TestCase):
    def test_load_dataset_records_handles_varied_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path_list = Path(tmp) / "list.json"
            path_list.write_text(json.dumps([{ "text": "a" }]), encoding="utf-8")
            records = load_dataset_records(path_list)
            self.assertEqual(len(records), 1)

            path_nested = Path(tmp) / "nested.json"
            path_nested.write_text(json.dumps({"data": [{"text": "b"}]}), encoding="utf-8")
            nested_records = load_dataset_records(path_nested)
            self.assertEqual(len(nested_records), 1)

    def test_normalize_record_path_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "images" / "item.jpg"
            sample.parent.mkdir()
            sample.write_text("content", encoding="utf-8")

            normalized = normalize_record_path("images/item.jpg", root)
            self.assertTrue(normalized.endswith("images/item.jpg"))

            candidates = resolve_image_candidates(root, "images/item.jpg")
            self.assertTrue(any(str(sample) in cand for cand in candidates))

    def test_coerce_index(self) -> None:
        self.assertEqual(coerce_index(5), 5)
        self.assertEqual(coerce_index("10"), 10)
        self.assertIsNone(coerce_index(True))
        self.assertIsNone(coerce_index(""))


if __name__ == "__main__":
    unittest.main()
