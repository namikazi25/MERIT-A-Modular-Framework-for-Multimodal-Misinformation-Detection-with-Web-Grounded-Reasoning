"""Preflight 01 - offline tests: benchmark metadata must not reach model requests.

Boundaries respected by this module:
  * OFFLINE. A socket guard replaces ``socket.socket.connect`` and fails on any
    connection attempt, so an accidental live call is a test failure, not a warning.
  * NO CREDENTIALS. Every provider config passes an explicit dummy ``api_key``.
    ``.env`` is never read and no environment secret is required.
  * MOCKED CLIENTS. ``openai.OpenAI`` is replaced by a recorder, so the request is
    captured immediately before the SDK would send it. Nothing leaves the process.

Run with:  python -m unittest tests.test_input_separation -v
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import openai

from scripts import ai_judge
from scripts.llm_loader import LLMModelLoader
from scripts.question_generator import generate_investigative_questions
from scripts.relevancy_checker import assess_image_headline_relevancy
from scripts.visual_veracity_checker import assess_image_visual_veracity

# ``scripts.judge_study.judge_only`` calls ``load_dotenv()`` at import time. Stub the
# module first so importing the judge-study entry point cannot read credentials.
if "dotenv" not in sys.modules:  # pragma: no cover - import hygiene, not behaviour
    _dotenv_stub = types.ModuleType("dotenv")
    _dotenv_stub.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = _dotenv_stub

from scripts.judge_study.judges import judge_j1  # noqa: E402



# A 1x1 PNG. Small enough to keep test output readable, valid enough for Pillow-free use.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# Path fragments that carry benchmark answers or construction provenance.
_LABEL_BEARING_TOKENS = (
    "/fake/",
    "/real/",
    "MMFakeBench",
    "chatgpt_match",
    "gossipcop_midjourney",
    "fakeddit_photo_edit",
    "bbc_test",
    "ai-generated image",
)

def _string_leaves(obj):
    """Yield every string inside a nested request structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _string_leaves(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _string_leaves(value)


# Keys that are evaluation-only and must never be rendered into a request body.
_EVAL_ONLY_KEYS = ("gt_answers", "fake_cls", "image_source", "text_source", "dataset_index", "sample_details")


class NetworkGuard:
    """Fail loudly if any test attempts a socket connection."""

    def __enter__(self):
        self._original = socket.socket.connect

        def _blocked(_self, *args, **kwargs):
            raise AssertionError("OFFLINE TEST VIOLATION: network access was attempted")

        socket.socket.connect = _blocked
        return self

    def __exit__(self, *exc):
        socket.socket.connect = self._original
        return False


class RecordingClient:
    """Stands in for the provider SDK client and records outbound request bodies."""

    def __init__(self, reply: str | None = None):
        self.calls: list[dict] = []
        self.reply = reply or json.dumps(
            {"label": "Not Misinformation", "confidence": 0.5, "rationale": "stub", "key_factors": []}
        )

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._create)))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply), logprobs=None)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    @property
    def last_body(self) -> str:
        """Raw serialised request, for whole-body scans."""
        return json.dumps(self.calls[-1], ensure_ascii=False, default=str)

    @property
    def last_text(self) -> str:
        """Decoded string leaves of the request, so escaped JSON is readable."""
        return "\n".join(_string_leaves(self.calls[-1]))


class ModelInputSeparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._guard = NetworkGuard()
        self._guard.__enter__()
        self._real_openai = openai.OpenAI
        self.recorder = RecordingClient()
        openai.OpenAI = self.recorder  # type: ignore[assignment]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self) -> None:
        openai.OpenAI = self._real_openai  # type: ignore[assignment]
        self._guard.__exit__(None, None, None)

    # ---------- helpers ----------

    def _loader(self):
        # api_key is supplied explicitly, so no environment or .env access is needed.
        return LLMModelLoader(
            {"provider": "openai", "model": "gpt-4o", "api_key": "dummy-key", "temperature": 0.0}
        )

    def _image(self, name: str) -> str:
        path = Path(self.tmp.name) / name
        path.write_bytes(_PNG_1X1)
        return str(path)

    def _final_obj(self, *, label_bearing: bool) -> dict:
        """Two final_obj payloads that differ ONLY in evaluation-only metadata."""
        if label_bearing:
            image_path = str(
                Path(self.tmp.name)
                / "MMFakeBench_test"
                / "fake"
                / "chatgpt_match_test_500"
                / "chatgpt_match_test_62.png"
            )
            details = {
                "gt_answers": "Fake",
                "fake_cls": "textual_veracity_distortion",
                "image_source": "AI-generated Image",
                "text_source": "VisualNews",
                "dataset_index": 5062,
            }
        else:
            image_path = str(
                Path(self.tmp.name) / "inputs" / "sample_0001.png"
            )
            details = {
                "gt_answers": "True",
                "fake_cls": "original",
                "image_source": "VisualNews",
                "text_source": "VisualNews",
                "dataset_index": 1,
            }

        return {
            "image_path": image_path,
            "headline": "City council approves the annual transport budget",
            "relevancy": {"aligned": True, "confidence": 0.9, "explanation": "image matches the subject"},
            "visual_veracity": {"ai_generated": False, "confidence": 0.2, "explanation": "no artifacts"},
            "best_qa_per_chain": [
                {
                    "question": "Did the council approve the budget?",
                    "answer": "Yes, reported by the local paper.",
                    "confidence": 0.8,
                    "citations": [{"url": "https://example.org/news/budget", "title": "Budget approved"}],
                }
            ],
            "sample_details": details,
            "sample_index": 1,
            "dataset_order_index": details["dataset_index"],
        }

    # ---------- boundary tests ----------

    def test_judge_request_is_invariant_to_evaluation_only_metadata(self) -> None:
        """Changing only ground truth, class and the class-bearing path must not change the request."""
        loader = self._loader()
        ai_judge.judge_from_structured(self._final_obj(label_bearing=True), loader)
        ai_judge.judge_from_structured(self._final_obj(label_bearing=False), loader)

        self.assertEqual(len(self.recorder.calls), 2, "expected exactly two recorded judge calls")
        first, second = self.recorder.calls
        self.assertEqual(
            json.dumps(first, sort_keys=True, default=str),
            json.dumps(second, sort_keys=True, default=str),
            "model-facing judge request changed when only evaluation-only metadata changed",
        )

    def test_judge_request_contains_no_label_bearing_tokens(self) -> None:
        loader = self._loader()
        ai_judge.judge_from_structured(self._final_obj(label_bearing=True), loader)
        body = self.recorder.last_text.lower()
        for token in _LABEL_BEARING_TOKENS:
            self.assertNotIn(token, body, f"label-bearing token {token!r} reached the model request")

    def test_judge_request_excludes_evaluation_only_fields(self) -> None:
        """Scan the serialised body, not just the top level, for private fields."""
        loader = self._loader()
        ai_judge.judge_from_structured(self._final_obj(label_bearing=True), loader)
        body = self.recorder.last_text
        for key in _EVAL_ONLY_KEYS:
            self.assertNotIn(key, body, f"evaluation-only field {key!r} reached the model request")

    def test_legitimate_content_is_preserved(self) -> None:
        """Sanitising must not delete real claim wording or real computed findings."""
        loader = self._loader()
        final_obj = self._final_obj(label_bearing=True)
        final_obj["headline"] = "Claim check: the photo is real, the caption is fake"
        final_obj["visual_veracity"] = {
            "ai_generated": True,
            "confidence": 0.91,
            "explanation": "the model flagged this image as AI-generated",
        }
        ai_judge.judge_from_structured(final_obj, loader)
        text = self.recorder.last_text

        self.assertIn("the photo is real, the caption is fake", text)
        self.assertIn('"ai_generated": true', text.lower())
        self.assertIn("the model flagged this image as AI-generated", text)
        self.assertIn("did the council approve the budget?", text.lower())

    def test_historical_judge_still_drops_citation_urls(self) -> None:
        """Documents the known evidence-retention gap: this task must not change it."""
        loader = self._loader()
        ai_judge.judge_from_structured(self._final_obj(label_bearing=True), loader)
        text = self.recorder.last_text

        self.assertNotIn("https://example.org/news/budget", text)
        self.assertIn("citations_count", text)

    def test_image_bytes_reach_the_request(self) -> None:
        """An opaque identifier must not replace the image itself."""
        loader = self._loader()
        expected = "data:image/png;base64," + base64.b64encode(_PNG_1X1).decode("ascii")

        assess_image_headline_relevancy(self._image("chatgpt_match_test_62.png"), "A headline", loader)
        assess_image_visual_veracity(self._image("bbc_test_0.png"), loader)
        generate_investigative_questions(self._image("real_guardian_test_1.png"), "A headline", loader, chains=1)

        self.assertEqual(len(self.recorder.calls), 3)
        for call in self.recorder.calls:
            parts = call["messages"] if "messages" in call else []
            self.assertTrue(
                any(
                    part.get("type") == "image_url"
                    and part.get("image_url", {}).get("url") == expected
                    for message in parts
                    for part in (message.get("content") or [])
                    if isinstance(part, dict)
                ),
                "the intended image bytes did not reach the request",
            )

    def test_image_stages_do_not_leak_the_file_name(self) -> None:
        loader = self._loader()
        assess_image_headline_relevancy(self._image("chatgpt_match_test_62.png"), "A headline", loader)
        assess_image_visual_veracity(self._image("bbc_test_0.png"), loader)
        generate_investigative_questions(self._image("real_guardian_test_1.png"), "A headline", loader, chains=1)

        for call in self.recorder.calls:
            text_parts = [
                part.get("text", "")
                for message in call["messages"]
                for part in (message.get("content") or [])
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            blob = " ".join(text_parts).lower()
            for token in ("chatgpt_match", "bbc_test", "/fake/", "/real/"):
                self.assertNotIn(token, blob, f"{token!r} reached the prompt text of an image stage")


    def test_judge_request_ignores_the_bundle_image_path(self) -> None:
        """The judge-study entry point (J1) mirrors main.py and must be sanitised too."""
        loader = self._loader()
        bundle = {
            "sample_id": str(Path(self.tmp.name) / "MMFakeBench_test" / "fake" / "DMM_test_500" / "dmm_7.png"),
            "image_path": str(Path(self.tmp.name) / "MMFakeBench_test" / "fake" / "DMM_test_500" / "dmm_7.png"),
            "claim": "City council approves the annual transport budget",
            "relevancy": {"aligned": True, "confidence": 0.9, "explanation": "matches"},
            "visual_veracity": {"ai_generated": False, "confidence": 0.2, "explanation": "clean"},
            "answers": [
                {
                    "question": "Did the council approve the budget?",
                    "answer": "Yes.",
                    "confidence": 0.8,
                    "citations": [{"url": "https://example.org/news/budget"}],
                }
            ],
            "documents": {"Did the council approve the budget?": []},
        }
        judge_j1(bundle, loader)
        text = self.recorder.last_text.lower()
        for token in _LABEL_BEARING_TOKENS:
            self.assertNotIn(token, text, f"judge-study entry point leaked {token!r}")

    def test_both_entry_points_render_the_same_shape(self) -> None:
        """main.py and the judge-study reconstruction must produce the same payload keys."""
        from scripts.judge_study.judge_only import reconstruct_judge_input

        main_style = self._final_obj(label_bearing=True)
        bundle_style = reconstruct_judge_input(
            {
                "image_path": main_style["image_path"],
                "claim": main_style["headline"],
                "relevancy": main_style["relevancy"],
                "visual_veracity": main_style["visual_veracity"],
                "answers": main_style["best_qa_per_chain"],
            }
        )
        self.assertEqual(
            list(ai_judge._model_facing_payload(main_style)),
            list(ai_judge._model_facing_payload(bundle_style)),
        )
        self.assertNotIn("image_path", ai_judge._model_facing_payload(bundle_style))


class OfflineGuardTest(unittest.TestCase):
    def test_network_guard_blocks_connections(self) -> None:
        with NetworkGuard():
            with self.assertRaises(AssertionError):
                socket.socket().connect(("example.org", 443))


if __name__ == "__main__":
    unittest.main()
