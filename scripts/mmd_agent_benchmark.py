#!/usr/bin/env python3
"""
MMD Agent benchmark implementation for comparison with MIRAGE.

Implements the 3-stage sequential reasoning from the MMD-Agent paper:
1. Textual veracity check (with Wikipedia knowledge retrieval)
2. Visual veracity check (fact-conflicting content)
3. Cross-modal consistency verification

Usage:
    python mmd_agent_benchmark.py \
        --max-samples 1000 \
        --dataset-root data/MMFakeBench_test \
        --save-jsonl results/mmd_agent_run.jsonl

Outputs JSONL compatible with scripts/evaluate.py for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataloader import MMFakeBenchDataset
from scripts.llm_loader import LLMModelLoader
from scripts.utils.media import image_to_data_url
from scripts.utils.json_utils import extract_json_object

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# MMD Agent prompts - EXACT from the paper
# Source: eval/prompt_template/MMD_Agent/

TEXTUAL_VERACITY_PROMPT = """Given a news caption, news caption is: {headline}
Determine if there is credible objective evidence that SUPPORTS or REFUTES the news caption. Please follow the instructions bellow:
Thought 1: You need to find the key entity noun in the news caption. The key entity noun could be person or object or location or event, etc. The key entity noun should be included in the text caption. 
Action 1: Search [key entity noun].
Observation: {wiki_knowledge}
Thought 2: According to Observation and other credible objective evidence, please analysis there is any objective fact that SUPPORTS or REFUTES the news caption, or if there is NOT ENOUGH INFORMATION. Analysis is: [Analysis].
Action 2: Draw the conclusion based on the analysis in the thought 2: if there is any credible objective evidence refuting the news caption, please answer in the form: 'Finish[TEXT REFUTES].'. If no, please answer in the form: 'Finish[TEXT SUPPORTS].'.
The answer is:"""


VISUAL_VERACITY_PROMPT = """According to the given news image, determine if the news image goes againes the objective facts? Please follow the instructions bellow:
Thought 1: Please describe the content in the news image that goes against the objective fact.
Observation: [Fact-conflicting Description]
Action 1: Draw the conclusion based on the observation: if there is any credible objective fact refuting the news image, please answer in the form: 'Finish[IMAGE REFUTES].'. If no, please answer in the form: 'Finish[IMAGE SUPPORTS].'.
The answer is:"""


CROSS_MODAL_CONSISTENCY_PROMPT = """Given a multimodal misinformation, it contains both news caption and news image. News caption is: {headline}
Determine if the news caption matches the content news image. You should answer in the following forms: 'Finish[MATCH].' or 'Finish[MISMATCH].'. Please follow the instructions bellow:
IMAGE DESCRIPTION: {image_description}
Draw the conclusion: Based on the [IMAGE DESCRIPTION] of the news image, does the news caption match the content of news image? If yes, please answer in the form: 'Finish[MATCH].'. If no, please answer in the form: 'Finish[MISMATCH].'.
The answer is:"""


# Image description prompt for Stage 3
IMAGE_DESCRIPTION_PROMPT = """Describe the content of this image in detail. Focus on:
- Main subjects and objects
- Actions or events depicted
- Setting and context
- Any notable details

Provide a clear, factual description:"""


def search_wikipedia(query: str) -> str:
    """Simplified Wikipedia search - returns placeholder for now.
    
    In production, this would use Wikipedia API or web search.
    For benchmark purposes, we'll use web search via the existing infrastructure.
    """
    try:
        from scripts.search_provider import web_search
        
        # Search Wikipedia specifically
        search_query = f"site:wikipedia.org {query}"
        results = web_search(search_query, provider=os.getenv("SEARCH_PROVIDER", "brave"))
        
        # Extract top result descriptions
        search_results = results.get("results", [])
        if search_results:
            summaries = []
            for result in search_results[:3]:  # Top 3 results
                title = result.get("title", "")
                desc = result.get("description", "")
                if desc:
                    summaries.append(f"{title}: {desc}")
            
            if summaries:
                return "\n".join(summaries)
    except Exception as e:
        print(f"  Wikipedia search error: {e}")
    
    return "No external knowledge available."


def stage_1_textual_veracity(
    headline: str,
    loader: LLMModelLoader,
) -> Dict[str, Any]:
    """Stage 1: Check textual veracity with Wikipedia knowledge."""
    
    # Extract key entity for Wikipedia search
    entity_prompt = f"Extract the main subject/entity from this headline for a Wikipedia search (respond with just the entity name): {headline}"
    
    try:
        model = loader.get_model()
        entity_resp = model.invoke([
            {"role": "system", "content": "You extract key entities for Wikipedia searches."},
            {"role": "user", "content": entity_prompt}
        ])
        entity = getattr(entity_resp, "content", "").strip()
        
        # Search Wikipedia
        wiki_knowledge = search_wikipedia(entity)
        
        # Check textual veracity using EXACT paper prompt
        prompt = TEXTUAL_VERACITY_PROMPT.format(
            headline=headline,
            wiki_knowledge=wiki_knowledge
        )
        
        resp = model.invoke([
            {"role": "system", "content": "You are a fact-checking assistant following MMD-Agent methodology."},
            {"role": "user", "content": prompt}
        ])
        
        text = getattr(resp, "content", resp)
        raw_text = text if isinstance(text, str) else str(text)
        
        # Parse Finish[TEXT REFUTES] or Finish[TEXT SUPPORTS] format
        verdict = "INSUFFICIENT"
        if "Finish[TEXT REFUTES]" in raw_text:
            verdict = "REFUTES"
        elif "Finish[TEXT SUPPORTS]" in raw_text:
            verdict = "SUPPORTS"
        
        return {
            "verdict": verdict,
            "confidence": 0.8 if verdict != "INSUFFICIENT" else 0.5,
            "explanation": raw_text.strip(),
            "wiki_knowledge": wiki_knowledge,
            "raw": raw_text
        }
    
    except Exception as e:
        return {
            "verdict": "INSUFFICIENT",
            "confidence": 0.5,
            "explanation": f"Error: {str(e)}",
            "wiki_knowledge": "",
            "raw": ""
        }


def stage_2_visual_veracity(
    image_path: str,
    headline: str,
    loader: LLMModelLoader,
) -> Dict[str, Any]:
    """Stage 2: Check for fact-conflicting visual content."""
    
    try:
        data_url = image_to_data_url(image_path)
        model = loader.get_model()
        
        # Use EXACT paper prompt
        prompt = VISUAL_VERACITY_PROMPT
        
        messages = [
            {"role": "system", "content": "You are an image forensics expert following MMD-Agent methodology."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url}
                ]
            }
        ]
        
        resp = model.invoke(messages)
        text = getattr(resp, "content", resp)
        raw_text = text if isinstance(text, str) else str(text)
        
        # Parse Finish[IMAGE REFUTES] or Finish[IMAGE SUPPORTS] format
        verdict = "SUPPORTS"
        if "Finish[IMAGE REFUTES]" in raw_text:
            verdict = "REFUTES"
        elif "Finish[IMAGE SUPPORTS]" in raw_text:
            verdict = "SUPPORTS"
        
        # Extract fact-conflicting description if present
        artifacts = []
        if "Observation:" in raw_text:
            parts = raw_text.split("Observation:")
            if len(parts) > 1:
                observation = parts[1].split("Action")[0].strip()
                artifacts = [observation]
        
        return {
            "verdict": verdict,
            "confidence": 0.8 if verdict == "REFUTES" else 0.7,
            "explanation": raw_text.strip(),
            "artifacts": artifacts,
            "raw": raw_text
        }
    
    except Exception as e:
        return {
            "verdict": "SUPPORTS",
            "confidence": 0.5,
            "explanation": f"Error: {str(e)}",
            "artifacts": [],
            "raw": ""
        }


def stage_3_cross_modal_consistency(
    image_path: str,
    headline: str,
    loader: LLMModelLoader,
) -> Dict[str, Any]:
    """Stage 3: Check cross-modal consistency."""
    
    try:
        data_url = image_to_data_url(image_path)
        model = loader.get_model()
        
        # Step 1: Get image description (as per paper's prompt structure)
        desc_messages = [
            {"role": "system", "content": "You describe images accurately and objectively."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                    {"type": "image_url", "image_url": data_url}
                ]
            }
        ]
        
        desc_resp = model.invoke(desc_messages)
        image_description = getattr(desc_resp, "content", "").strip()
        
        # Step 2: Check cross-modal consistency using EXACT paper prompt
        prompt = CROSS_MODAL_CONSISTENCY_PROMPT.format(
            headline=headline,
            image_description=image_description
        )
        
        messages = [
            {"role": "system", "content": "You verify image-text alignment following MMD-Agent methodology."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url}
                ]
            }
        ]
        
        resp = model.invoke(messages)
        text = getattr(resp, "content", resp)
        raw_text = text if isinstance(text, str) else str(text)
        
        # Parse Finish[MATCH] or Finish[MISMATCH] format
        verdict = "PARTIAL"
        if "Finish[MATCH]" in raw_text:
            verdict = "CONSISTENT"
        elif "Finish[MISMATCH]" in raw_text:
            verdict = "MISMATCH"
        
        return {
            "verdict": verdict,
            "confidence": 0.8 if verdict != "PARTIAL" else 0.5,
            "explanation": raw_text.strip(),
            "image_description": image_description,
            "raw": raw_text
        }
    
    except Exception as e:
        return {
            "verdict": "PARTIAL",
            "confidence": 0.5,
            "explanation": f"Error: {str(e)}",
            "image_description": "",
            "raw": ""
        }


def mmd_agent_classify(
    stage1: Dict[str, Any],
    stage2: Dict[str, Any],
    stage3: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MMD Agent classification logic (sequential):
    
    1. If text REFUTES → textual_veracity_distortion
    2. Else if image REFUTES → visual_veracity_distortion  
    3. Else if cross-modal MISMATCH → mismatch
    4. Else → original (authentic)
    """
    
    # Stage 1: Text check
    if stage1.get("verdict") == "REFUTES":
        return {
            "label": "Misinformation",
            "fake_cls": "textual_veracity_distortion",
            "confidence": stage1.get("confidence", 0.5),
            "rationale": f"Textual veracity check: {stage1.get('explanation', '')}",
            "key_factors": ["Text contradicts external knowledge"]
        }
    
    # Stage 2: Visual check
    if stage2.get("verdict") == "REFUTES":
        return {
            "label": "Misinformation",
            "fake_cls": "visual_veracity_distortion",
            "confidence": stage2.get("confidence", 0.5),
            "rationale": f"Visual veracity check: {stage2.get('explanation', '')}",
            "key_factors": stage2.get("artifacts", ["Visual artifacts detected"])
        }
    
    # Stage 3: Cross-modal consistency
    if stage3.get("verdict") == "MISMATCH":
        return {
            "label": "Misinformation",
            "fake_cls": "mismatch",
            "confidence": stage3.get("confidence", 0.5),
            "rationale": f"Cross-modal consistency: {stage3.get('explanation', '')}",
            "key_factors": ["Image-text mismatch"]
        }
    
    # Default: Authentic
    avg_conf = (
        stage1.get("confidence", 0.5) + 
        stage2.get("confidence", 0.5) + 
        stage3.get("confidence", 0.5)
    ) / 3.0
    
    return {
        "label": "Not Misinformation",
        "fake_cls": "original",
        "confidence": avg_conf,
        "rationale": "All checks passed: text supported, image authentic, cross-modal consistent",
        "key_factors": ["No contradictions detected"]
    }


def _scan_existing_outputs(output_path: Path) -> Dict[str, Any]:
    """Read an existing JSONL and return the processed count plus the last record."""

    processed_count = 0
    last_record: Optional[Dict[str, Any]] = None

    with output_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Existing output contains invalid JSON on line {line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Existing output line {line_no} is not a JSON object.")

            expected_iteration = processed_count
            expected_sample_index = processed_count + 1

            iteration_index = record.get("iteration_index")
            if iteration_index is not None and int(iteration_index) != expected_iteration:
                raise ValueError(
                    "Existing output is not sequential: "
                    f"line {line_no} has iteration_index={iteration_index}, "
                    f"expected {expected_iteration}."
                )

            sample_index = record.get("sample_index")
            if sample_index is not None and int(sample_index) != expected_sample_index:
                raise ValueError(
                    "Existing output is not sequential: "
                    f"line {line_no} has sample_index={sample_index}, "
                    f"expected {expected_sample_index}."
                )

            processed_count += 1
            last_record = record

    return {
        "processed_count": processed_count,
        "last_record": last_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MMD Agent benchmark")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/MMFakeBench_test",
        help="Dataset root directory"
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="",
        help="Optional dataset JSON path"
    )
    parser.add_argument(
        "--save-jsonl",
        type=str,
        default="",
        help="Output JSONL path (default: results/mmd_agent_YYYYMMDD-HHMMSS.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("ALIGN_MODEL", "gpt-4o-mini"),
        help="LLM model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL and continue from the existing row count"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing JSONL output file instead of refusing or resuming"
    )

    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together.")

    # Setup paths
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    # Default output path
    if not args.save_jsonl:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.save_jsonl = f"results/mmd_agent_{timestamp}.jsonl"

    output_path = Path(args.save_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resume_info = {
        "processed_count": 0,
        "last_record": None,
    }
    if output_path.exists():
        if args.resume:
            resume_info = _scan_existing_outputs(output_path)
        elif output_path.stat().st_size > 0 and not args.overwrite:
            print(
                f"Error: Output already exists and is non-empty: {output_path}\n"
                "Use --resume to continue or --overwrite to replace it."
            )
            return

    # Resolve dataset JSON
    dataset_json = None
    if args.dataset_json:
        dataset_json = args.dataset_json
    else:
        # Try to find it automatically
        dataset_name = dataset_root.name
        candidates = [
            dataset_root / "source" / f"{dataset_name}.json",
            dataset_root / f"{dataset_name}.json"
        ]
        for candidate in candidates:
            if candidate.exists():
                dataset_json = str(candidate)
                break
    
    if not dataset_json:
        print(f"Error: Could not find dataset JSON in {dataset_root}")
        return
    
    print(f"MMD Agent Benchmark")
    print(f"Dataset: {dataset_root}")
    print(f"JSON: {dataset_json}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.max_samples}")
    print(f"Output: {output_path}")
    
    # Initialize LLM
    loader = LLMModelLoader({
        "provider": os.getenv("ALIGN_PROVIDER", "openai"),
        "model": args.model,
        "temperature": args.temperature,
    })
    
    # Load dataset with same stratification as MIRAGE
    ds = MMFakeBenchDataset(
        json_path=dataset_json,
        image_root=str(dataset_root),
        balanced=False,
        seed=42,  # Same seed as MIRAGE
        skip_missing=True,
        return_image=False,
        stratify_by=["fake_cls"],  # Same stratification
        verbose=True
    )

    total_samples = min(args.max_samples, len(ds))
    existing_count = int(resume_info["processed_count"])
    start_index = existing_count

    if existing_count > 0:
        last_record = resume_info["last_record"]
        if start_index > total_samples:
            print(
                "Error: Existing output has more rows than the requested sample count "
                f"({existing_count} > {total_samples}). Increase --max-samples or use --overwrite."
            )
            return
        if last_record is not None and start_index <= len(ds):
            expected_sample = ds[start_index - 1]
            expected_image_path = expected_sample.get("image_path")
            expected_headline = expected_sample.get("text", "")
            if last_record.get("image_path") != expected_image_path:
                print(
                    "Error: Existing output does not match the dataset order for resume.\n"
                    f"Last saved image_path: {last_record.get('image_path')}\n"
                    f"Expected image_path: {expected_image_path}"
                )
                return
            if last_record.get("headline") != expected_headline:
                print(
                    "Error: Existing output headline does not match the dataset order for resume."
                )
                return

    if args.resume and existing_count:
        print(
            f"\nResuming from sample {existing_count + 1} of {total_samples} "
            f"using existing output: {output_path}"
        )
    else:
        print(f"\nProcessing {total_samples} samples...")

    start_time = time.time()
    processed_this_run = max(0, total_samples - start_index)

    # Process samples
    mode = "a" if args.resume and existing_count > 0 else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for i in range(start_index, total_samples):
            sample = ds[i]
            img_path = sample.get("image_path")
            headline = sample.get("text", "")
            
            print(f"\n[{i+1}/{total_samples}] {Path(img_path).name}")
            
            # Track usage before this sample
            usage_before = dict(getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0}))
            
            # Run MMD Agent 3-stage pipeline
            print("  Stage 1: Textual veracity...")
            stage1 = stage_1_textual_veracity(headline, loader)
            print(f"    Verdict: {stage1.get('verdict')} (conf: {stage1.get('confidence'):.2f})")
            
            print("  Stage 2: Visual veracity...")
            stage2 = stage_2_visual_veracity(img_path, headline, loader)
            print(f"    Verdict: {stage2.get('verdict')} (conf: {stage2.get('confidence'):.2f})")
            
            print("  Stage 3: Cross-modal consistency...")
            stage3 = stage_3_cross_modal_consistency(img_path, headline, loader)
            print(f"    Verdict: {stage3.get('verdict')} (conf: {stage3.get('confidence'):.2f})")
            
            # Final classification
            judgement = mmd_agent_classify(stage1, stage2, stage3)
            print(f"  Final: {judgement.get('label')} ({judgement.get('fake_cls')})")
            
            # Calculate token usage for this sample
            usage_after = getattr(loader, "usage_total", {"prompt": 0, "completion": 0, "total": 0})
            sample_usage = {
                "prompt": max(0, int(usage_after.get("prompt", 0)) - int(usage_before.get("prompt", 0))),
                "completion": max(0, int(usage_after.get("completion", 0)) - int(usage_before.get("completion", 0))),
                "total": max(0, int(usage_after.get("total", 0)) - int(usage_before.get("total", 0)))
            }
            
            # Build output object in MIRAGE format
            output_obj = {
                "image_path": img_path,
                "headline": headline,
                "sample_index": i + 1,
                "dataset_order_index": sample.get("dataset_index"),
                "iteration_index": i,
                "provider": os.getenv("ALIGN_PROVIDER", "openai"),
                "model": args.model,
                "settings": {
                    "temperature": args.temperature,
                    "method": "mmd_agent"
                },
                "mmd_agent": {
                    "stage1_textual": stage1,
                    "stage2_visual": stage2,
                    "stage3_cross_modal": stage3
                },
                "judgement": judgement,
                "token_usage": sample_usage,
                "sample_details": {
                    "dataset_index": sample.get("dataset_index"),
                    "gt_answers": sample.get("gt_answers"),
                    "fake_cls": sample.get("fake_cls"),
                    "text_source": sample.get("text_source"),
                    "image_source": sample.get("image_source")
                }
            }
            
            # Write to JSONL
            f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
            f.flush()

    # Summary
    duration = time.time() - start_time
    usage_total = getattr(loader, "usage_total", {})

    print(f"\n=== Run Complete ===")
    print(f"Processed this run: {processed_this_run} samples")
    print(f"Output rows total: {total_samples}")
    if processed_this_run > 0:
        print(f"Duration: {duration:.1f}s ({duration/processed_this_run:.2f}s per sample)")
    else:
        print(f"Duration: {duration:.1f}s")
    print(f"Token usage: prompt={usage_total.get('prompt', 0)} "
          f"completion={usage_total.get('completion', 0)} "
          f"total={usage_total.get('total', 0)}")
    print(f"Output: {output_path}")
    
    # Run evaluation
    print(f"\n=== Running Evaluation ===")
    try:
        from scripts.evaluate import evaluate
        
        metrics_path = output_path.with_suffix(".metrics.json")
        csv_path = output_path.with_suffix(".metrics.csv")
        
        evaluate(
            output_path,
            Path(dataset_json),
            dataset_root,
            save_report=metrics_path,
            save_csv=csv_path
        )
        
        print(f"Metrics saved: {metrics_path}")
        print(f"CSV saved: {csv_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
