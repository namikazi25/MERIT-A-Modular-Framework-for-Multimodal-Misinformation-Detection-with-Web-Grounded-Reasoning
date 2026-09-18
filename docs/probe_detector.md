# PROBE-DINOv2 preparation and comparison gate

PROBE-DINOv2 is the intended dedicated generation detector for the next named detector comparison. GLM-5.3-Flash remains the reasoning backbone; Jev remains the evidence-checking component. PROBE does not replace either role. The historical ViT detector and its results retain their original identity.

Status on 2026-09-18: preprocessing and score aggregation are implemented and tested with synthetic fixtures. Checkpoint loading and real PROBE inference are not implemented or validated. `load_probe_detector()` fails explicitly; there is no fallback to ViT or GLM. No PROBE predictions, accuracy, improvement or calibration result exists for MERIT.

## Verified upstream contract

The inspected [official repository](https://github.com/Amamiya-C/PROBE-AIGI-Detection/tree/b145f7130004c02725e9b3703954a3329ebf56de) is pinned at `b145f7130004c02725e9b3703954a3329ebf56de`. The public [checkpoint repository](https://modelscope.cn/models/shuinishaojiu/PROBE-AIGI-Detection) lists `DINOv2_best_model_step_34999.pth`, 1,217,679,947 bytes. Its advertised SHA256 and file revision are recorded in `config/probe_detector.json`; the file has not been downloaded or independently hashed.

[Evaluation code](https://github.com/Amamiya-C/PROBE-AIGI-Detection/blob/b145f7130004c02725e9b3703954a3329ebf56de/Detector/evaluate_dino.py) uses RGB, ImageNet normalization and a nonoverlapping 336-pixel grid. Undersized dimensions repeat pixels without resizing. Partial right/bottom edges are dropped. Default augmentations are disabled. The output is sigmoid of the mean patch logits, with a strictly-greater-than-0.5 positive threshold. `scripts/redesign/probe_detector.py` independently implements these numeric operations. It rejects invalid/oversized inputs instead of assigning the upstream zero-patch fallback score of 0.5. The local bounds are 16 million pixels and 64 patches; these bounds must be disclosed in any comparison.

The [classifier code](https://github.com/Amamiya-C/PROBE-AIGI-Detection/blob/b145f7130004c02725e9b3703954a3329ebf56de/Detector/model/dino_classifier.py) applies a linear head to the backbone CLS token. Its active backbone path points to an author-local directory; a comment names `facebook/dinov2-with-registers-large`. That comment does not establish an exact released configuration/revision. The upstream evaluator also derives evaluation labels from folder names; the MERIT preparation path accepts verified image bytes and does not use those labels or paths.

## Blockers and next implementation

The pinned repository has no licence file, GitHub reports no licence, and the ModelScope licence fields are empty. This is an unresolved permission status, not a claim that use is prohibited. The supervisor document requires an official, suitably licensed checkpoint, so checkpoint download/execution is blocked pending a documented licence or author permission. The exact backbone configuration is also unverified.

After those gates clear:

1. Record source and checkpoint use/redistribution terms and the exact backbone configuration hash. Recheck the existing session/download allowance; do not reset it on resume.
2. Fetch only the DINO checkpoint, verify its advertised hash, and load locally with `weights_only=True`, strict key matching and no remote code. Instantiate the verified backbone configuration locally; do not silently download an alternate backbone.
3. Run a synthetic CPU smoke test, measure memory/latency, and compare the prepared tensors and outputs with the official implementation under the pinned settings. Current synthetic tests establish intended preprocessing behavior, not end-to-end upstream numerical equivalence.
4. Admit only the approved development core; check image hashes before each call. Keep generation scores and detector identity in local records and pass only typed computed signals to the judge.
5. Preregister `A1_PROBE_DINOv2_v1` versus the GLM VLM generation signal using the same five development images, frozen evidence and unchanged final reasoning model/prompt. Keep the old `A1` ViT results separate. Do not run a named PROBE condition without its real checkpoint.

Generation-origin gold labels are required for detector discrimination/calibration claims. Misinformation labels alone are not generation-origin labels. Downstream misinformation outcomes can be compared separately, with abstentions and false positives visible. A generation detector cannot establish local manipulation, source provenance or the truth of a claim.

An unsent author enquiry, if needed: “Could you clarify the licence and permitted research use/redistribution of the PROBE code and DINOv2 checkpoint, and identify the exact backbone configuration/revision used by `DINOv2_best_model_step_34999.pth`? The current classifier references a local backbone directory.” No message has been sent.
