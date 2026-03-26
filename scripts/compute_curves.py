"""
Compute PR-AUC, AUROC, Balanced Accuracy, and generate plots.
Matched to MERIT/MMD-Agent JSONL output format.

Usage (from your Lightning.ai studio root):

    python compute_curves.py \
        --inputs \
            results/run-1k-val-full-system.jsonl \
            results/run-1k-val-ablation-no-visual-02-10-25.jsonl \
            results/run-val-no-relevancy.jsonl \
            results/run-1k-val-ablation-no-rag-02-10-25.jsonl \
            results/run-1k-val-ablation-judge-only.jsonl \
            results/mmd_agent_baseline.jsonl \
        --labels \
            "Full MERIT" \
            "No Visual" \
            "No Relevancy" \
            "No Claim Verification" \
            "Judge Only" \
            "MMD-Agent (GPT-4o-mini)" \
        --output-dir figures

Requires: pip install scikit-learn matplotlib
"""

import json
import argparse
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(filepath):
    """
    Load JSONL results file.
    
    Fields used:
      - judgement.label: "Misinformation" or "Not Misinformation"
      - judgement.confidence: float 0-1
      - sample_details.gt_answers: "Fake" or "Real"
      - sample_details.fake_cls: category string
    """
    y_true = []
    y_scores = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            
            # Ground truth: 1 = Misinformation, 0 = Not Misinformation
            gt_answer = row.get('sample_details', {}).get('gt_answers', '')
            if gt_answer == 'Fake':
                gt = 1
            elif gt_answer == 'Real':
                gt = 0
            else:
                fake_cls = row.get('sample_details', {}).get('fake_cls', '')
                if fake_cls in ('mismatch', 'textual_veracity_distortion', 'visual_veracity_distortion'):
                    gt = 1
                elif fake_cls == 'original':
                    gt = 0
                else:
                    print(f"WARNING: Unknown gt='{gt_answer}', fake_cls='{fake_cls}', skipping")
                    continue
            
            y_true.append(gt)
            
            # Model confidence -> P(Misinformation)
            judgement = row.get('judgement', {})
            judge_label = judgement.get('label', '')
            judge_conf = judgement.get('confidence', 0.5)
            
            if judge_label == 'Not Misinformation':
                score = 1.0 - judge_conf
            elif judge_label == 'Misinformation':
                score = judge_conf
            else:
                score = 0.5
            
            y_scores.append(score)
    
    return np.array(y_true), np.array(y_scores)


def compute_metrics(y_true, y_scores, label="Model"):
    y_pred = (y_scores >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    try:
        if len(np.unique(y_scores)) <= 1:
            auroc = 0.5
        else:
            auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5
    
    try:
        pr_auc = average_precision_score(y_true, y_scores)
    except ValueError:
        pr_auc = 0.0
    
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  AUROC:             {auroc:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print(f"  Samples:           {len(y_true)}")
    print(f"  Class dist:        {sum(y_true)} misinfo, {len(y_true)-sum(y_true)} authentic")
    print(f"  Unique scores:     {len(np.unique(y_scores))}")
    
    return {
        'label': label,
        'bal_acc': bal_acc,
        'auroc': auroc,
        'pr_auc': pr_auc,
        'y_true': y_true,
        'y_scores': y_scores
    }


def plot_roc_curves(results_list, save_path='roc_curves.pdf'):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    colors = ['#2b4570', '#3B71CA', '#14A085', '#D4A017', '#DC3545', '#6c757d']
    
    for i, r in enumerate(results_list):
        color = colors[i % len(colors)]
        try:
            fpr, tpr, _ = roc_curve(r['y_true'], r['y_scores'])
            ax.plot(fpr, tpr, label=f"{r['label']} ({r['auroc']:.3f})",
                    linewidth=2, color=color)
        except ValueError:
            print(f"WARNING: Could not compute ROC for {r['label']}")
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to {save_path}")


def plot_pr_curves(results_list, save_path='pr_curves.pdf'):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    colors = ['#2b4570', '#3B71CA', '#14A085', '#D4A017', '#DC3545', '#6c757d']
    
    for i, r in enumerate(results_list):
        color = colors[i % len(colors)]
        try:
            precision, recall, _ = precision_recall_curve(r['y_true'], r['y_scores'])
            ax.plot(recall, precision, label=f"{r['label']} ({r['pr_auc']:.3f})",
                    linewidth=2, color=color)
        except ValueError:
            print(f"WARNING: Could not compute PR for {r['label']}")
    
    if len(results_list) > 0:
        prevalence = sum(results_list[0]['y_true']) / len(results_list[0]['y_true'])
        ax.axhline(y=prevalence, color='k', linestyle='--', alpha=0.3,
                   label=f'Random ({prevalence:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PR curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()
    
    assert len(args.inputs) == len(args.labels)
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    for filepath, label in zip(args.inputs, args.labels):
        print(f"\nLoading {filepath}...")
        y_true, y_scores = load_results(filepath)
        result = compute_metrics(y_true, y_scores, label)
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*75}")
    print(f"  {'Config':<30} {'Bal.Acc':>10} {'AUROC':>10} {'PR-AUC':>10}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['label']:<30} {r['bal_acc']*100:>9.2f}% {r['auroc']:>10.4f} {r['pr_auc']:>10.4f}")
    
    # Plots
    plot_roc_curves(all_results, os.path.join(args.output_dir, 'roc_curves.pdf'))
    plot_pr_curves(all_results, os.path.join(args.output_dir, 'pr_curves.pdf'))
    
    # LaTeX
    print(f"\n  LATEX TABLE ROWS:")
    for r in all_results:
        print(f"    {r['label']} & {r['bal_acc']*100:.2f} & {r['auroc']:.4f} & {r['pr_auc']:.4f} \\\\")


if __name__ == '__main__':
    main()