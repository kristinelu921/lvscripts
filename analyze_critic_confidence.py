#!/usr/bin/env python3
"""
Analyze critic confidence scores vs correctness
Generates statistics and visualizations
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_ground_truth(gt_path):
    """Load ground truth questions and answers"""
    with open(gt_path, 'r') as f:
        data = json.load(f)

    # Flatten to uid -> correct_choice mapping
    gt_map = {}
    for video_id, questions in data.items():
        for q in questions:
            gt_map[q['uid']] = q.get('correct_choice')

    return gt_map

def load_critic_assessments(videos_dir):
    """Load all critic assessment files"""
    assessments = []
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            if file.endswith('_critic_assessment.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        assessments.extend(data)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    return assessments

def analyze_confidence_correctness(assessments, ground_truth):
    """Analyze correlation between confidence and correctness"""

    # Categories for analysis
    results = {
        'correct_high': [],  # Correct + High confidence (>=80)
        'correct_medium': [],  # Correct + Medium confidence (50-79)
        'correct_low': [],  # Correct + Low confidence (<50)
        'wrong_high': [],  # Wrong + High confidence (>=80)
        'wrong_medium': [],  # Wrong + Medium confidence (50-79)
        'wrong_low': [],  # Wrong + Low confidence (<50)
        'no_gt': []  # No ground truth available
    }

    all_correct_conf = []
    all_wrong_conf = []

    for assessment in assessments:
        uid = assessment.get('uid')
        predicted = assessment.get('answer')
        confidence = assessment.get('confidence', -1)

        # Skip if no valid confidence
        if confidence < 0:
            continue

        # Get ground truth
        correct_choice = ground_truth.get(uid)

        if correct_choice is None:
            results['no_gt'].append(assessment)
            continue

        # Convert predicted answer to index (0-3)
        # Handle both string ("0", "A") and int (0) formats
        if isinstance(predicted, str):
            if predicted.isdigit():
                predicted_idx = int(predicted)
            elif len(predicted) == 1 and predicted.isalpha():
                # Convert A->0, B->1, etc.
                predicted_idx = ord(predicted.upper()) - ord('A')
            else:
                print(f"Warning: Cannot parse answer '{predicted}' for {uid}")
                continue
        elif isinstance(predicted, int):
            predicted_idx = predicted
        else:
            print(f"Warning: Unexpected answer type for {uid}: {type(predicted)}")
            continue

        # Check correctness
        is_correct = (predicted_idx == correct_choice)

        # Categorize
        if is_correct:
            all_correct_conf.append(confidence)
            if confidence >= 80:
                results['correct_high'].append(assessment)
            elif confidence >= 50:
                results['correct_medium'].append(assessment)
            else:
                results['correct_low'].append(assessment)
        else:
            all_wrong_conf.append(confidence)
            if confidence >= 80:
                results['wrong_high'].append(assessment)
            elif confidence >= 50:
                results['wrong_medium'].append(assessment)
            else:
                results['wrong_low'].append(assessment)

    return results, all_correct_conf, all_wrong_conf

def print_statistics(results, all_correct_conf, all_wrong_conf):
    """Print detailed statistics"""

    total = sum(len(v) for k, v in results.items() if k != 'no_gt')

    print("="*80)
    print("CRITIC CONFIDENCE ANALYSIS")
    print("="*80)
    print(f"\nTotal assessments analyzed: {total}")
    print(f"Assessments without ground truth: {len(results['no_gt'])}")

    # Overall accuracy
    total_correct = len(results['correct_high']) + len(results['correct_medium']) + len(results['correct_low'])
    accuracy = total_correct / total * 100 if total > 0 else 0
    print(f"\nOverall Accuracy: {total_correct}/{total} ({accuracy:.1f}%)")

    # Confidence statistics
    print("\n" + "-"*80)
    print("CONFIDENCE STATISTICS")
    print("-"*80)

    if all_correct_conf:
        print(f"\nCorrect Answers:")
        print(f"  Count: {len(all_correct_conf)}")
        print(f"  Mean confidence: {np.mean(all_correct_conf):.1f}%")
        print(f"  Median confidence: {np.median(all_correct_conf):.1f}%")
        print(f"  Std dev: {np.std(all_correct_conf):.1f}%")

    if all_wrong_conf:
        print(f"\nWrong Answers:")
        print(f"  Count: {len(all_wrong_conf)}")
        print(f"  Mean confidence: {np.mean(all_wrong_conf):.1f}%")
        print(f"  Median confidence: {np.median(all_wrong_conf):.1f}%")
        print(f"  Std dev: {np.std(all_wrong_conf):.1f}%")

    # Calibration analysis
    print("\n" + "-"*80)
    print("CALIBRATION ANALYSIS (Confidence vs Correctness)")
    print("-"*80)

    print(f"\nHigh Confidence (≥80%):")
    print(f"  Correct: {len(results['correct_high'])}")
    print(f"  Wrong: {len(results['wrong_high'])}")
    total_high = len(results['correct_high']) + len(results['wrong_high'])
    if total_high > 0:
        acc_high = len(results['correct_high']) / total_high * 100
        print(f"  Accuracy: {acc_high:.1f}%")

    print(f"\nMedium Confidence (50-79%):")
    print(f"  Correct: {len(results['correct_medium'])}")
    print(f"  Wrong: {len(results['wrong_medium'])}")
    total_med = len(results['correct_medium']) + len(results['wrong_medium'])
    if total_med > 0:
        acc_med = len(results['correct_medium']) / total_med * 100
        print(f"  Accuracy: {acc_med:.1f}%")

    print(f"\nLow Confidence (<50%):")
    print(f"  Correct: {len(results['correct_low'])}")
    print(f"  Wrong: {len(results['wrong_low'])}")
    total_low = len(results['correct_low']) + len(results['wrong_low'])
    if total_low > 0:
        acc_low = len(results['correct_low']) / total_low * 100
        print(f"  Accuracy: {acc_low:.1f}%")

    # Ideal calibration check
    print("\n" + "-"*80)
    print("CALIBRATION QUALITY")
    print("-"*80)
    print("\nIdeal calibration: High confidence → High accuracy")
    print("                   Low confidence → Low accuracy")

    if total_high > 0 and total_low > 0:
        if acc_high > acc_low:
            print(f"\n✅ GOOD: High conf accuracy ({acc_high:.1f}%) > Low conf accuracy ({acc_low:.1f}%)")
        else:
            print(f"\n❌ POOR: High conf accuracy ({acc_high:.1f}%) ≤ Low conf accuracy ({acc_low:.1f}%)")

def create_visualizations(results, all_correct_conf, all_wrong_conf, output_dir='./'):
    """Create visualization plots"""

    # Create output directory if needed
    Path(output_dir).mkdir(exist_ok=True)

    # Figure 1: Confidence distribution by correctness
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    bins = range(0, 105, 5)
    axes[0].hist(all_correct_conf, bins=bins, alpha=0.6, label='Correct', color='green', edgecolor='black')
    axes[0].hist(all_wrong_conf, bins=bins, alpha=0.6, label='Wrong', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score (%)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data_to_plot = [all_correct_conf, all_wrong_conf]
    bp = axes[1].boxplot(data_to_plot, labels=['Correct', 'Wrong'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Confidence Score (%)', fontsize=12)
    axes[1].set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved: {output_dir}/confidence_distribution.png")

    # Figure 2: Calibration curve
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate accuracy for confidence bins
    bins = [0, 20, 40, 60, 80, 100]
    bin_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    bin_accuracies = []
    bin_counts = []

    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]

        # Count correct and total in this bin
        correct_in_bin = sum(1 for c in all_correct_conf if low <= c < high)
        wrong_in_bin = sum(1 for c in all_wrong_conf if low <= c < high)
        total_in_bin = correct_in_bin + wrong_in_bin

        if total_in_bin > 0:
            accuracy = correct_in_bin / total_in_bin * 100
            bin_accuracies.append(accuracy)
            bin_counts.append(total_in_bin)
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)

    # Plot calibration
    x_pos = np.arange(len(bin_labels))
    bars = ax.bar(x_pos, bin_accuracies, color=['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850'],
                   edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'n={count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Ideal calibration line (diagonal)
    ideal_x = np.array([10, 30, 50, 70, 90])  # Midpoints of bins
    ax.plot(x_pos, ideal_x, 'k--', linewidth=2, label='Ideal Calibration', alpha=0.7)

    ax.set_xlabel('Confidence Range (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Curve: Confidence vs Actual Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_curve.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_dir}/calibration_curve.png")

    # Figure 3: Confusion matrix style
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['High\n(≥80%)', 'Medium\n(50-79%)', 'Low\n(<50%)']
    correct_counts = [len(results['correct_high']), len(results['correct_medium']), len(results['correct_low'])]
    wrong_counts = [len(results['wrong_high']), len(results['wrong_medium']), len(results['wrong_low'])]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, correct_counts, width, label='Correct', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, wrong_counts, width, label='Wrong', color='#e74c3c', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax.set_title('Correctness by Confidence Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/correctness_by_confidence.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_dir}/correctness_by_confidence.png")

    plt.close('all')

def main():
    # Paths
    parser = argparse.ArgumentParser(description='Analyze critic confidence')
    parser.add_argument('--videos_dir', type=str, default='/mnt/ssd/data/longvideobench/videos_processed', help='Path to videos directory')
    parser.add_argument('--gt_path', type=str, default='/mnt/ssd/data/longvideobench/downloaded_videos_questions.json', help='Path to ground truth questions')
    parser.add_argument('--output_dir', type=str, default='/mnt/ssd/data/critic_analysis', help='Path to output directory')
    args = parser.parse_args()

    videos_dir = args.videos_dir
    gt_path = args.gt_path
    output_dir = args.output_dir

    print("Loading data...")
    ground_truth = load_ground_truth(gt_path)
    print(f"✓ Loaded ground truth for {len(ground_truth)} questions")

    assessments = load_critic_assessments(videos_dir)
    print(f"✓ Loaded {len(assessments)} critic assessments")

    print("\nAnalyzing confidence vs correctness...")
    results, all_correct_conf, all_wrong_conf = analyze_confidence_correctness(assessments, ground_truth)

    print_statistics(results, all_correct_conf, all_wrong_conf)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(results, all_correct_conf, all_wrong_conf, output_dir)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")

if __name__ == "__main__":
    main()