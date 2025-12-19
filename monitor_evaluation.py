#!/usr/bin/env python3
"""
Monitor evaluation progress
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta


def monitor_evaluation(output_dir: str, log_file: str = None):
    """Monitor evaluation progress and display stats"""

    print(f"Monitoring evaluation in: {output_dir}")
    print(f"Press Ctrl+C to stop monitoring\n")

    last_checkpoint = 0

    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')

            print("="*80)
            print(f"EVALUATION PROGRESS MONITOR")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

            # Check for checkpoints
            checkpoints = list(Path(output_dir).glob('results_checkpoint_*.json'))
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1]

                with open(latest_checkpoint, 'r') as f:
                    results = json.load(f)

                checkpoint_num = int(latest_checkpoint.stem.split('_')[-1])

                if checkpoint_num > last_checkpoint:
                    last_checkpoint = checkpoint_num
                    print(f"\n✓ Checkpoint {checkpoint_num} reached")

                # Calculate stats
                os_correct = sum(1 for r in results if r.get('os_correct'))
                final_correct = sum(1 for r in results if r.get('final_correct'))
                errors = sum(1 for r in results if 'error' in r)

                total_time = sum(sum(r.get('timing', {}).values()) for r in results)
                avg_time = total_time / len(results) if results else 0

                print(f"\nQuestions processed: {len(results)}")
                print(f"OS Model correct: {os_correct} ({os_correct/len(results)*100:.1f}%)")
                print(f"Final correct: {final_correct} ({final_correct/len(results)*100:.1f}%)")
                print(f"Errors: {errors}")
                print(f"Avg time per question: {avg_time:.1f}s")

                if len(results) > 0:
                    remaining = 300 - len(results)
                    eta_seconds = remaining * avg_time
                    eta = timedelta(seconds=int(eta_seconds))
                    print(f"ETA: ~{eta}")

                # Show recent questions
                print(f"\nRecent questions:")
                for r in results[-5:]:
                    status = "✓" if r.get('final_correct') else "✗"
                    print(f"  {status} [{r['video_id']}/{r['question_id']}] {r.get('final_prediction', 'N/A')}")

            else:
                print("\nWaiting for first checkpoint...")

            # Check traces
            traces_dir = Path(output_dir) / 'traces'
            if traces_dir.exists():
                trace_count = len(list(traces_dir.glob('*/*_trace.json')))
                print(f"\nTraces saved: {trace_count}")

            # Show log tail if available
            if log_file and os.path.exists(log_file):
                print(f"\nRecent log output:")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")

            # Check for completion
            results_file = Path(output_dir) / 'evaluation_results.json'
            if results_file.exists():
                print("\n" + "="*80)
                print("EVALUATION COMPLETE!")
                print("="*80)

                with open(results_file, 'r') as f:
                    final_results = json.load(f)

                metrics = final_results['metrics']
                print(f"\nFinal Results:")
                print(f"  Total: {metrics['summary']['total_questions']} questions")
                print(f"  OS Accuracy: {metrics['accuracy']['os_accuracy']:.2f}%")
                print(f"  Final Accuracy: {metrics['accuracy']['final_accuracy']:.2f}%")
                print(f"  Improvement: {metrics['accuracy']['improvement']:+.2f}%")
                print(f"  Total Time: {metrics['summary']['total_time_seconds']/60:.1f} minutes")
                print(f"\nResults: {results_file}")
                print(f"Traces: {traces_dir}")
                break

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Monitor evaluation progress")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/resource/evaluation_results_300q',
        help='Output directory to monitor'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='/resource/evaluation_log.txt',
        help='Log file to tail'
    )

    args = parser.parse_args()
    monitor_evaluation(args.output_dir, args.log_file)
