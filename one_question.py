#!/usr/bin/env python3
# scripts/run_one_question.py
"""
Run a single end-to-end pipeline for ONE question on ONE video:
  1) OS model answer (os_model.query_model_iterative_with_retry)
  2) Critic assessment (critic_model_os.critic_assess)
  3) Critic response re-evaluation (critic_response.query_model_iterative_with_retry)

Writes under <vid_dir>/<video_id>/:
  - <video_id>_answers.json
  - <video_id>_critic_assessment.json
  - <video_id>_re_evaluated.json
"""

import argparse
import asyncio
import json
import os
from typing import Dict, Any

from os_model import (
    Pipeline as OSPipeline,
    query_model_iterative_with_retry as os_query_with_retry,
)
from critic_model_os import (
    CriticPipeline,
    critic_assess,
)
from critic_response import (
    Pipeline as CriticRespPipeline,
    query_model_iterative_with_retry as critic_resp_query_with_retry,
    create_enhanced_prompt,
)

async def run_one_question(
    vid_dir: str,
    question: str,
    video_id: str = "01",
    uid: str = "q1",
    llm_model: str = "openai/gpt-oss-120b",
    vlm_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
) -> Dict[str, Any]:
    vid_path = f"{vid_dir}/{video_id}"
    os.makedirs(vid_path, exist_ok=True)

    # Required context
    global_summary_path = f"{vid_path}/captions/global_summary.txt"
    ces_logs_path = f"{vid_path}/captions/CES_logs.txt"
    if not os.path.exists(global_summary_path):
        raise FileNotFoundError(f"Missing global summary: {global_summary_path}")
    if not os.path.exists(ces_logs_path):
        raise FileNotFoundError(f"Missing CES logs: {ces_logs_path}")

    # 1) OS model answer
    answers_file = f"{vid_path}/{video_id}_answers.json"
    os_model = OSPipeline(llm_model, vlm_model)
    answer = await os_query_with_retry(os_model, question, uid, vid_path, answers_file)

    if isinstance(answer, str):
        # Already completed — load prior
        try:
            with open(answers_file, "r") as f:
                prev = json.load(f)
            answer = next((a for a in prev if a.get("uid") == uid), None)
        except Exception:
            answer = None

    if not answer or not isinstance(answer, dict):
        raise RuntimeError("OS model did not return a valid answer dict")

    # 2) Critic assessment
    critic = CriticPipeline(llm_model, vlm_model)
    assessment = await critic_assess(
        critic,
        question,
        uid,
        answer.get("answer"),
        answer.get("reasoning", ""),
        answer.get("evidence_frame_numbers", []),
        vid_dir,
        video_id,
    )

    critic_file = f"{vid_path}/{video_id}_critic_assessment.json"
    try:
        if os.path.exists(critic_file):
            with open(critic_file, "r") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing.append(assessment)
                data = existing
            else:
                data = [existing, assessment]
        else:
            data = [assessment]
        with open(critic_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write critic assessment: {e}")

    # 3) Critic response re-evaluation for this assessment
    enhanced_question = create_enhanced_prompt(assessment)
    critic_resp_model = CriticRespPipeline(llm_model, vlm_model)
    re_eval = await critic_resp_query_with_retry(
        critic_resp_model,
        enhanced_question,
        uid,
        vid_path,
    )

    return {
        "os_answer": answer,
        "critic_assessment": assessment,
        "critic_re_evaluation": re_eval,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Run OS -> Critic -> Critic-Response for one question on one video",
    )
    parser.add_argument("vid_dir", help="Video directory root, e.g., scripts/videos_two")
    parser.add_argument("video_id", help="Video id folder name, e.g., 00000048")
    parser.add_argument("uid", help="Unique id of the question, e.g., q1")
    parser.add_argument("question", help="Question text to answer")
    parser.add_argument("--llm_model", default="openai/gpt-oss-120b")
    parser.add_argument("--vlm_model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            run_one_question(
                args.vid_dir,
                args.video_id,
                args.uid,
                args.question,
                args.llm_model,
                args.vlm_model,
            )
        )
    finally:
        loop.close()

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()