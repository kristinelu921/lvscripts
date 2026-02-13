#!/usr/bin/env python3
"""
Gemini API Client
A simple Python interface for Google's Gemini API
"""

import os
import json
import argparse
import glob
import google.generativeai as genai
from typing import Optional, List, Dict, Any


class GeminiClient:
    """Client for interacting with Google's Gemini API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client

        Args:
            api_key: Google AI API key. If not provided, will use GEMINI_API_KEY env variable
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in GEMINI_API_KEY environment variable")

        genai.configure(api_key=self.api_key)
        self.model = None

    def set_model(self, model_name: str = "gemini-1.5-pro"):
        """
        Set the Gemini model to use

        Args:
            model_name: Name of the Gemini model (default: gemini-1.5-pro - most capable)
                       Other options: gemini-2.0-flash-thinking-exp, gemini-exp-1206
        """
        self.model = genai.GenerativeModel(model_name)
        return self

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: The text prompt
            **kwargs: Additional parameters for generation

        Returns:
            Generated text response
        """
        if not self.model:
            self.set_model()

        response = self.model.generate_content(prompt, **kwargs)
        return response.text

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """
        Generate text from a prompt and image

        Args:
            prompt: The text prompt
            image_path: Path to the image file
            **kwargs: Additional parameters for generation

        Returns:
            Generated text response
        """
        if not self.model:
            self.set_model()

        from PIL import Image
        img = Image.open(image_path)

        response = self.model.generate_content([prompt, img], **kwargs)
        return response.text

    def generate_with_images(self, prompt: str, image_paths: List[str], **kwargs) -> str:
        """
        Generate text from a prompt and multiple images

        Args:
            prompt: The text prompt
            image_paths: List of paths to image files
            **kwargs: Additional parameters for generation

        Returns:
            Generated text response
        """
        if not self.model:
            self.set_model()

        from PIL import Image

        # Build content list with prompt and all images
        content = [prompt]
        for img_path in image_paths:
            img = Image.open(img_path)
            content.append(img)
            #TODO: also add the timestamp of the frame to the content

        response = self.model.generate_content(content, **kwargs)
        return response.text

    def generate_with_video_and_images(self, prompt: str, video_path: str, image_paths: List[str] = None, video=True, **kwargs) -> str:
        """
        Generate text from a prompt with video and optional images

        Args:
            prompt: The text prompt
            video_path: Path to the video file
            image_paths: Optional list of paths to image files (evidence frames)
            **kwargs: Additional parameters for generation

        Returns:
            Generated text response
        """
        if not self.model:
            self.set_model()

        import time
        import subprocess
        import tempfile
        from PIL import Image

        
        # Wait for video to be processed
        if video:

            # Preprocess video: Reduces resolution to 480p and sets frame rate to 1 FPS
            print(f"    Preprocessing video: {os.path.basename(video_path)}...")
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video_path = temp_video.name
            temp_video.close()
            try:
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-vf', 'scale=-2:480,fps=1',
                    '-c:v', 'libx264', '-crf', '28',
                    '-c:a', 'aac', '-b:a', '32k',
                    '-y', temp_video_path
                ], check=True, capture_output=True, text=True)
                print(f"    Video preprocessed successfully")
                upload_video_path = temp_video_path
            except subprocess.CalledProcessError as e:
                print(f"    Warning: Video preprocessing failed, using original video")
                print(f"    Error: {e.stderr}")
                upload_video_path = video_path
                temp_video_path = None
            # Upload video file
            print(f"    Uploading video: {os.path.basename(video_path)}...")
            video_file = genai.upload_file(path=upload_video_path)

            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_path}")

            print(f"    Video uploaded successfully")   

        # Build content list with video, prompt, and optional images
        if video:
            content = [video_file, prompt]
        else:
            content = [prompt]

        if image_paths:
            for img_path in image_paths:
                timestamp = img_path.split('/')[-1].split('.')[0]
                img = Image.open(img_path)
                content.append(img)
                content.append(f"Timestamp: {timestamp}")

        response = self.model.generate_content(content, **kwargs)

        # Clean up uploaded file
        if video:
            genai.delete_file(video_file.name)

        # Clean up temporary preprocessed video
            if video and temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        return response.text

    def list_models(self) -> List[str]:
        """
        List available Gemini models

        Returns:
            List of model names
        """
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]


def get_neighbor_frames(video_frames_dir: str, frame_numbers: List[int], neighbor_count: int = 10) -> List[str]:
    """
    Get frames and their neighbors from pre-extracted frames folder

    Args:
        video_frames_dir: Path to the video frames directory (e.g., /path/to/videos_processed/video_id/frames)
        frame_numbers: List of frame numbers to get (with neighbors)
        neighbor_count: Number of frames to get before and after each frame (default: 10)

    Returns:
        List of paths to frame images
    """
    if not os.path.exists(video_frames_dir):
        print(f"Warning: Frames directory not found: {video_frames_dir}")
        return []

    collected_frames = []

    for frame_num in frame_numbers: # frame_num is like 1149
        # Get neighbors around this frame
        try:
            frame_num = int(frame_num.split('_')[-1].split('.')[0])
            for offset in range(-neighbor_count, neighbor_count + 1):
                target_frame = frame_num + offset
                if target_frame < 1 or target_frame > 10000:  # Frames start at 1 and go to 10000
                    continue

                frame_filename = f"frame_{target_frame:04d}.jpg"
                frame_path = os.path.join(video_frames_dir, frame_filename)

                if os.path.exists(frame_path):
                    collected_frames.append(frame_path)
        except Exception as e:
            print(f"  Warning: Error getting neighbor frames for frame {frame_num}: {e}")
            continue

    return sorted(set(collected_frames))


def main():
    """Process video traces and analyze with Gemini API"""

    parser = argparse.ArgumentParser(description='Analyze video reasoning traces with Gemini')
    parser.add_argument('trace_file', type=str, help='Path to trace JSON file')
    parser.add_argument('--video-folder', type=str, default='/mnt/ssd/data/lvbench/videos_processed')
    parser.add_argument('--videos', type=str, default='/mnt/ssd/data/lvbench/videos')
    parser.add_argument('--model', type=str, default='gemini-3-flash-preview', help='Gemini model to use')
    parser.add_argument('--output_dir', type=str, default='gemini_analysis', help='Output directory for analysis')
    parser.add_argument('--critic', action='store_true', help='Use critic analysis mode')
    parser.add_argument('--judge', action='store_true', help='Use judge analysis mode')
    args = parser.parse_args()

    with open('env.json', 'r') as f:
        env = json.load(f)
    gemini_api_key = env['gemini_api_key']

    # Initialize Gemini client
    client = GeminiClient(gemini_api_key)
    client.set_model(args.model)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trace file
    with open(args.trace_file, 'r') as f:
        trace = json.load(f)

    print(f"Processing {len(trace)} items from trace file...")

    # Process each item in trace
    if args.critic:
        for idx, item in enumerate(trace):
            video_id = item['video_id']
            uid = item.get('uid', '')
            print(f"processing index {idx} of {len(trace)}: uid number {uid} from video {video_id}")
            question = item['question']
            candidates = item['candidates']
            pre_critic_answer = item['pre_critic_answer']
            critic_answer = item['critic_answer']
            critic_reasoning = item['critic_reasoning']
            correct_choice_idx = item['correct_choice_idx']
            correct_answer = item['correct_answer']
            evidence_frames = item.get('evidence_frames', [])
            confidence = item['confidence']
            possible_errors = item['possible_errors']
            suggestion = item['suggestion']
            criteria_results = item['criteria_results']
            global_summary_path = os.path.join(args.video_folder, video_id, "captions", "global_summary.txt")
            global_context = ""
            if os.path.exists(global_summary_path):
                with open(global_summary_path, 'r') as f:
                    global_context = f.read().strip()
                    global_context = f"\n\nGlobal video context/summary:\n{global_context}\n"
            else:
                print(f"  Warning: No global summary found at {global_summary_path}")
            
            output_file = os.path.join(args.output_dir, f"critic_analysis{video_id}.txt")

            # Check if we've already completed the critic analysis for this UID
            if os.path.exists(output_file):
                curr_output = open(output_file, "r").read()
                if f"UID: {uid}" in curr_output:
                    print(f"  Already completed critic analysis for UID: {uid}")
                    continue
                else:
                    print(f"  Starting new critic analysis for UID: {uid}")
    
    

            video_frames_dir = f"{args.video_folder}/{video_id}/frames"
            frames_paths = []
            for frame in evidence_frames:
                    # frame is like "frames/frame_1149.jpg"
                    new_frames = get_neighbor_frames(video_frames_dir, [frame], 10)
                    for frame_path in new_frames:
                        # get_neighbor_frames already returns full paths
                        if os.path.exists(frame_path):
                            frames_paths.append(frame_path)
                        else:
                            print(f"  Warning: Frame not found: {frame_path}")

            frame_list = "\n".join([f"  - {os.path.basename(f)}" for f in frames_paths])
            if frame_list:
                frame_info = f"\n\nEvidence frames (attached as images):\n{frame_list}. The reasoner also saw the 15-20 frames surrounding each frame."
            else:
                frame_info = "\n\nNo evidence frames available."

            prompt = f"""This is a reasoning trace for a video question analysis:{global_context}

            Question: {question}
            Answer choices: {candidates}

            We had a critic model evaluate the frames, reasoning trace, and answers generated by an LLM/VLM pipeline. The LLM passed criteria that we used to determine if the scene was correct, and the frames were enough. The LLM also passed its first answer as {pre_critic_answer}. The critic evaluated the criteria and outputted a passing percentage as follows:The criteria results were: {criteria_results}. The confidence was {confidence}%. The possible errors were: {possible_errors}. The suggestion for improvement was: {suggestion}. The critic suggested an answer that was: {critic_answer} (indices: 0, 1, 2, 3, 4 for each candidate).

            Reasoning trace: {critic_reasoning}

            Given that this was the correct choice and answer, I want you to analyze if / why the critic was wrong, and what could have been done differently to make the critic pass. 
            Correct choice index: {correct_choice_idx}
            Correct answer: {correct_answer}
            {frame_info}

            Please analyze why the reasoning trace was wrong by looking at the provided evidence frames:
            1. Was the scene selection correct?
            2. Are there enough frames to make the decision?
            3. Was the reasoning blatantly wrong?
            4. What should have been done differently?
            5. Do the frames actually support the correct answer?

            Was the critic right in its analysis of the scene selection? Was it right in its analysis of the frames? Was it right in its analysis of the reasoning? Was it right in its analysis of the answer?

            I also want a suggestion for how to improve the reasoning trace and the critic model. What else could a pipeline, given an LLM + caption search + frames to query a VLM with, contain so that this question could have reasoned correctly? What caused the pipeline to go down the path to the wrong answer that it did, instead of the right answer? If you hadn't known the correct answer, would you have done what the pipeline did?

            What could the critic have done differently to correctly analyze whether or not the scene selection was correct? The criteria were passed to it by hte LLM. Is thre a better critic setup that might work instead? You can be creative in ideas, but be rational and reasonable. What instructions could I give the critic to better analyze the scene selection?
            """

            video_path = None #don't need a video path for this one
            try:
                response = client.generate_with_video_and_images(prompt, video_path, frames_paths, video=False)
            except Exception as e:
                print(f"  ✗ Error analyzing {video_id}: {e}")
                continue

            # Save response
            output_file = os.path.join(args.output_dir, f"critic_analysis{video_id}.txt")
            with open(output_file, "a") as f:
                f.write(f"\n{'='*80}\n\n")
                f.write(f"Critic analysis for UID: {uid}\n")
                f.write(f"Video ID: {video_id}\n")
                f.write(f"{'='*80}\n\n")

                f.write(f"\n\nUID: {uid}\n")
                f.write(f"\n\nVideo ID: {video_id}\n")
                f.write(response)
            
            print(f"Critic analysis saved to {output_file}")

    elif args.judge:
        pass
    else:
        for idx, item in enumerate(trace):
            video_id = item['video_id']
            uid = item.get('uid', '')
            question = item['question']
            candidates = item['candidates']
            pre_critic_answer = item['pre_critic_answer']
            pre_critic_reasoning = item['pre_critic_reasoning']
            correct_choice_idx = item['correct_choice_idx']
            correct_answer = item['correct_answer']
            evidence_frames = item.get('evidence_frames', [])

            # Find the actual video file
            video_file = os.path.join(args.videos, f"{video_id}.mp4")
            if os.path.exists(video_file):
                actual_video_file = video_file
            else:
                print(f"  ✗ Warning: No video found at {video_file}")
                actual_video_file = None

            # Build full paths to frame images
            frame_paths = []
            if evidence_frames:
                video_frames_dir = f"{args.video_folder}/{video_id}/frames"
                for frame in evidence_frames:
                    # frame is like "frames/frame_1149.jpg"
                    new_frames = get_neighbor_frames(video_frames_dir, [frame], 10)
                    for frame_path in new_frames:
                        # get_neighbor_frames already returns full paths
                        if os.path.exists(frame_path):
                            frame_paths.append(frame_path)
                        else:
                            print(f"  Warning: Frame not found: {frame_path}")

            # Load global context/summary
            global_summary_path = os.path.join(args.video_folder, video_id, "captions", "global_summary.txt")
            global_context = ""
            if os.path.exists(global_summary_path):
                with open(global_summary_path, 'r') as f:
                    global_context = f.read().strip()
                    global_context = f"\n\nGlobal video context/summary:\n{global_context}\n"
            else:
                print(f"  Warning: No global summary found at {global_summary_path}")

            # Build frame list for prompt
            frame_list = "\n".join([f"  - {os.path.basename(f)}" for f in frame_paths])
            if frame_list:
                frame_info = f"\n\nEvidence frames (attached as images):\n{frame_list}. The reasoner also saw the 15-20 frames surrounding each frame."
            else:
                frame_info = "\n\nNo evidence frames available."

            # Build analysis prompt
            prompt = f"""This is a reasoning trace for a video question analysis:{global_context}

        Question: {question}
        Answer choices: {candidates}
        Given answer by model: {pre_critic_answer} (indices: 0, 1, 2, 3, 4 for each candidate)
        Reasoning trace: {pre_critic_reasoning}
        Correct choice index: {correct_choice_idx}
        Correct answer: {correct_answer}
        {frame_info}

        Please analyze why the reasoning trace was wrong by looking at the provided evidence frames:
        1. Was the scene selection correct?
        2. Are there enough frames to make the decision?
        3. Was the reasoning blatantly wrong?
        4. What should have been done differently?
        5. Do the frames actually support the correct answer?

        I also want a suggestion for how to improve the reasoning trace. What else could a pipeline, given an LLM + caption search + frames to query a VLM with, contain so that this question could have reasoned correctly? What caused the pipeline to go down the path to the wrong answer that it did, instead of the right answer? If you hadn't known the correct answer, would you have done what the pipeline did?
        """

            video_info = f" (video: {os.path.basename(actual_video_file)})" if actual_video_file else " (no video)"
            print(f"[{idx+1}/{len(trace)}] Analyzing {video_id}{video_info} with {len(frame_paths)} frames...")

            try:
                # Use multimodal API with video and frames if available
                if actual_video_file and frame_paths:
                    print("Using video and frames")
                    response = client.generate_with_video_and_images(prompt, actual_video_file, frame_paths, video=False)
                elif actual_video_file:
                    print("Using video and no frames")
                    response = client.generate_with_video_and_images(prompt, actual_video_file)
                elif frame_paths:
                    print("Using frames and no video")
                    response = client.generate_with_images(prompt, frame_paths)
                else:
                    response = client.generate_text(prompt)

                # Save response
                output_file = os.path.join(args.output_dir, f"analysis_{video_id}.txt")
                with open(output_file, "a") as f:
                    f.write(f"Video ID: {video_id}\n")
                    f.write(f"UID: {uid}\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(response)
                    f.write(f"\n\n{'='*80}\n")

                print(f"  ✓ Saved to {output_file}")

            except Exception as e:
                print(f"  ✗ Error analyzing {video_id}: {e}")
                continue

    print(f"\nCompleted! Analyzed {len(trace)} videos. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
