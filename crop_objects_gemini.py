#!/usr/bin/env python3
"""
Object Detection and Cropping Tool using Google Gemini API
Detects objects in images using text prompts and crops them for detailed VLM analysis.
"""

import os
import json
import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Optional, Tuple


def load_api_key(env_file='env.json') -> str:
    """Load Gemini API key from env.json"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, env_file)

    with open(env_path, 'r') as f:
        env_data = json.load(f)

    return env_data['gemini_api_key']


def detect_objects_with_gemini(
    image_path: str,
    object_query: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash-exp"
) -> List[Dict]:
    """
    Detect objects in an image using Gemini's bounding box detection.

    Args:
        image_path: Path to the image file
        object_query: Text description of objects to detect (e.g., "bird on branch")
        api_key: Gemini API key
        model_name: Gemini model to use (gemini-2.0-flash-exp or gemini-2.5-flash-preview-04-17)

    Returns:
        List of detected objects with bounding boxes in format:
        [{"label": "bird", "box_2d": [y0, x0, y1, x1], "confidence": 0.95}, ...]
        Coordinates are normalized to [0, 1000]
    """
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Load image
    img = Image.open(image_path)

    # Create prompt for object detection
    prompt = f"""Detect all instances of: {object_query}

Return the bounding boxes in JSON format. For each detected object, provide:
- label: description of the object
- box_2d: bounding box coordinates [y0, x0, y1, x1] normalized to [0, 1000]

Return ONLY a JSON array of detected objects, no other text.
Example format:
[
  {{"label": "bird on branch", "box_2d": [100, 200, 300, 400]}},
  {{"label": "bird on branch", "box_2d": [500, 600, 700, 800]}}
]

If no objects are found, return an empty array: []"""

    # Query Gemini
    response = model.generate_content([prompt, img])
    response_text = response.text.strip()

    # Parse JSON response
    # Remove markdown code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()

    try:
        detections = json.loads(response_text)
        return detections if isinstance(detections, list) else []
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse Gemini response as JSON: {e}")
        print(f"Response: {response_text[:500]}")
        return []


def denormalize_bbox(bbox: List[int], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized bounding box coordinates [0, 1000] to pixel coordinates.

    Args:
        bbox: Bounding box in format [y0, x0, y1, x1] with values in [0, 1000]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple (x0, y0, x1, y1) in pixel coordinates
    """
    y0, x0, y1, x1 = bbox

    # Convert from [0, 1000] to [0, 1] then to pixel coordinates
    x0_px = int((x0 / 1000.0) * img_width)
    y0_px = int((y0 / 1000.0) * img_height)
    x1_px = int((x1 / 1000.0) * img_width)
    y1_px = int((y1 / 1000.0) * img_height)

    return (x0_px, y0_px, x1_px, y1_px)


def crop_and_save_objects(
    image_path: str,
    detections: List[Dict],
    output_dir: str,
    base_name: Optional[str] = None
) -> List[str]:
    """
    Crop detected objects from image and save them.

    Args:
        image_path: Path to source image
        detections: List of detected objects with bounding boxes
        output_dir: Directory to save cropped images
        base_name: Optional base name for cropped files (default: use source filename)

    Returns:
        List of paths to saved cropped images
    """
    if not detections:
        print("No objects detected, nothing to crop")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Use source filename as base if not provided
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    cropped_paths = []

    for idx, detection in enumerate(detections):
        bbox = detection.get('box_2d')
        if not bbox or len(bbox) != 4:
            print(f"Warning: Invalid bbox for detection {idx}: {bbox}")
            continue

        # Convert normalized coordinates to pixels
        x0, y0, x1, y1 = denormalize_bbox(bbox, img_width, img_height)

        # Ensure coordinates are within image bounds
        x0 = max(0, min(x0, img_width))
        y0 = max(0, min(y0, img_height))
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))

        # Skip invalid boxes
        if x1 <= x0 or y1 <= y0:
            print(f"Warning: Invalid crop region for detection {idx}: ({x0}, {y0}, {x1}, {y1})")
            continue

        # Crop image
        cropped_img = img.crop((x0, y0, x1, y1))

        # Generate filename
        label = detection.get('label', 'object').replace(' ', '_').replace('/', '_')
        crop_filename = f"{base_name}_crop_{idx}_{label}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)

        # Save cropped image
        cropped_img.save(crop_path, quality=95)
        cropped_paths.append(crop_path)

        print(f"✓ Saved crop {idx+1}/{len(detections)}: {crop_filename}")

    return cropped_paths


def crop_objects_from_frame(
    frame_path: str,
    object_query: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict:
    """
    Complete pipeline: detect objects and crop them from a video frame.

    Args:
        frame_path: Path to video frame (e.g., "vid_path/frames/frame_0050.jpg")
        object_query: Text description of what to detect (e.g., "bird on branch")
        output_dir: Where to save crops (default: same dir as frame with "_crops" suffix)
        api_key: Gemini API key (default: load from env.json)

    Returns:
        Dict with results:
        {
            "success": True/False,
            "detections": [...],
            "cropped_paths": [...],
            "error": "..." (if any)
        }
    """
    try:
        # Load API key if not provided
        if api_key is None:
            api_key = load_api_key()

        # Set default output directory
        if output_dir is None:
            frame_dir = os.path.dirname(frame_path)
            video_dir = os.path.dirname(frame_dir)  # Parent of "frames" dir
            output_dir = os.path.join(video_dir, "cropped_objects")

        print(f"Detecting objects in: {os.path.basename(frame_path)}")
        print(f"Query: '{object_query}'")

        # Detect objects
        detections = detect_objects_with_gemini(
            image_path=frame_path,
            object_query=object_query,
            api_key=api_key
        )

        print(f"Found {len(detections)} object(s)")

        if not detections:
            return {
                "success": True,
                "detections": [],
                "cropped_paths": [],
                "message": f"No objects matching '{object_query}' found in frame"
            }

        # Crop and save objects
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        cropped_paths = crop_and_save_objects(
            image_path=frame_path,
            detections=detections,
            output_dir=output_dir,
            base_name=frame_name
        )

        return {
            "success": True,
            "detections": detections,
            "cropped_paths": cropped_paths,
            "message": f"Successfully cropped {len(cropped_paths)} object(s)"
        }

    except Exception as e:
        return {
            "success": False,
            "detections": [],
            "cropped_paths": [],
            "error": str(e)
        }


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Crop objects from video frames using Gemini')
    parser.add_argument('frame_path', help='Path to video frame')
    parser.add_argument('query', help='Object to detect (e.g., "bird on branch")')
    parser.add_argument('--output-dir', help='Output directory for crops')
    args = parser.parse_args()

    result = crop_objects_from_frame(
        frame_path=args.frame_path,
        object_query=args.query,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    if result['success']:
        print(f"✓ Success: {result['message']}")
        if result['cropped_paths']:
            print(f"\nCropped images saved to:")
            for path in result['cropped_paths']:
                print(f"  - {path}")
    else:
        print(f"✗ Error: {result['error']}")
    print("="*60)


if __name__ == "__main__":
    main()
