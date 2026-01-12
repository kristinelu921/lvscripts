#!/usr/bin/env python3
"""
Test script to verify subtitle integration in critic model
"""
import sys
sys.path.insert(0, '/mnt/ssd/data/lvscripts')

from critic_model_os import load_subtitle_mapping, get_subtitles_for_frames

def test_subtitle_loading():
    """Test that subtitle mapping loads correctly"""
    print("=" * 60)
    print("Testing Subtitle Integration")
    print("=" * 60)

    # Test 1: Load subtitle mapping
    print("\n1. Loading subtitle mapping...")
    subtitles = load_subtitle_mapping()
    print(f"   ✓ Loaded {len(subtitles)} videos")

    # Test 2: Get subtitles for specific frames
    print("\n2. Testing subtitle extraction for frames...")

    # Use the example from the subtitle file
    test_video_id = "pr8-agDG8sI"
    test_frames = [
        "frames/frame_0003.jpg",
        "frames/frame_0005.jpg",
        "frames/frame_0007.jpg",
        "frames/frame_0100.jpg"  # Non-existent frame
    ]

    frame_subtitles = get_subtitles_for_frames(test_video_id, test_frames)

    print(f"   Video ID: {test_video_id}")
    print(f"   Test frames: {test_frames}")
    print(f"   Found subtitles for {len(frame_subtitles)} frames:")

    for frame, subtitle in frame_subtitles.items():
        print(f"      {frame}: \"{subtitle}\"")

    # Test 3: Test with non-existent video
    print("\n3. Testing with non-existent video...")
    missing_subtitles = get_subtitles_for_frames("nonexistent_video", test_frames)
    print(f"   ✓ Returned empty dict: {len(missing_subtitles) == 0}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    test_subtitle_loading()
