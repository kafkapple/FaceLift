#!/usr/bin/env python3
"""
Extract video clips around specific frames for inspection.

Usage:
    python extract_frame_context.py --video input.mp4 --frames 5900,11800,17700 --window 3.0 --speed 0.25
    
Examples:
    # Default: 3 seconds before/after, 0.25x speed (slow motion)
    python extract_frame_context.py --video cam.mp4 --frames 5900
    
    # Custom window and speed
    python extract_frame_context.py --video cam.mp4 --frames 5900,11800 --window 5.0 --speed 0.1
    
    # Multiple cameras
    python extract_frame_context.py --video /path/to/videos_undist/0.mp4 --frames 5900 --output /tmp/output
"""

import argparse
import cv2
import os
from pathlib import Path


def extract_context_video(
    video_path: str,
    target_frames: list[int],
    window_seconds: float = 3.0,
    speed: float = 0.25,
    output_dir: str = None,
    output_fps: float = None,
):
    """
    Extract video clips around target frames.
    
    Args:
        video_path: Path to input video
        target_frames: List of frame indices to extract context around
        window_seconds: Seconds before and after target frame
        speed: Playback speed (0.25 = 4x slower, 0.1 = 10x slower)
        output_dir: Output directory (default: same as video)
        output_fps: Output FPS (default: original_fps * speed)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate window in frames
    window_frames = int(original_fps * window_seconds)
    
    # Output FPS for slow motion
    if output_fps is None:
        output_fps = original_fps * speed
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(video_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(video_path).stem
    
    print(f"=" * 60)
    print(f"Video: {video_path}")
    print(f"  Original: {total_frames} frames, {original_fps:.1f} fps")
    print(f"  Size: {width}x{height}")
    print(f"=" * 60)
    print(f"Settings:")
    print(f"  Window: ±{window_seconds}s = ±{window_frames} frames")
    print(f"  Speed: {speed}x (output {output_fps:.1f} fps)")
    print(f"  Output: {output_dir}")
    print(f"=" * 60)
    
    for target_frame in target_frames:
        start_frame = max(0, target_frame - window_frames)
        end_frame = min(total_frames - 1, target_frame + window_frames)
        num_frames = end_frame - start_frame + 1
        
        # Duration in output video
        output_duration = num_frames / output_fps
        
        output_filename = f"{video_name}_frame{target_frame}_w{window_seconds}s_speed{speed}x.mp4"
        output_path = output_dir / output_filename
        
        print(f"\nTarget frame: {target_frame}")
        print(f"  Range: {start_frame} ~ {end_frame} ({num_frames} frames)")
        print(f"  Output duration: {output_duration:.1f}s")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for f_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"  Warning: Failed to read frame {f_idx}")
                break
            
            # Add frame info overlay
            # Background for text
            cv2.rectangle(frame, (5, 5), (350, 85), (0, 0, 0), -1)
            
            # Frame number
            cv2.putText(frame, f'Frame: {f_idx}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Relative position
            rel_pos = f_idx - target_frame
            rel_text = f"Target {'+' if rel_pos >= 0 else ''}{rel_pos}"
            cv2.putText(frame, rel_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Highlight target frame
            if f_idx == target_frame:
                cv2.putText(frame, '>>> TARGET FRAME <<<', (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 15)
            
            # Progress bar at bottom
            progress = (f_idx - start_frame) / num_frames
            bar_width = int(width * progress)
            cv2.rectangle(frame, (0, height-10), (bar_width, height), (0, 255, 0), -1)
            
            # Mark target position on progress bar
            target_pos = int(width * (target_frame - start_frame) / num_frames)
            cv2.line(frame, (target_pos, height-15), (target_pos, height), (0, 0, 255), 3)
            
            out.write(frame)
        
        out.release()
        print(f"  Saved: {output_path}")
    
    cap.release()
    print(f"\n{'=' * 60}")
    print("Done!")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract video clips around specific frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with JUMP_FRAMES
  python extract_frame_context.py \\
      --video /path/to/video.mp4 \\
      --frames 5900,11800,17700

  # Slow motion (10x slower) with 5 second window
  python extract_frame_context.py \\
      --video /path/to/video.mp4 \\
      --frames 5900 \\
      --window 5.0 \\
      --speed 0.1

  # Custom output directory
  python extract_frame_context.py \\
      --video /path/to/video.mp4 \\
      --frames 5900 \\
      --output /tmp/inspection
        """
    )
    
    parser.add_argument('--video', '-v', required=True,
                        help='Input video path')
    parser.add_argument('--frames', '-f', required=True,
                        help='Target frame(s), comma-separated (e.g., 5900,11800,17700)')
    parser.add_argument('--window', '-w', type=float, default=3.0,
                        help='Window in seconds before/after target (default: 3.0)')
    parser.add_argument('--speed', '-s', type=float, default=0.25,
                        help='Playback speed multiplier (default: 0.25 = 4x slower)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: same as video)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Override output FPS (default: auto based on speed)')
    
    args = parser.parse_args()
    
    # Parse frames
    target_frames = [int(f.strip()) for f in args.frames.split(',')]
    
    extract_context_video(
        video_path=args.video,
        target_frames=target_frames,
        window_seconds=args.window,
        speed=args.speed,
        output_dir=args.output,
        output_fps=args.fps,
    )


if __name__ == '__main__':
    main()
