#!/usr/bin/env python3
"""
Script to concatenate rendered images from multiple frame directories into a video.

Usage:
    python make_video.py -m /path/to/output/cook_spinach --iteration 1000 --start 0 --end 299 -o ./video/concat.mp4
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Concatenate rendered images into video")
    parser.add_argument("-m", "--model_path", type=str, default="/scratch/rchkl2380/Workspace/4D_SOTA/train_outputs/3dgs_output/freetime_finetune_col/cook_spinach",
                        help="Path to the model output directory (e.g., cook_spinach)")
    parser.add_argument("--iteration", type=int, default=6000,
                        help="Which iteration results to use")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end", type=int, default=299,
                        help="End frame number (default: 299)")
    parser.add_argument("-o", "--output", type=str, default="/scratch/rchkl2380/Workspace/4D_SOTA/train_outputs/3dgs_output/freetime_finetune_col/cook_spinach/video.mp4",
                        help="Output video path (default: ./video/concat.mp4)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--render_type", type=str, default="renders", choices=["renders", "gt"],
                        help="Type of images to concatenate: 'renders' or 'gt' (default: renders)")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect image paths
    image_paths = []
    missing_frames = []
    
    print(f"Collecting images from iteration {args.iteration}...")
    for frame_num in range(args.start, args.end + 1):
        frame_dir = model_path / f"frame_{frame_num}" / "test" / f"ours_{args.iteration}" / args.render_type
        image_file = frame_dir / "00000.png"
        
        if image_file.exists():
            image_paths.append(str(image_file))
        else:
            missing_frames.append(frame_num)
    
    if not image_paths:
        print("Error: No images found!")
        return
    
    print(f"Found {len(image_paths)} images out of {args.end - args.start + 1} frames")
    if missing_frames:
        print(f"Missing {len(missing_frames)} frames")
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"Error: Cannot read first image: {image_paths[0]}")
        return
    
    height, width = first_img.shape[:2]
    print(f"Video resolution: {width}x{height} @ {args.fps} fps")
    
    # Try multiple codecs in order of preference
    codecs = [
        ('X264', 'H.264'),
        ('avc1', 'H.264 (avc1)'),
        ('XVID', 'XVID'),
        ('MJPG', 'Motion JPEG'),
        ('mp4v', 'MPEG-4')
    ]
    
    out = None
    for codec, codec_name in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec_name}")
                break
            else:
                out.release()
                out = None
        except:
            continue
    
    if out is None or not out.isOpened():
        print(f"Error: Cannot create video writer for {output_path}")
        print("Tried codecs: X264, avc1, XVID, MJPG, mp4v")
        return
    
    # Write frames
    print("Writing video...")
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image: {img_path}")
            continue
        
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
    
    out.release()
    print(f"Video saved to: {output_path}")
    print(f"Total frames written: {len(image_paths)}")


if __name__ == "__main__":
    main()