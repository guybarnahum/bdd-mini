import json
import cv2
import random
import subprocess
import argparse
import sys
from pathlib import Path

FFMPEG_BIN = "ffmpeg"

def get_color(idx):
    random.seed(idx)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

def draw_hud(img, frame_idx, video_name, fmt):
    # Draw a black bar at the top
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    text = f"Format: {fmt.upper()} | Video: {video_name} | Frame: {frame_idx}"
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# --- DATA LOADERS ---
def load_from_coco(split, output_dir):
    json_path = Path(output_dir) / "annotations" / f"{split}.json"
    if not json_path.exists(): 
        print(f"❌ Annotations not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    vid_id_map = {v['id']: v['file_name'] for v in data['videos']}
    video_content = {}
    
    # Pre-group annotations by image_id
    img_to_anns = {}
    for ann in data['annotations']:
        iid = ann['image_id']
        if iid not in img_to_anns: img_to_anns[iid] = []
        img_to_anns[iid].append(ann)
        
    # Group images by video
    for img in data['images']:
        vid_name = vid_id_map.get(img['video_id'], "unknown")
        if vid_name not in video_content: video_content[vid_name] = []
        
        # Parse Annotations: [x, y, w, h, track_id]
        anns = img_to_anns.get(img['id'], [])
        parsed_anns = []
        for a in anns:
            x, y, w, h = a['bbox']
            parsed_anns.append((x, y, w, h, a['track_id']))
        
        # FIX: Construct full path from Deep Structure
        # JSON file_name is "Video/img1/00000001.jpg"
        # We prepend output_dir/split/
        full_path = Path(output_dir) / split / img['file_name']
            
        video_content[vid_name].append({
            "file_name": str(full_path),
            "anns": parsed_anns
        })
        
    return video_content

def load_from_mot(split, output_dir):
    # FIX: Root is now the split folder itself containing video subfolders
    split_root = Path(output_dir) / split
    if not split_root.exists(): return None
    
    video_content = {}
    
    # Iterate over video folders in the split directory
    for vid_dir in split_root.iterdir():
        if not vid_dir.is_dir(): continue
        vid_name = vid_dir.name
        
        # Check for Deep MOT structure: gt/gt.txt and img1/
        gt_path = vid_dir / "gt" / "gt.txt"
        img_dir = vid_dir / "img1"
        
        if not gt_path.exists() or not img_dir.exists(): continue
        
        # 1. Parse GT file
        frame_data = {}
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # MOT Format: frame, id, x, y, w, h, ...
                f_idx = int(parts[0])
                tid = int(parts[1])
                bbox = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                
                if f_idx not in frame_data: frame_data[f_idx] = []
                frame_data[f_idx].append((bbox[0], bbox[1], bbox[2], bbox[3], tid))
                
        # 2. Get Images (Sorted numerically from img1 folder)
        all_files = sorted(list(img_dir.glob("*.jpg")))
        
        frames_list = []
        for i, img_path in enumerate(all_files):
            # Map filename '00000001.jpg' -> int(1) to match GT
            try:
                gt_frame_idx = int(img_path.stem)
            except:
                gt_frame_idx = i + 1 
            
            anns = frame_data.get(gt_frame_idx, [])
            frames_list.append({
                "file_name": str(img_path),
                "anns": anns
            })
            
        video_content[vid_name] = frames_list
        
    return video_content

# --- RENDERER ---
def render_video(fmt, split, data_dir, specific_video=None):
    # Removed old img_root logic; we now trust the loaders to provide full paths
    
    # User requested: output/rendered
    movie_out_dir = Path("output/rendered")
    movie_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading data from {fmt.upper()} format in {data_dir}...")
    
    if fmt == "coco":
        content = load_from_coco(split, data_dir)
    else:
        content = load_from_mot(split, data_dir)
        
    if not content:
        print(f"❌ Could not load data for {fmt}. Check if files exist.")
        return

    # Select Video
    if specific_video:
        target_vid = specific_video
        if target_vid not in content:
            print(f"❌ Video '{target_vid}' not found in {list(content.keys())[:3]}...")
            return
    else:
        target_vid = random.choice(list(content.keys()))
        print(f"🎲 Randomly selected video: {target_vid}")

    frames = content[target_vid]
    frames.sort(key=lambda x: x['file_name'])
    
    if not frames:
        print("❌ No frames found.")
        return

    # Setup FFmpeg
    # FIX: Use the full path provided by loader
    first_path = Path(frames[0]['file_name'])
    if not first_path.exists():
        print(f"❌ Image missing: {first_path}")
        return
        
    sample = cv2.imread(str(first_path))
    h, w = sample.shape[:2]
    
    # --- FIX: Ensure dimensions are even for libx264 ---
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1
    # --------------------------------------------------
    
    # Save as: output/rendered/VideoName_format.mp4
    out_file = movie_out_dir / f"{target_vid}_{fmt}.mp4"
    
    # FFmpeg command
    command = [
        FFMPEG_BIN, '-y', 
        '-f', 'rawvideo', 
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', 
        '-pix_fmt', 'bgr24', 
        '-r', '10',  # 10 fps playback
        '-i', '-', 
        '-c:v', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        '-preset', 'fast',
        str(out_file)
    ]

    print(f"🎬 Rendering {len(frames)} frames to {out_file}...")
    try:
        # Changed stderr to None to expose FFmpeg errors if they happen
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=None)
        
        for i, frame_info in enumerate(frames):
            # FIX: Load image using the absolute path
            img = cv2.imread(frame_info['file_name'])
            if img is None: continue
            
            # --- FIX: Resize image if it doesn't match the forced even dimensions ---
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            # ----------------------------------------------------------------------
            
            # Draw Boxes
            for x, y, bw, bh, tid in frame_info['anns']:
                color = get_color(tid)
                cv2.rectangle(img, (int(x), int(y)), (int(x+bw), int(y+bh)), color, 2)
                cv2.putText(img, str(tid), (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            draw_hud(img, i, target_vid, fmt)
            
            # Write to pipe
            proc.stdin.write(img.tobytes())
            
            if i % 10 == 0: sys.stdout.write(".")
            sys.stdout.flush()
            
        proc.stdin.close()
        proc.wait()
        print(f"\n✅ Done!")
    except Exception as e:
        print(f"\n❌ FFmpeg Error: {e}")
        if proc: proc.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["coco", "mot"], default="coco", help="Source format to visualize")
    parser.add_argument("--split", default="train", help="Which split (train/val/test)")
    parser.add_argument("--dir", default="output/dataset", help="Path to your dataset root")
    parser.add_argument("--video", default=None, help="Name of specific video to render")
    args = parser.parse_args()
    
    render_video(args.format, args.split, args.dir, args.video)