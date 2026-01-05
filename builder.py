import os
import json
import random
import sys
import shutil
import configparser
from pathlib import Path
from tqdm import tqdm

# --- Import Dependencies ---
try:
    from remotezip import RemoteZip
except ImportError:
    print("❌ Missing 'remotezip'. Please run: ./setup.sh")
    sys.exit(1)

try:
    import tomllib as toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        print("❌ Missing TOML library. Please run: ./setup.sh")
        sys.exit(1)

# --- CONFIGURATION ---
CONFIG_FILE = "config.toml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    with open(CONFIG_FILE, "rb") as f:
        return toml.load(f)

cfg = load_config()

# Load basic settings
LABELS_URL = cfg['download']['labels_url']
IMAGES_URL = cfg['download']['images_url']
NUM_VIDEOS = cfg['dataset']['num_videos']
SEED = cfg['dataset']['seed']
OUTPUT_DIR = Path(cfg['dataset']['output_dir'])
FRAME_STEP = cfg['dataset'].get('frame_step', 5)

# Output Formats (Default to COCO if missing)
EXPORT_FORMATS = cfg['dataset'].get('export_formats', ["coco"])

# Load Ratios
TRAIN_RATIO = cfg['dataset'].get('train_ratio', 0.70)
VAL_RATIO   = cfg['dataset'].get('val_ratio', 0.15)
TEST_RATIO  = cfg['dataset'].get('test_ratio', 0.15)

# --- DEBUG HELPER ---
def print_debug_structure(data, filename="Unknown"):
    """Prints structure for debugging."""
    print(f"\n🚫 STRUCTURE MISMATCH IN: {filename}")
    print(f"   Type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict):
            print(f"   Item [0] Keys: {list(first.keys())}")

def download_file(url, dest_path):
    import requests
    if dest_path.exists():
        print(f"✅ Found existing labels at {dest_path}")
        return
    print(f"⬇️  Downloading Labels...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    f.write(chunk)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

# --- EXPORTERS ---

def save_coco_format(split_name, video_data, output_root):
    """Generates Standard COCO-Video JSON."""
    anno_dir = output_root / "annotations"
    anno_dir.mkdir(exist_ok=True)
    
    # Check if we should skip
    if not video_data: return

    print(f"   - Building {split_name}.json (COCO)...")
    
    CLASS_MAP = {"pedestrian": 1, "rider": 1, "car": 2, "truck": 2, "bus": 2, "train": 2}
    coco = {
        "videos": [], "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]
    }
    
    global_img_id = 1
    global_ann_id = 1
    track_map = {} 
    global_track_id = 1

    for video_idx, vid_data in enumerate(video_data):
        video_name = vid_data['name']
        coco["videos"].append({"id": video_idx + 1, "file_name": video_name})
        
        frames = sorted(vid_data['frames'], key=lambda x: x.get('frameIndex', 0))
        
        for frame_data in frames:
            f_idx = frame_data.get('frameIndex', 0)
            if f_idx % FRAME_STEP != 0: continue
            
            img_file = f"{video_name}-{f_idx+1:07d}.jpg"
            
            coco["images"].append({
                "id": global_img_id,
                "video_id": video_idx + 1,
                "file_name": img_file,
                "frame_id": f_idx,
                "height": 720, "width": 1280
            })
            
            objects = frame_data.get('labels', frame_data.get('objects', []))
            for obj in objects:
                if obj['category'] not in CLASS_MAP: continue
                cat_id = CLASS_MAP[obj['category']]
                t_key = (video_name, obj['id'])
                if t_key not in track_map:
                    track_map[t_key] = global_track_id
                    global_track_id += 1
                
                if 'box2d' in obj:
                    x1, y1 = obj['box2d']['x1'], obj['box2d']['y1']
                    w = obj['box2d']['x2'] - x1
                    h = obj['box2d']['y2'] - y1
                else: continue

                coco["annotations"].append({
                    "id": global_ann_id,
                    "image_id": global_img_id,
                    "category_id": cat_id,
                    "track_id": track_map[t_key],
                    "bbox": [x1, y1, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "video_id": video_idx + 1
                })
                global_ann_id += 1
            global_img_id += 1
            
    with open(anno_dir / f"{split_name}.json", 'w') as f:
        json.dump(coco, f)

def save_mot_format(split_name, video_data, output_root):
    """Generates MOTChallenge Format (gt.txt, seqinfo.ini)."""
    mot_root = output_root / "mot_format" / split_name
    mot_root.mkdir(parents=True, exist_ok=True)
    
    if not video_data: return
    print(f"   - Building {split_name}/ (MOTChallenge)...")
    
    CLASS_MAP = {"pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5, "train": 6}
    
    for vid_data in video_data:
        v_name = vid_data['name']
        
        # 1. Create Folder Structure: output/mot_format/train/VideoName/gt/
        seq_dir = mot_root / v_name
        gt_dir = seq_dir / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Generate gt.txt
        # Format: frame, id, left, top, width, height, conf(1), class, vis(1)
        gt_path = gt_dir / "gt.txt"
        
        frames = sorted(vid_data['frames'], key=lambda x: x.get('frameIndex', 0))
        valid_frames = [f for f in frames if f.get('frameIndex', 0) % FRAME_STEP == 0]
        
        with open(gt_path, 'w') as f_gt:
            # Re-index frames to start at 1 and be continuous for MOT
            for local_frame_idx, frame_data in enumerate(valid_frames, start=1):
                
                objects = frame_data.get('labels', frame_data.get('objects', []))
                for obj in objects:
                    if obj['category'] not in CLASS_MAP: continue
                    cls_id = CLASS_MAP[obj['category']]
                    
                    # Convert UUID track ID to Integer for MOT format
                    # Simple hash-based approach for local uniqueness
                    t_id_int = abs(hash(obj['id'])) % 100000 
                    
                    if 'box2d' in obj:
                        x1 = obj['box2d']['x1']
                        y1 = obj['box2d']['y1']
                        w = obj['box2d']['x2'] - x1
                        h = obj['box2d']['y2'] - y1
                        
                        # Write Line
                        # conf=1, vis=1 (defaults)
                        line = f"{local_frame_idx},{t_id_int},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{cls_id},1\n"
                        f_gt.write(line)

        # 3. Generate seqinfo.ini
        # Required by TrackEval
        with open(seq_dir / "seqinfo.ini", 'w') as f_ini:
            f_ini.write("[Sequence]\n")
            f_ini.write(f"name={v_name}\n")
            f_ini.write(f"imDir=images_linked\n") # Placeholder, usually points to img1
            f_ini.write(f"frameRate={30/FRAME_STEP}\n")
            f_ini.write(f"seqLength={len(valid_frames)}\n")
            f_ini.write(f"imWidth=1280\n")
            f_ini.write(f"imHeight=720\n")
            f_ini.write(f"imExt=.jpg\n")

# --- MAIN BUILDER ---

def build_mini_dataset():
    # 1. Validate Ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️ Warning: Ratios sum to {total_ratio:.2f}, not 1.0.")

    # 2. Setup Directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    cache_dir = data_dir / "image_cache"
    cache_dir.mkdir(exist_ok=True)
    out_dir = OUTPUT_DIR
    
    splits_needed = []
    if TRAIN_RATIO > 0: splits_needed.append("train")
    if VAL_RATIO > 0:   splits_needed.append("val")
    if TEST_RATIO > 0:  splits_needed.append("test")
    
    img_roots = {}
    for s in splits_needed:
        path = out_dir / "images" / s
        path.mkdir(parents=True, exist_ok=True)
        img_roots[s] = path
    
    print(f"⚙️  Config: {NUM_VIDEOS} videos | Step: {FRAME_STEP} | Formats: {EXPORT_FORMATS}")

    # 3. Get Labels
    labels_zip_name = os.path.basename(LABELS_URL)
    labels_zip = data_dir / labels_zip_name
    download_file(LABELS_URL, labels_zip)

    # 4. Pick Videos
    print("🎲 Selecting random videos...")
    import zipfile
    parsed_videos = []
    
    with zipfile.ZipFile(labels_zip, 'r') as z_lbl:
        all_json = [f for f in z_lbl.namelist() if f.endswith(".json") and "train" in f]
        all_json.sort() # Deterministic Sort
        
        random.seed(SEED)
        if len(all_json) < NUM_VIDEOS:
            selected_files = all_json
        else:
            selected_files = random.sample(all_json, NUM_VIDEOS)
        
        for f in selected_files:
            content = z_lbl.read(f)
            raw_data = json.loads(content)
            
            # Defensive checks
            if not isinstance(raw_data, list) or len(raw_data) == 0: continue
            if not isinstance(raw_data[0], dict): continue
            
            first = raw_data[0]
            v_name = first.get('videoName', os.path.basename(f).replace('.json', ''))
            
            # Check required keys
            if 'labels' not in first and 'objects' not in first:
                print(f"❌ Missing labels in {f}")
                print_debug_structure(raw_data, f)
                continue

            parsed_videos.append({"name": v_name, "frames": raw_data})

    print(f"✅ Parsed {len(parsed_videos)} valid videos.")
    if len(parsed_videos) == 0: sys.exit(1)

    # 5. Apply Splits
    random.shuffle(parsed_videos)
    n_train = int(len(parsed_videos) * TRAIN_RATIO)
    n_val = int(len(parsed_videos) * VAL_RATIO)
    
    train_set = parsed_videos[:n_train]
    val_set = parsed_videos[n_train:n_train+n_val]
    test_set = parsed_videos[n_train+n_val:] if TEST_RATIO > 0 else []

    # Map video -> split
    video_to_split = {}
    for v in train_set: video_to_split[v['name']] = "train"
    for v in val_set:   video_to_split[v['name']] = "val"
    for v in test_set:  video_to_split[v['name']] = "test"

    print(f"📊 Final Split: {len(train_set)} Train, {len(val_set)} Val, {len(test_set)} Test")

    # 6. Stream Images
    print(f"☁️  Checking Cache & Streaming Frames...")
    try:
        with RemoteZip(IMAGES_URL) as rz:
            all_files = rz.namelist()
            files_to_process = []
            needed_videos = set(video_to_split.keys())
            
            for filename in all_files:
                if not filename.endswith('.jpg'): continue
                parts = filename.split('/')
                if len(parts) >= 2 and parts[-2] in needed_videos:
                    files_to_process.append((filename, video_to_split[parts[-2]]))
            
            dl_count, cache_count = 0, 0
            
            for file_info in tqdm(files_to_process):
                file_path, split = file_info
                try:
                    fname = os.path.basename(file_path)
                    
                    # Frame Index Logic
                    frame_part = fname.replace('.jpg', '')
                    if '-' in frame_part: frame_str = frame_part.split('-')[-1]
                    else: frame_str = frame_part[-7:]
                    try: frame_idx = int(frame_str) - 1
                    except: frame_idx = 0
                    
                    if frame_idx % FRAME_STEP != 0: continue 

                    # Cache & Copy Logic
                    cached_file = cache_dir / fname
                    if not cached_file.exists():
                        rz.extract(file_path, path=data_dir)
                        shutil.move(str(data_dir / file_path), str(cached_file))
                        dl_count += 1
                    else:
                        cache_count += 1
                    
                    target = img_roots[split] / fname
                    if not target.exists():
                        shutil.copy2(str(cached_file), str(target))

                except Exception as e: pass
            
            print(f"✅ Images: {dl_count} Downloaded, {cache_count} Cached.")

    except Exception as e:
        print(f"❌ Streaming Error: {e}")
        sys.exit(1)

    if (data_dir / "bdd100k").exists(): shutil.rmtree(data_dir / "bdd100k")

    # 7. Generate Output Formats
    print("📝 Generating Outputs...")
    split_datasets = [("train", train_set), ("val", val_set), ("test", test_set)]

    for split, data in split_datasets:
        if not data: continue
        
        # Always output COCO (Standard MOTIP) if requested
        if "coco" in EXPORT_FORMATS:
            save_coco_format(split, data, out_dir)
            
        # Optional MOT Challenge Format
        if "mot" in EXPORT_FORMATS:
            save_mot_format(split, data, out_dir)

    print(f"🚀 Done! Mini-BDD ready at: {out_dir}")

if __name__ == "__main__":
    try:
        build_mini_dataset()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Progress saved in 'data/image_cache'.")
        sys.exit(0)