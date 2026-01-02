import os
import json
import random
import sys
import shutil
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
OUTPUT_DIR = cfg['dataset']['output_dir']
FRAME_STEP = cfg['dataset'].get('frame_step', 5)

# Load Ratios
TRAIN_RATIO = cfg['dataset'].get('train_ratio', 0.70)
VAL_RATIO   = cfg['dataset'].get('val_ratio', 0.15)
TEST_RATIO  = cfg['dataset'].get('test_ratio', 0.15)

# --- DEBUG HELPER ---
def print_debug_structure(data, filename="Unknown"):
    """Prints the actual structure of the JSON to help user debug mismatches."""
    print(f"\n🚫 STRUCTURE MISMATCH IN: {filename}")
    print(f"   Type: {type(data)}")
    
    if isinstance(data, list):
        print(f"   List Length: {len(data)}")
        if len(data) > 0:
            first = data[0]
            print(f"   Item [0] Type: {type(first)}")
            if isinstance(first, dict):
                print(f"   Item [0] Keys: {list(first.keys())}")
                sample = json.dumps(first, indent=2)[:300] + "..."
                print(f"   Sample Data:\n{sample}")
    elif isinstance(data, dict):
        print(f"   Root Keys: {list(data.keys())}")
        sample = json.dumps(data, indent=2)[:300] + "..."
        print(f"   Sample Data:\n{sample}")
    print("---------------------------------------------------\n")

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

def build_mini_dataset():
    # 1. Validate Ratios
    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️ Warning: Ratios sum to {total_ratio:.2f}, not 1.0.")

    # 2. Setup Directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Persistent Cache for Images
    cache_dir = data_dir / "image_cache"
    cache_dir.mkdir(exist_ok=True)
    
    out_dir = Path(OUTPUT_DIR)
    
    # Define splits
    splits_needed = []
    if TRAIN_RATIO > 0: splits_needed.append("train")
    if VAL_RATIO > 0:   splits_needed.append("val")
    if TEST_RATIO > 0:  splits_needed.append("test")
    
    img_roots = {}
    for s in splits_needed:
        path = out_dir / "images" / s
        path.mkdir(parents=True, exist_ok=True)
        img_roots[s] = path
        
    anno_out = out_dir / "annotations"
    anno_out.mkdir(exist_ok=True)

    print(f"⚙️  Builder Config: {NUM_VIDEOS} videos | Step: {FRAME_STEP} | Seed: {SEED}")
    print(f"📊 Ratios: Train {TRAIN_RATIO:.2f} | Val {VAL_RATIO:.2f} | Test {TEST_RATIO:.2f}")

    # 3. Get Labels
    labels_zip_name = os.path.basename(LABELS_URL)
    labels_zip = data_dir / labels_zip_name
    download_file(LABELS_URL, labels_zip)

    # 4. Pick Videos (Defensive)
    print("🎲 Selecting random videos...")
    import zipfile
    
    parsed_videos = []
    
    with zipfile.ZipFile(labels_zip, 'r') as z_lbl:
        all_json = [f for f in z_lbl.namelist() if f.endswith(".json") and "train" in f]
        
        random.seed(SEED)
        if len(all_json) < NUM_VIDEOS:
            selected_files = all_json
        else:
            selected_files = random.sample(all_json, NUM_VIDEOS)
        
        for f in selected_files:
            content = z_lbl.read(f)
            raw_data = json.loads(content)
            
            # --- 🛡️ DEFENSIVE PARSING ---
            if not isinstance(raw_data, list):
                print(f"❌ Error: Expected JSON list in {f}, but got {type(raw_data)}.")
                print_debug_structure(raw_data, f)
                continue 

            if len(raw_data) == 0:
                print(f"⚠️ Warning: JSON list is empty in {f}. Skipping.")
                continue

            first_frame = raw_data[0]
            if not isinstance(first_frame, dict):
                 print(f"❌ Error: List items must be dicts in {f}.")
                 print_debug_structure(raw_data, f)
                 continue

            has_video_name = 'videoName' in first_frame
            has_labels = 'labels' in first_frame or 'objects' in first_frame
            
            if not has_labels:
                print(f"❌ Error: Missing 'labels' or 'objects' key in {f}.")
                print_debug_structure(raw_data, f)
                continue
            
            if has_video_name:
                v_name = first_frame['videoName']
            else:
                v_name = os.path.basename(f).replace('.json', '')
            
            parsed_videos.append({
                "name": v_name,
                "frames": raw_data
            })

    print(f"✅ Parsed {len(parsed_videos)} valid videos.")
    if len(parsed_videos) == 0:
        print("❌ No valid videos found! Check the debug output above.")
        sys.exit(1)

    # --- 5. Apply Splits ---
    random.shuffle(parsed_videos)
    total = len(parsed_videos)
    
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    
    train_set = parsed_videos[:n_train]
    val_set = parsed_videos[n_train:n_train+n_val]
    
    test_start = n_train + n_val
    if TEST_RATIO > 0:
        test_set = parsed_videos[test_start:]
    else:
        test_set = []
        if test_start < total:
             print(f"ℹ️  Dropping {total - test_start} videos due to 0% test ratio.")

    video_to_split_map = {} 
    for v in train_set: video_to_split_map[v['name']] = "train"
    for v in val_set:   video_to_split_map[v['name']] = "val"
    for v in test_set:  video_to_split_map[v['name']] = "test"

    print(f"📊 Final Split: {len(train_set)} Train, {len(val_set)} Val, {len(test_set)} Test")

    # 6. Stream Images with Caching
    print(f"☁️  Checking Cache & Streaming Missing Frames...")
    
    try:
        with RemoteZip(IMAGES_URL) as rz:
            all_files = rz.namelist()
            files_to_process = []
            needed_videos = set(video_to_split_map.keys())
            
            for filename in all_files:
                if not filename.endswith('.jpg'): continue
                parts = filename.split('/')
                if len(parts) >= 2:
                    v_name = parts[-2]
                    if v_name in needed_videos:
                        split = video_to_split_map[v_name]
                        files_to_process.append((filename, split))
            
            print(f"⬇️  Processing {len(files_to_process)} candidate frames (Cache Check)...")
            
            count_downloaded = 0
            count_cached = 0
            
            for file_info in tqdm(files_to_process):
                file_path, split = file_info
                try:
                    fname = os.path.basename(file_path)
                    
                    # Frame Logic
                    frame_part = fname.replace('.jpg', '')
                    if '-' in frame_part: frame_str = frame_part.split('-')[-1]
                    else: frame_str = frame_part[-7:]
                    
                    try:
                        frame_idx = int(frame_str) - 1
                    except ValueError:
                        frame_idx = 0
                    
                    if frame_idx % FRAME_STEP != 0:
                        continue 
                    
                    # 1. Check Local Cache
                    cached_file = cache_dir / fname
                    if not cached_file.exists():
                        # Download to cache
                        rz.extract(file_path, path=data_dir)
                        # rz extracts full path: data/bdd100k/images/track/train/video/frame.jpg
                        extracted_path = data_dir / file_path
                        shutil.move(str(extracted_path), str(cached_file))
                        count_downloaded += 1
                    else:
                        count_cached += 1
                    
                    # 2. Copy from Cache to Target Split
                    target = img_roots[split] / fname
                    if not target.exists():
                        shutil.copy2(str(cached_file), str(target))
                        
                except Exception as e:
                    pass
            
            print(f"✅ Finished: {count_downloaded} Downloaded, {count_cached} Used Cache.")
            
    except Exception as e:
        # Check if it was a user interrupt inside the stream logic
        # (Though RemoteZip usually raises standard errors, keyboard interrupt is handled at main level)
        print(f"❌ Streaming Error: {e}")
        sys.exit(1)

    # Cleanup extraction temp folders (but keep cache!)
    if (data_dir / "bdd100k").exists():
        shutil.rmtree(data_dir / "bdd100k")

    # 7. Generate COCO JSONs
    print("📝 Generating COCO JSONs...")
    datasets = [("train", train_set), ("val", val_set), ("test", test_set)]
    
    for split_name, video_data in datasets:
        if not video_data: continue
        print(f"   - Building {split_name}.json...")
        coco_format = convert_to_coco(video_data)
        out_json = anno_out / f"{split_name}.json"
        with open(out_json, 'w') as f:
            json.dump(coco_format, f)
    
    print(f"🚀 Done! Mini-BDD ready at: {out_dir}")

def convert_to_coco(bdd_data_list):
    CLASS_MAP = {
        "pedestrian": 1, "rider": 1,
        "car": 2, "truck": 2, "bus": 2, "train": 2
    }
    
    coco = {
        "videos": [], "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]
    }
    
    global_img_id = 1
    global_ann_id = 1
    track_map = {} 
    global_track_id = 1

    for video_idx, vid_data in enumerate(bdd_data_list):
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
                    x1 = obj['box2d']['x1']
                    y1 = obj['box2d']['y1']
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
    return coco

if __name__ == "__main__":
    try:
        build_mini_dataset()
    except KeyboardInterrupt:
        print("\n\n🛑 Execution interrupted by user.")
        print("   ✅ Partial progress saved in 'data/image_cache'.")
        print("   🚀 Run the script again to resume downloading.\n")
        sys.exit(0)