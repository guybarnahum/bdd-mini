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

LABELS_URL = cfg['download']['labels_url']
IMAGES_URL = cfg['download']['images_url']
NUM_VIDEOS = cfg['dataset']['num_videos']
SEED = cfg['dataset']['seed']
OUTPUT_DIR = cfg['dataset']['output_dir']
FRAME_STEP = cfg['dataset'].get('frame_step', 5)

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
                # Print a small sample of the first item
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
    # 1. Setup Directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    out_dir = Path(OUTPUT_DIR)
    img_out = out_dir / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    anno_out = out_dir / "annotations"
    anno_out.mkdir(exist_ok=True)

    print(f"⚙️  Builder Config: {NUM_VIDEOS} videos | Step: {FRAME_STEP} | Seed: {SEED}")

    # 2. Get Labels
    labels_zip_name = os.path.basename(LABELS_URL)
    labels_zip = data_dir / labels_zip_name
    download_file(LABELS_URL, labels_zip)

    # 3. Pick Videos from Labels
    print("🎲 Selecting random videos from labels...")
    import zipfile
    
    selected_video_names = []
    mini_annotations = []
    
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
            
            # --- 🛡️ DEFENSIVE PARSING START ---
            
            # 1. Validate: Must be a List
            if not isinstance(raw_data, list):
                print(f"❌ Error: Expected JSON list in {f}, but got {type(raw_data)}.")
                print_debug_structure(raw_data, f)
                continue # Skip this file

            # 2. Validate: Must not be empty
            if len(raw_data) == 0:
                print(f"⚠️ Warning: JSON list is empty in {f}. Skipping.")
                continue

            # 3. Validate: Items must have required keys
            first_frame = raw_data[0]
            if not isinstance(first_frame, dict):
                 print(f"❌ Error: List items must be dicts in {f}.")
                 print_debug_structure(raw_data, f)
                 continue

            # Check for critical keys
            # We need 'videoName' (or a way to infer it) and 'labels'/'objects'
            has_video_name = 'videoName' in first_frame
            has_labels = 'labels' in first_frame or 'objects' in first_frame
            
            if not has_video_name:
                # If missing, warn but try to recover using filename
                # print(f"⚠️ Note: 'videoName' missing in {f}. Will use filename.")
                pass 
            
            if not has_labels:
                # This is critical. If no labels, we can't train.
                print(f"❌ Error: Missing 'labels' or 'objects' key in {f}.")
                print("   Expected keys: 'labels', 'objects', 'videoName', 'frameIndex'")
                print_debug_structure(raw_data, f)
                continue

            # --- PARSING ---
            # Extract Video Name safely
            if has_video_name:
                v_name = first_frame['videoName']
            else:
                v_name = os.path.basename(f).replace('.json', '')
            
            # Wrap in standard structure
            video_entry = {
                "name": v_name,
                "frames": raw_data
            }
            
            mini_annotations.append(video_entry)
            selected_video_names.append(v_name)
            # --- 🛡️ DEFENSIVE PARSING END ---

    print(f"✅ Selected {len(selected_video_names)} valid videos.")
    if len(selected_video_names) == 0:
        print("❌ No valid videos found! Check the debug output above.")
        sys.exit(1)

    # 4. Stream Images
    print(f"☁️  Connecting to Remote Zip... (This may take 10-20s)")
    
    try:
        with RemoteZip(IMAGES_URL) as rz:
            all_files = rz.namelist()
            needed_videos = set(selected_video_names)
            files_to_extract = []
            
            for filename in all_files:
                if not filename.endswith('.jpg'): continue
                parts = filename.split('/')
                if len(parts) >= 2:
                    v_name = parts[-2]
                    if v_name in needed_videos:
                        files_to_extract.append(filename)
            
            print(f"⬇️  Streaming frames for {len(needed_videos)} videos...")
            
            count = 0
            for file_path in tqdm(files_to_extract):
                try:
                    fname = os.path.basename(file_path)
                    # Robust Frame Index Parsing
                    frame_part = fname.replace('.jpg', '')
                    if '-' in frame_part:
                         frame_str = frame_part.split('-')[-1]
                    else:
                         frame_str = frame_part[-7:] # Fallback
                    
                    try:
                        frame_idx = int(frame_str) - 1
                    except ValueError:
                        frame_idx = count
                    
                    if frame_idx % FRAME_STEP != 0:
                        continue 
                    
                    target = img_out / fname
                    if not target.exists():
                        rz.extract(file_path, path=data_dir)
                        full_extracted_path = data_dir / file_path
                        shutil.move(str(full_extracted_path), str(target))
                    count += 1
                except Exception as e:
                    pass
            
            print(f"✅ Extracted {count} frames.")
            
    except Exception as e:
        print(f"❌ Streaming Error: {e}")
        sys.exit(1)

    if (data_dir / "bdd100k").exists():
        shutil.rmtree(data_dir / "bdd100k")

    # 5. Generate COCO JSON
    print("📝 Generating COCO JSON...")
    coco_format = convert_to_coco(mini_annotations)
    
    out_json = anno_out / "train.json"
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
        
        # Sort frames
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
            
            # Support both 'labels' and 'objects' keys
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
                else:
                    continue

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
    build_mini_dataset()