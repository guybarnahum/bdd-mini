import os
import json
import random
import sys
import shutil
import zipfile
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

# --- GLOBAL CLASS MAPPINGS ---

# 1. Output Mapping (String -> MOTIP ID)
# Used by both COCO and MOT exporters
UNIFIED_CLASS_MAP = {
    "pedestrian": 1, 
    "rider": 1, 
    "car": 2, 
    "truck": 2, 
    "bus": 2, 
    "train": 2, 
    "vehicle": 2
}

# 2. Output Categories Definition (for COCO JSON)
COCO_CATEGORIES = [
    {"id": 1, "name": "person"}, 
    {"id": 2, "name": "vehicle"}
]

# 3. VisDrone Raw ID -> String Label
# Used during VisDrone parsing
VISDRONE_RAW_MAP = {
    1: "pedestrian", # Pedestrian
    2: "pedestrian", # People
    4: "car",        # Car
    5: "car",        # Van
    6: "car",        # Truck
    9: "car"         # Bus
}

# Load basic settings
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

def save_manifest(splits, output_root):
    """Generates a simple text manifest of all processed videos."""
    manifest_path = output_root / "manifest.txt"
    
    print(f"📝 Saving manifest to {manifest_path}...")
    
    with open(manifest_path, 'w') as f:
        f.write(f"BDD-Mini Build Manifest\n")
        f.write(f"=======================\n\n")
        
        for split_name, video_list in splits.items():
            if not video_list: continue
            
            f.write(f"[{split_name.upper()}] - {len(video_list)} videos\n")
            f.write("-" * 40 + "\n")
            
            # Sort for readability
            sorted_videos = sorted(video_list, key=lambda x: x['name'])
            
            for v in sorted_videos:
                # Format: Source | Video Name | Frame Count
                source = v.get('source_type', 'unknown').ljust(10)
                name = v['name'].ljust(30)
                frame_count = len(v.get('frames', []))
                
                f.write(f"{source} | {name} | {frame_count} frames\n")
            
            f.write("\n")

# --- EXPORTERS ---

def save_coco_format(split_name, video_data, output_root):
    """Generates Standard COCO-Video JSON."""
    anno_dir = output_root / "annotations"
    anno_dir.mkdir(exist_ok=True)
    
    # Check if we should skip
    if not video_data: return

    print(f"   - Building {split_name}.json (COCO)...")
    
    CLASS_MAP = UNIFIED_CLASS_MAP
    coco = {
        "videos": [], "images": [], "annotations": [],
        "categories": COCO_CATEGORIES
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
            
            # Use auxiliary path if available (VisDrone), else standard BDD naming
            if 'aux_path' in frame_data:
                # visdrone/video/0001.jpg -> video-0001.jpg for flat output
                suffix = os.path.basename(frame_data['aux_path'])
                img_file = f"{video_name}-{suffix}"
            else:
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
    
    CLASS_MAP = UNIFIED_CLASS_MAP
    
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
                    t_id_int = abs(hash((v_name, obj['id']))) % 100000 
                    
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
    
    print(f"⚙️  Config: Seed {SEED} | Step: {FRAME_STEP} | Formats: {EXPORT_FORMATS}")

    # 3. Get Labels (BDD)
    bdd_cfg = cfg.get('bdd', {})
    bdd_enabled = bdd_cfg.get('enabled', True)
    
    if bdd_enabled:
        labels_url = bdd_cfg['labels_url']
        labels_zip_name = os.path.basename(labels_url)
        labels_zip = data_dir / labels_zip_name
        download_file(labels_url, labels_zip)

    # 4. Pick Videos
    print("🎲 Selecting random videos...")
    parsed_videos = []
    
    # --- BDD Source ---
    if bdd_enabled:
        with zipfile.ZipFile(labels_zip, 'r') as z_lbl:
            all_json = [f for f in z_lbl.namelist() if f.endswith(".json") and "train" in f]
            all_json.sort() # Deterministic Sort
            
            random.seed(SEED)
            bdd_num = bdd_cfg.get('num_videos', 10) # Default to 10 if missing
            if len(all_json) < bdd_num:
                selected_files = all_json
            else:
                selected_files = random.sample(all_json, bdd_num)
            
            for f in selected_files:
                content = z_lbl.read(f)
                raw_data = json.loads(content)
                
                # Defensive checks
                if not isinstance(raw_data, list) or len(raw_data) == 0: continue
                if not isinstance(raw_data[0], dict): continue
                
                first = raw_data[0]
                v_name = first.get('videoName', os.path.basename(f).replace('.json', ''))
                
                if 'labels' not in first and 'objects' not in first:
                    print(f"❌ Missing labels in {f}")
                    continue

                parsed_videos.append({
                    "name": v_name, 
                    "frames": raw_data, 
                    "source_type": "bdd"
                })

    # --- VisDrone Source ---
    vis_cfg = cfg.get('visdrone', {})
    if vis_cfg.get('enabled', False):
        print("🚁 Processing VisDrone metadata...")
        lbl_zip_path = Path(vis_cfg.get('labels_zip', ''))
        img_zip_path = Path(vis_cfg.get('images_zip', ''))
        
        if not lbl_zip_path.exists() or not img_zip_path.exists():
            print("\n❌ Error: VisDrone is enabled but zip files are missing!")
            print(f"   - Missing: {lbl_zip_path if not lbl_zip_path.exists() else ''}")
            print(f"   - Missing: {img_zip_path if not img_zip_path.exists() else ''}")
            print("   👉 Please download them to the 'data/' folder or disable [visdrone] in config.")
            sys.exit(1)
        
        if lbl_zip_path.exists():
            with zipfile.ZipFile(lbl_zip_path, 'r') as z_vis:
                txt_files = sorted([f for f in z_vis.namelist() if f.endswith(".txt") and "__MACOSX" not in f])
                random.seed(SEED)
                vis_num = vis_cfg.get('num_videos', 10)
                selected_vis = random.sample(txt_files, min(len(txt_files), vis_num))
                
                CAT_MAP = VISDRONE_RAW_MAP
                
                for f in selected_vis:
                    lines = z_vis.read(f).decode('utf-8').strip().split('\n')
                    seq_name = Path(f).stem
                    
                    frame_dict = {}
                    for line in lines:
                        p = line.split(',')
                        if len(p)<8: continue
                        cat_id = int(p[7])
                        if cat_id not in CAT_MAP: continue
                        
                        f_idx = int(p[0])
                        obj = {
                            "category": CAT_MAP[cat_id],
                            "id": int(p[1]),
                            "box2d": {"x1":int(p[2]), "y1":int(p[3]), "x2":int(p[2])+int(p[4]), "y2":int(p[3])+int(p[5])}
                        }
                        if f_idx not in frame_dict: frame_dict[f_idx] = []
                        frame_dict[f_idx].append(obj)
                    
                    # Convert to BDD format
                    vis_frames = []
                    if frame_dict:
                        max_frame = max(frame_dict.keys())
                        for i in range(1, max_frame+1):
                            vis_frames.append({
                                "frameIndex": i-1,
                                "videoName": seq_name,
                                "labels": frame_dict.get(i, []),
                                "aux_path": f"{seq_name}/{i:07d}.jpg"
                            })
                        
                        parsed_videos.append({
                            "name": seq_name,
                            "frames": vis_frames,
                            "source_type": "visdrone",
                            "zip_path": vis_cfg.get('images_zip')
                        })

    if len(parsed_videos) == 0:
        print("\n❌ No videos selected! Check that:")
        print("   1. [bdd] or [visdrone] is enabled in config.toml")
        print("   2. You have downloaded the required zip files for enabled sources.")
        sys.exit(1)

    print(f"✅ Parsed {len(parsed_videos)} valid videos.")

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
    
    # Also map video -> source info for streaming
    video_source_map = {v['name']: v for v in parsed_videos}

    print(f"📊 Final Split: {len(train_set)} Train, {len(val_set)} Val, {len(test_set)} Test")

    # 6. Stream Images
    print(f"☁️  Checking Cache & Streaming Frames...")
    
    # Define Download Sources
    download_queue = []
    
    # BDD Source
    if bdd_enabled:
        download_queue.append({
            "type": "bdd", 
            "opener": RemoteZip, 
            "path": bdd_cfg['images_url']
        })
        
    # VisDrone Source
    if vis_cfg.get('enabled', False):
        download_queue.append({
            "type": "visdrone", 
            "opener": zipfile.ZipFile, 
            "path": vis_cfg.get('images_zip')
        })

    dl_count, cache_count = 0, 0

    for source in download_queue:
        try:
            # Check if source file exists for local zips
            if source['type'] == 'visdrone' and not Path(source['path']).exists():
                print(f"⚠️  Skipping VisDrone images: {source['path']} not found")
                continue

            with source['opener'](source['path']) as z:
                all_files = z.namelist()
                files_to_process = []
                
                # Filter files for this source
                for filename in all_files:
                    if not filename.endswith('.jpg'): continue
                    
                    parts = filename.split('/')
                    # BDD: .../video/frame.jpg, VisDrone: .../video/frame.jpg
                    if len(parts) >= 2:
                        v_name = parts[-2]
                        
                        # Check if this video is needed AND belongs to this source
                        if v_name in video_to_split:
                            v_info = video_source_map[v_name]
                            if v_info['source_type'] == source['type']:
                                files_to_process.append((filename, video_to_split[v_name], v_name))
                
                if not files_to_process: continue
                
                # Process files
                for file_path, split, v_name in tqdm(files_to_process, desc=f"Extracting {source['type']}"):
                    try:
                        fname = os.path.basename(file_path)
                        
                        # Robust Frame Index Logic
                        if '-' in fname: frame_str = fname.replace('.jpg', '').split('-')[-1]
                        else: frame_str = fname.replace('.jpg', '') # VisDrone usually just number
                        
                        try: frame_idx = int(frame_str) - 1
                        except: frame_idx = 0
                        
                        if frame_idx % FRAME_STEP != 0: continue 

                        # Cache Logic (Prefix with source to avoid collisions)
                        cache_fname = f"{source['type']}_{v_name}_{fname}"
                        cached_file = cache_dir / cache_fname
                        
                        if not cached_file.exists():
                            z.extract(file_path, path=data_dir)
                            extracted_path = data_dir / file_path
                            shutil.move(str(extracted_path), str(cached_file))
                            dl_count += 1
                        else:
                            cache_count += 1
                        
                        # --- FIX: Filename flattening logic ---
                        # If the filename already contains the video name (BDD), don't prepend it again.
                        if fname.startswith(v_name):
                            out_name = fname
                        else:
                            out_name = f"{v_name}-{fname}"
                        # -------------------------------------
                        
                        target = img_roots[split] / out_name
                        if not target.exists():
                            shutil.copy2(str(cached_file), str(target))

                    except Exception as e: pass
            
            # Cleanup temp extract folder for this source
            if source['type'] == 'bdd' and (data_dir / "bdd100k").exists():
                shutil.rmtree(data_dir / "bdd100k")
                
        except Exception as e:
            print(f"❌ Error streaming from {source['type']}: {e}")
            if source['type'] == 'bdd': sys.exit(1)

    print(f"✅ Images: {dl_count} Downloaded, {cache_count} Cached.")

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
    
    # Save the human-readable manifest
    splits_dict = {"train": train_set, "val": val_set, "test": test_set}
    save_manifest(splits_dict, out_dir)
    
    # Dump manifest to console
    print("-" * 20 + " Manifest " + "-" * 20)
    with open(out_dir / "manifest.txt", "r") as f:
        print(f.read())
    print("-" * 50)

    print(f"🚀 Done! Mini-BDD ready at: {out_dir}")

if __name__ == "__main__":
    try:
        build_mini_dataset()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Progress saved in 'data/image_cache'.")
        sys.exit(0)