import os
import json
import random
import sys
import shutil
import zipfile
import re
import argparse
from pathlib import Path
from tqdm import tqdm

# --- Import Dependencies ---
try:
    from remotezip import RemoteZip
except ImportError:
    print("❌ Missing 'remotezip'. Please run: ./setup.sh")
    sys.exit(1)

try:
    import boto3
except ImportError:
    print("❌ Missing 'boto3'. For S3 support, run: pip install boto3")

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
MANIFEST_FILE = "manifest.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    with open(CONFIG_FILE, "rb") as f:
        return toml.load(f)

def load_manifest():
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Manifest file '{MANIFEST_FILE}' not found. Run 'python manifest.py' first.")
        sys.exit(1)
    with open(MANIFEST_FILE, "r") as f:
        return json.load(f)

cfg = load_config()
manifest_data = load_manifest()

# --- GLOBAL CLASS MAPPINGS ---
UNIFIED_CLASS_MAP = {
    "pedestrian": 1, "rider": 1, "car": 2, "truck": 2, 
    "bus": 2, "train": 2, "vehicle": 2
}

COCO_CATEGORIES = [
    {"id": 1, "name": "person"}, 
    {"id": 2, "name": "vehicle"}
]

VISDRONE_RAW_MAP = {
    1: "pedestrian", 2: "pedestrian", 4: "car", 
    5: "car", 6: "car", 9: "car"
}

DANCETRACK_RAW_MAP = { 1: "pedestrian" }

# Load Settings
SEED = cfg['dataset']['seed']
OUTPUT_DIR = Path(cfg['dataset']['output_dir'])
FRAME_STEP = cfg['dataset'].get('frame_step', 5)

# Build Lookup Sets for Manifest
# Format: { "video_name": "val" } or { "video_name": "test" }
LOCKED_VIDEOS = {}
for entry in manifest_data.get('val', []):
    LOCKED_VIDEOS[entry['name']] = 'val'
for entry in manifest_data.get('test', []):
    LOCKED_VIDEOS[entry['name']] = 'test'

# --- HELPER FUNCTIONS ---

def get_s3_presigned_url(s3_path):
    try:
        parts = s3_path.replace("s3://", "").split("/", 1)
        s3 = boto3.client('s3')
        return s3.generate_presigned_url('get_object', Params={'Bucket': parts[0], 'Key': parts[1]}, ExpiresIn=3600)
    except Exception as e:
        print(f"❌ Error generating S3 URL for {s3_path}: {e}")
        return None

def download_file(url, dest_path):
    if url.startswith("s3://"):
        print(f"⬇️  Downloading Labels from S3: {url}")
        try:
            parts = url.replace("s3://", "").split("/", 1)
            boto3.client('s3').download_file(parts[0], parts[1], str(dest_path))
            return
        except Exception:
             print("❌ S3 Download failed"); sys.exit(1)

    import requests
    if dest_path.exists():
        print(f"✅ Found existing labels at {dest_path}")
        return
    print(f"⬇️  Downloading Labels...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)): f.write(chunk)


# --- SELECTION STRATEGIES (Manifest Aware) ---

def select_bdd_videos(bdd_cfg, labels_zip):
    if not bdd_cfg.get('enabled', False): return []
    
    # We collect ALL valid videos first, then split later
    selected_videos = []

    print(f"⚖️  Scanning BDD videos...")
    
    with zipfile.ZipFile(labels_zip, 'r') as z_lbl:
        all_json = sorted([f for f in z_lbl.namelist() if f.endswith(".json") and "train" in f])
        
        # We process ALL json files to find our manifest targets + candidates
        pbar = tqdm(total=len(all_json), desc="BDD Scan", unit="file")
        
        for f in all_json:
            content = z_lbl.read(f)
            raw_data = json.loads(content)
            if not raw_data or not isinstance(raw_data[0], dict): 
                pbar.update(1); continue
            
            v_name = raw_data[0].get('videoName', os.path.basename(f).replace('.json', ''))
            
            # Determine Frame Step based on split
            # Test = Full FPS (step=1), Val/Train = Reduced FPS (step=FRAME_STEP)
            split = LOCKED_VIDEOS.get(v_name, 'train')
            step = 1 if split == 'test' else FRAME_STEP
            
            selected_videos.append({
                "name": v_name, 
                "frames": raw_data, 
                "source_type": "bdd",
                "split": split,
                "step": step
            })
            pbar.update(1)
        pbar.close()
        
    print(f"   ✅ BDD: Found {len(selected_videos)} total videos")
    return selected_videos

def select_dancetrack_videos(dt_cfg):
    if not dt_cfg.get('enabled', False): return []

    selected_videos = []
    dt_urls = dt_cfg['images_url'] if isinstance(dt_cfg['images_url'], list) else [dt_cfg['images_url']]
    
    print(f"⚖️  Scanning DanceTrack videos...")
    
    for zip_url in dt_urls:
        actual_url = get_s3_presigned_url(zip_url) if zip_url.startswith("s3://") else zip_url
        try:
            with RemoteZip(actual_url) as z_dt:
                seq_gt_files = [f for f in z_dt.namelist() if f.endswith('gt/gt.txt')]
                
                for gt_f in tqdm(seq_gt_files, desc="DT Scan"):
                    lines = z_dt.read(gt_f).decode('utf-8').strip().split('\n')
                    
                    max_frame = 0
                    frame_dict = {}
                    valid_seq = False
                    
                    for line in lines:
                        p = line.split(',')
                        if len(p) < 8 or int(float(p[7])) not in DANCETRACK_RAW_MAP: continue
                        
                        valid_seq = True
                        f_idx = int(p[0])
                        max_frame = max(max_frame, f_idx)
                        
                        obj = {
                            "category": DANCETRACK_RAW_MAP[int(float(p[7]))], 
                            "id": int(p[1]), 
                            "box2d": {
                                "x1": float(p[2]), "y1": float(p[3]), 
                                "x2": float(p[2]) + float(p[4]), "y2": float(p[3]) + float(p[5])
                            }
                        }
                        frame_dict.setdefault(f_idx, []).append(obj)
                    
                    if not valid_seq: continue

                    seq_name = gt_f.split('/')[-3] if len(gt_f.split('/')) > 2 else "unknown"
                    
                    split = LOCKED_VIDEOS.get(seq_name, 'train')
                    step = 1 if split == 'test' else FRAME_STEP
                    
                    selected_videos.append({
                        "name": seq_name,
                        "frames": [{"frameIndex": i-1, "videoName": seq_name, "labels": frame_dict.get(i, [])} for i in range(1, max_frame+1)],
                        "source_type": "dancetrack",
                        "zip_url": zip_url,
                        "split": split,
                        "step": step
                    })
        except Exception as e: 
            print(f"⚠️  Skipping DT Zip {zip_url}: {e}")
            
    print(f"   ✅ DanceTrack: Found {len(selected_videos)} total videos")
    return selected_videos

def select_visdrone_videos(vis_cfg):
    if not vis_cfg.get('enabled', False): return []
    
    selected_videos = []
    lbl_zip_path = Path(vis_cfg.get('labels_zip', ''))
    if not lbl_zip_path.exists(): 
        print("❌ VisDrone enabled but zip missing"); sys.exit(1)

    print(f"⚖️  Scanning VisDrone videos...")
    
    with zipfile.ZipFile(lbl_zip_path, 'r') as z_vis:
        txt_files = sorted([f for f in z_vis.namelist() if f.endswith(".txt") and "__MACOSX" not in f])
        
        for f in tqdm(txt_files, desc="VisDrone Scan"):
            lines = z_vis.read(f).decode('utf-8').strip().split('\n')
            max_f = 0
            frame_dict = {}
            valid = False
            
            for line in lines:
                p = line.split(',')
                if len(p)<8 or int(p[7]) not in VISDRONE_RAW_MAP: continue
                
                valid = True
                f_idx = int(p[0])
                max_f = max(max_f, f_idx)
                
                obj = {
                    "category": VISDRONE_RAW_MAP[int(p[7])], 
                    "id": int(p[1]), 
                    "box2d": {
                        "x1":int(p[2]), "y1":int(p[3]), 
                        "x2":int(p[2])+int(p[4]), "y2":int(p[3])+int(p[5])
                    }
                }
                frame_dict.setdefault(f_idx, []).append(obj)
            
            if not valid: continue
            
            seq_name = Path(f).stem
            
            split = LOCKED_VIDEOS.get(seq_name, 'train')
            step = 1 if split == 'test' else FRAME_STEP
            
            selected_videos.append({
                "name": seq_name,
                "frames": [{"frameIndex": i-1, "videoName": seq_name, "labels": frame_dict.get(i, [])} for i in range(1, max_f+1)],
                "source_type": "visdrone",
                "split": split,
                "step": step
            })

    print(f"   ✅ VisDrone: Found {len(selected_videos)} total videos")
    return selected_videos

# --- EXPORTERS ---

def save_coco_format(split_name, video_data, output_root):
    anno_dir = output_root / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    if not video_data: return
    print(f"   - Building {split_name}.json (COCO)...")
    
    coco = {"videos": [], "images": [], "annotations": [], "categories": COCO_CATEGORIES}
    global_img_id, global_ann_id, global_track_id, track_map = 1, 1, 1, {}

    for video_idx, vid_data in enumerate(video_data):
        video_name = vid_data['name']
        current_step = vid_data['step'] # Use video-specific step (1 or 5)
        
        coco["videos"].append({"id": video_idx + 1, "file_name": video_name})
        
        frames = sorted(vid_data['frames'], key=lambda x: x.get('frameIndex', 0))
        # Filter logic using dynamic step
        valid_frames = [f for f in frames if f.get('frameIndex', 0) % current_step == 0]
        
        for local_idx, frame_data in enumerate(valid_frames, start=1):
            coco["images"].append({
                "id": global_img_id, "video_id": video_idx + 1,
                "file_name": f"{video_name}/img1/{local_idx:08d}.jpg",
                "frame_id": local_idx, "height": 720, "width": 1280
            })
            
            for obj in frame_data.get('labels', frame_data.get('objects', [])):
                if obj['category'] not in UNIFIED_CLASS_MAP: continue
                cat_id = UNIFIED_CLASS_MAP[obj['category']]
                t_key = (video_name, obj['id'])
                if t_key not in track_map: track_map[t_key] = global_track_id; global_track_id += 1
                
                if 'box2d' in obj:
                    x1, y1 = obj['box2d']['x1'], obj['box2d']['y1']
                    w, h = obj['box2d']['x2'] - x1, obj['box2d']['y2'] - y1
                    coco["annotations"].append({
                        "id": global_ann_id, "image_id": global_img_id, "category_id": cat_id,
                        "track_id": track_map[t_key], "bbox": [x1, y1, w, h], "area": w*h,
                        "iscrowd": 0, "video_id": video_idx + 1
                    })
                    global_ann_id += 1
            global_img_id += 1
    with open(anno_dir / f"{split_name}.json", 'w') as f: json.dump(coco, f)

def save_mot_gt_file(split_name, vid_data, output_root):
    v_name = vid_data['name']
    current_step = vid_data['step']
    
    seq_dir = output_root / split_name / v_name
    gt_dir = seq_dir / "gt"; gt_dir.mkdir(parents=True, exist_ok=True)
    
    frames = sorted(vid_data['frames'], key=lambda x: x.get('frameIndex', 0))
    valid_frames = [f for f in frames if f.get('frameIndex', 0) % current_step == 0]
    
    with open(gt_dir / "gt.txt", 'w') as f_gt:
        for local_frame_idx, frame_data in enumerate(valid_frames, start=1):
            for obj in frame_data.get('labels', frame_data.get('objects', [])):
                if obj['category'] not in UNIFIED_CLASS_MAP: continue
                cls_id = UNIFIED_CLASS_MAP[obj['category']]
                t_id_int = abs(hash((v_name, obj['id']))) % 100000 
                if 'box2d' in obj:
                    x1, y1 = obj['box2d']['x1'], obj['box2d']['y1']
                    w, h = obj['box2d']['x2'] - x1, obj['box2d']['y2'] - y1
                    f_gt.write(f"{local_frame_idx},{t_id_int},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{cls_id},1\n")
    with open(seq_dir / "seqinfo.ini", 'w') as f_ini:
        f_ini.write(f"[Sequence]\nname={v_name}\nimDir=img1\nframeRate={30/current_step}\nseqLength={len(valid_frames)}\nimWidth=1280\nimHeight=720\nimExt=.jpg\n")

def save_seqmap(split_name, video_data, output_root):
    if not video_data: return
    seqmap_path = output_root / f"{split_name}_seqmap.txt"
    print(f"   - Building {seqmap_path.name}...")
    with open(seqmap_path, 'w') as f:
        f.write("name\n")
        for v in sorted(video_data, key=lambda x: x['name']): f.write(f"{v['name']}\n")

def save_manifest_log(splits, output_root):
    with open(output_root / "manifest_log.txt", 'w') as f:
        f.write(f"BDD-Mini Build Log (Manifest Aware)\n===================================\n\n")
        for split_name, video_list in splits.items():
            if not video_list: continue
            f.write(f"[{split_name.upper()}] - {len(video_list)} videos\n" + "-" * 40 + "\n")
            for v in sorted(video_list, key=lambda x: x['name']):
                step_info = "FULL" if v['step'] == 1 else "SKIP-5"
                f.write(f"{v.get('source_type', 'unknown').ljust(10)} | {v['name'].ljust(30)} | {step_info} | {len(v.get('frames', []))} raw frames\n")
            f.write("\n")

# --- MAIN BUILDER ---

def build_mini_dataset(target_split=None, video_limit=20):
    data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
    cache_dir = data_dir / "image_cache"; cache_dir.mkdir(exist_ok=True)
    out_dir = OUTPUT_DIR
    print(f"⚙️  Config: Seed {SEED} | Manifest: {MANIFEST_FILE}")
    
    if target_split:
        limit_str = f"{video_limit} videos" if video_limit > 0 else "ALL videos"
        print(f"🌟 EXCLUSIVE MODE: Generating ONLY '{target_split}' split with {limit_str}")

    # 🛑 SMART SELECTION: Only load what is strictly necessary based on the argument
    
    all_videos = []
    
    # Define flags for what needs to be loaded based on split
    need_bdd = (target_split == 'bdd') or (target_split is None and cfg.get('bdd', {}).get('enabled', True))
    need_dancetrack = (target_split == 'dancetrack') or (target_split is None and cfg.get('dancetrack', {}).get('enabled', True))
    need_visdrone = (target_split == 'visdrone') or (target_split is None and cfg.get('visdrone', {}).get('enabled', True))

    # 1. Prepare BDD Labels (Only if BDD is required)
    labels_zip = None
    if need_bdd:
        labels_url = cfg['bdd']['labels_url']
        labels_zip = data_dir / (labels_url.split("/")[-1] if labels_url.startswith("s3://") else os.path.basename(labels_url))
        download_file(labels_url, labels_zip)

    # 2. Select Candidates (Optimized)
    if need_bdd:
        all_videos.extend(select_bdd_videos(cfg.get('bdd', {}), labels_zip))
    
    if need_dancetrack:
        all_videos.extend(select_dancetrack_videos(cfg.get('dancetrack', {})))
        
    if need_visdrone:
        all_videos.extend(select_visdrone_videos(cfg.get('visdrone', {})))

    if not all_videos: print("\n❌ No videos selected!"); sys.exit(1)
    
    # --- LOGIC BRANCHING ---
    export_list = []
    
    if target_split:
        # === EXCLUSIVE SPLIT MODE ===
        # Filter videos strictly by source type
        special_set = [v for v in all_videos if v['source_type'] == target_split]
        
        if not special_set:
            print(f"\n❌ Error: No videos found for source '{target_split}'!")
            sys.exit(1)
            
        print(f"🔎 Found {len(special_set)} potential videos for '{target_split}'.")
        
        # Apply Limit if greater than 0
        if video_limit > 0 and len(special_set) > video_limit:
            print(f"✂️  Limiting to {video_limit} random videos.")
            random.seed(SEED)
            random.shuffle(special_set)
            special_set = special_set[:video_limit]
        
        # In exclusive mode, we create ONE split with the name of the source
        video_to_split = {v['name']: target_split for v in special_set}
        export_list.append((target_split, special_set))
        
        # For active name set, used in download loop
        video_source_map = {v['name']: v for v in special_set}
        
    else:
        # === STANDARD MODE (Train/Val/Test) ===
        # 3. Apply Manifest Split Logic
        train_set, val_set, test_set = [], [], []
        train_pool = []

        # Bin videos by manifest instructions
        for v in all_videos:
            if v['split'] == 'val':
                val_set.append(v)
            elif v['split'] == 'test':
                test_set.append(v)
            else:
                train_pool.append(v)

        # 4. Fill Training Budget from Pool (Source-Aware)
        random.seed(SEED)
        
        # Group training candidates by source
        train_candidates = {}
        for v in train_pool:
            src = v.get('source_type', 'unknown')
            train_candidates.setdefault(src, []).append(v)

        global_budget = cfg['dataset'].get('train_frame_budget', None)
        collected_frames = 0
        added_video_names = set()
        
        # A. Process Source-Specific Budgets First
        for source_name, candidates in train_candidates.items():
            source_cfg = cfg.get(source_name, {})
            specific_budget = source_cfg.get('frame_budget', None)
            
            if specific_budget is not None:
                print(f"⚖️  Applying specific budget for [{source_name}]: {specific_budget} frames")
                random.shuffle(candidates)
                source_collected = 0
                for v in candidates:
                    if source_collected >= specific_budget: break
                    
                    # Calculate effective frames
                    effective_len = len([x for i, x in enumerate(v['frames']) if i % FRAME_STEP == 0])
                    
                    train_set.append(v)
                    added_video_names.add(v['name'])
                    source_collected += effective_len
                    collected_frames += effective_len

        # B. Process Global Budget (Fallback for sources without specific limits)
        if global_budget is not None:
            print(f"⚖️  Filling remaining global budget: {global_budget} frames")
            remaining_candidates = [v for v in train_pool if v['name'] not in added_video_names]
            random.shuffle(remaining_candidates)
            
            for v in remaining_candidates:
                if collected_frames >= global_budget: break
                
                effective_len = len([x for i, x in enumerate(v['frames']) if i % FRAME_STEP == 0])
                train_set.append(v)
                collected_frames += effective_len

        video_to_split = {v['name']: s for s, lst in [("train", train_set), ("val", val_set), ("test", test_set)] for v in lst}
        video_source_map = {v['name']: v for v in all_videos}
        
        export_list = [("train", train_set), ("val", val_set), ("test", test_set)]
        
        print(f"📊 Final Distribution: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # 5. Stream Images
    print(f"☁️  Streaming Frames to Universal MOT Structure...")
    download_queue = []
    
    # Optimization: Only add enabled/relevant sources to download queue
    # Note: 'need_bdd' etc variables already calculated above can be reused
    
    if need_bdd:
        urls = cfg['bdd']['images_url'] if isinstance(cfg['bdd']['images_url'], list) else [cfg['bdd']['images_url']]
        for p in urls:
            url = get_s3_presigned_url(p) if p.startswith("s3://") else p
            if url: download_queue.append({"type": "bdd", "opener": RemoteZip if "http" in url else zipfile.ZipFile, "path": url})
            
    if need_visdrone:
        download_queue.append({"type": "visdrone", "opener": zipfile.ZipFile, "path": cfg['visdrone']['images_zip']})

    if need_dancetrack:
        urls = cfg['dancetrack']['images_url'] if isinstance(cfg['dancetrack']['images_url'], list) else [cfg['dancetrack']['images_url']]
        for p in urls:
            url = get_s3_presigned_url(p) if p.startswith("s3://") else p
            if url: download_queue.append({"type": "dancetrack", "opener": RemoteZip, "path": url})

    dl_count, cache_count = 0, 0
    
    # Pre-calculate active video set to skip unnecessary downloads
    active_video_names = set(video_to_split.keys())

    for source in download_queue:
        # Extra Safety: Skip non-target sources if in exclusive mode
        if target_split and source['type'] != target_split:
            continue
            
        try:
            if source['type'] == 'visdrone' and not Path(source['path']).exists(): continue
            with source['opener'](source['path']) as z:
                all_files = z.namelist()
                files_to_process = []
                
                possible_videos = [v for v in video_source_map if video_source_map[v]['source_type'] == source['type']]
                
                for filename in all_files:
                    if not filename.endswith('.jpg'): continue
                    
                    found_video = None
                    path_parts = filename.split('/')
                    for v_name in possible_videos:
                        if v_name in path_parts:
                            found_video = v_name
                            break
                    
                    # Only process if video is in our final selection
                    if found_video and found_video in active_video_names:
                        files_to_process.append((filename, video_to_split[found_video], found_video))
                
                if not files_to_process: continue
                
                for file_path, split, v_name in tqdm(files_to_process, desc=f"Extracting {source['type']}"):
                    try:
                        fname = os.path.basename(file_path)
                        digits = re.findall(r'\d+', fname)
                        if not digits: continue
                        frame_idx = int(digits[-1]) - 1 
                        
                        # Use video-specific step (1 for Test, 5 for others)
                        current_step = video_source_map[v_name]['step']
                        if frame_idx % current_step != 0: continue 

                        cache_fname = f"{source['type']}_{v_name}_{fname}"
                        cached_file = cache_dir / cache_fname
                        
                        if not cached_file.exists():
                            z.extract(file_path, path=data_dir)
                            shutil.move(str(data_dir / file_path), str(cached_file))
                            dl_count += 1
                        else:
                            cache_count += 1
                        
                        target_dir = out_dir / split / v_name / "img1"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        tgt = target_dir / f"{(frame_idx // current_step) + 1:08d}.jpg"
                        if not tgt.exists(): shutil.copy2(str(cached_file), str(tgt))

                    except Exception: pass
            
            for debris in ["bdd100k", "train", "val"]:
                if (data_dir / debris).exists(): shutil.rmtree(data_dir / debris)
                
        except Exception as e:
            print(f"❌ Error streaming from {source['type']}: {e}")
            if source['type'] == 'bdd': sys.exit(1)

    print(f"✅ Images: {dl_count} Downloaded, {cache_count} Cached.")
    
    print("📝 Generating Universal Metadata...")
    for split, data in export_list:
        save_coco_format(split, data, out_dir)
        for vid in data: save_mot_gt_file(split, vid, out_dir)
        save_seqmap(split, data, out_dir)
    
    print(f"📄 Archiving {CONFIG_FILE} and Manifest...")
    shutil.copy2(CONFIG_FILE, out_dir / CONFIG_FILE)
    shutil.copy2(MANIFEST_FILE, out_dir / MANIFEST_FILE)
    
    # Save log based on what we actually exported
    log_data = {split: data for split, data in export_list}
    save_manifest_log(log_data, out_dir)
    print(f"🚀 Done! Universal Mini-BDD ready at: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Mini-BDD Dataset")
    parser.add_argument("--split", type=str, choices=["visdrone", "dancetrack", "bdd"], default=None,
                        help="Exclusive Mode: Generate ONLY this split.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of videos to include in the exclusive split (Default: 20). Set 0 for ALL.")
    args = parser.parse_args()
    
    try: build_mini_dataset(target_split=args.split, video_limit=args.limit)
    except KeyboardInterrupt: print("\n🛑 Interrupted."); sys.exit(0)