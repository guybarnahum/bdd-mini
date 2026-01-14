import os
import json
import random
import sys
import shutil
import zipfile
import re
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
    from botocore.exceptions import NoCredentialsError, ClientError
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

def load_config():
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file '{CONFIG_FILE}' not found.")
        sys.exit(1)
    with open(CONFIG_FILE, "rb") as f:
        return toml.load(f)

cfg = load_config()

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
TRAIN_RATIO = cfg['dataset'].get('train_ratio', 0.70)
VAL_RATIO   = cfg['dataset'].get('val_ratio', 0.15)
TEST_RATIO  = cfg['dataset'].get('test_ratio', 0.15)

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

def check_budget(current_frames, target_budget):
    return current_frames < target_budget

# --- SELECTION STRATEGIES (Refactored) ---

def select_bdd_videos(bdd_cfg, labels_zip):
    if not bdd_cfg.get('enabled', False): return []
    
    budget = bdd_cfg.get('frame_budget', 2000)
    collected_frames = 0
    selected_videos = []

    print(f"⚖️  Selecting BDD videos (Budget: {budget} frames)...")
    
    with zipfile.ZipFile(labels_zip, 'r') as z_lbl:
        all_json = sorted([f for f in z_lbl.namelist() if f.endswith(".json") and "train" in f])
        random.seed(SEED)
        random.shuffle(all_json)
        
        pbar = tqdm(total=budget, desc="BDD Selection", unit="fr")
        
        for f in all_json:
            if not check_budget(collected_frames, budget): break
            
            content = z_lbl.read(f)
            raw_data = json.loads(content)
            if not raw_data or not isinstance(raw_data[0], dict): continue
            
            # Calculate effective frames
            effective_len = len([x for i, x in enumerate(raw_data) if i % FRAME_STEP == 0])
            
            v_name = raw_data[0].get('videoName', os.path.basename(f).replace('.json', ''))
            
            selected_videos.append({
                "name": v_name, 
                "frames": raw_data, 
                "source_type": "bdd"
            })
            
            collected_frames += effective_len
            pbar.update(effective_len)
        pbar.close()
        
    print(f"   ✅ BDD: {len(selected_videos)} videos selected ({collected_frames} frames)")
    return selected_videos

def select_dancetrack_videos(dt_cfg):
    if not dt_cfg.get('enabled', False): return []

    budget = dt_cfg.get('frame_budget', 2000)
    collected_frames = 0
    selected_videos = []
    
    dt_urls = dt_cfg['images_url'] if isinstance(dt_cfg['images_url'], list) else [dt_cfg['images_url']]
    
    print(f"⚖️  Selecting DanceTrack videos (Budget: {budget} frames)...")
    pbar = tqdm(total=budget, desc="DT Selection", unit="fr")

    for zip_url in dt_urls:
        if not check_budget(collected_frames, budget): break
        
        actual_url = get_s3_presigned_url(zip_url) if zip_url.startswith("s3://") else zip_url
        try:
            with RemoteZip(actual_url) as z_dt:
                seq_gt_files = [f for f in z_dt.namelist() if f.endswith('gt/gt.txt')]
                random.seed(SEED)
                random.shuffle(seq_gt_files)
                
                for gt_f in seq_gt_files:
                    if not check_budget(collected_frames, budget): break
                    
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

                    effective_len = len([i for i in range(1, max_frame+1) if (i-1) % FRAME_STEP == 0])
                    seq_name = gt_f.split('/')[-3] if len(gt_f.split('/')) > 2 else "unknown"
                    
                    selected_videos.append({
                        "name": seq_name,
                        "frames": [{"frameIndex": i-1, "videoName": seq_name, "labels": frame_dict.get(i, [])} for i in range(1, max_frame+1)],
                        "source_type": "dancetrack",
                        "zip_url": zip_url
                    })
                    
                    collected_frames += effective_len
                    pbar.update(effective_len)
        except Exception as e: 
            print(f"⚠️  Skipping DT Zip {zip_url}: {e}")
            
    pbar.close()
    print(f"   ✅ DanceTrack: {len(selected_videos)} videos selected ({collected_frames} frames)")
    return selected_videos

def select_visdrone_videos(vis_cfg):
    if not vis_cfg.get('enabled', False): return []
    
    budget = vis_cfg.get('frame_budget', 2000)
    collected_frames = 0
    selected_videos = []
    
    lbl_zip_path = Path(vis_cfg.get('labels_zip', ''))
    if not lbl_zip_path.exists(): 
        print("❌ VisDrone enabled but zip missing"); sys.exit(1)

    print(f"⚖️  Selecting VisDrone videos (Budget: {budget} frames)...")
    
    with zipfile.ZipFile(lbl_zip_path, 'r') as z_vis:
        txt_files = sorted([f for f in z_vis.namelist() if f.endswith(".txt") and "__MACOSX" not in f])
        random.seed(SEED)
        random.shuffle(txt_files)
        
        pbar = tqdm(total=budget, desc="VisDrone Selection", unit="fr")
        
        for f in txt_files:
            if not check_budget(collected_frames, budget): break

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
            
            effective_len = len([i for i in range(1, max_f+1) if (i-1) % FRAME_STEP == 0])
            seq_name = Path(f).stem
            
            selected_videos.append({
                "name": seq_name,
                "frames": [{"frameIndex": i-1, "videoName": seq_name, "labels": frame_dict.get(i, [])} for i in range(1, max_f+1)],
                "source_type": "visdrone"
            })
            
            collected_frames += effective_len
            pbar.update(effective_len)
        pbar.close()

    print(f"   ✅ VisDrone: {len(selected_videos)} videos selected ({collected_frames} frames)")
    return selected_videos

# --- EXPORTERS (Shortened for brevity, logic identical) ---

def save_coco_format(split_name, video_data, output_root):
    anno_dir = output_root / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    if not video_data: return
    print(f"   - Building {split_name}.json (COCO)...")
    
    coco = {"videos": [], "images": [], "annotations": [], "categories": COCO_CATEGORIES}
    global_img_id, global_ann_id, global_track_id, track_map = 1, 1, 1, {}

    for video_idx, vid_data in enumerate(video_data):
        video_name = vid_data['name']
        coco["videos"].append({"id": video_idx + 1, "file_name": video_name})
        
        frames = sorted(vid_data['frames'], key=lambda x: x.get('frameIndex', 0))
        valid_frames = [f for f in frames if f.get('frameIndex', 0) % FRAME_STEP == 0]
        
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
    seq_dir = output_root / split_name / v_name
    gt_dir = seq_dir / "gt"; gt_dir.mkdir(parents=True, exist_ok=True)
    
    frames = sorted(vid_data['frames'], key=lambda x: x.get('frameIndex', 0))
    valid_frames = [f for f in frames if f.get('frameIndex', 0) % FRAME_STEP == 0]
    
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
        f_ini.write(f"[Sequence]\nname={v_name}\nimDir=img1\nframeRate={30/FRAME_STEP}\nseqLength={len(valid_frames)}\nimWidth=1280\nimHeight=720\nimExt=.jpg\n")

def save_seqmap(split_name, video_data, output_root):
    if not video_data: return
    seqmap_path = output_root / f"{split_name}_seqmap.txt"
    print(f"   - Building {seqmap_path.name}...")
    with open(seqmap_path, 'w') as f:
        f.write("name\n")
        for v in sorted(video_data, key=lambda x: x['name']): f.write(f"{v['name']}\n")

def save_manifest(splits, output_root):
    with open(output_root / "manifest.txt", 'w') as f:
        f.write(f"BDD-Mini Build Manifest\n=======================\n\n")
        for split_name, video_list in splits.items():
            if not video_list: continue
            f.write(f"[{split_name.upper()}] - {len(video_list)} videos\n" + "-" * 40 + "\n")
            for v in sorted(video_list, key=lambda x: x['name']):
                f.write(f"{v.get('source_type', 'unknown').ljust(10)} | {v['name'].ljust(30)} | {len(v.get('frames', []))} frames\n")
            f.write("\n")

# --- MAIN BUILDER ---

def build_mini_dataset():
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 0.001:
        print(f"⚠️ Warning: Ratios sum to {(TRAIN_RATIO + VAL_RATIO + TEST_RATIO):.2f}, not 1.0.")

    data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
    cache_dir = data_dir / "image_cache"; cache_dir.mkdir(exist_ok=True)
    out_dir = OUTPUT_DIR
    print(f"⚙️  Config: Seed {SEED} | Step: {FRAME_STEP}")

    # 1. Prepare BDD Labels if needed
    labels_zip = None
    if cfg.get('bdd', {}).get('enabled', True):
        labels_url = cfg['bdd']['labels_url']
        labels_zip = data_dir / (labels_url.split("/")[-1] if labels_url.startswith("s3://") else os.path.basename(labels_url))
        download_file(labels_url, labels_zip)

    # 2. Select Videos (Using new helper functions)
    parsed_videos = []
    parsed_videos.extend(select_bdd_videos(cfg.get('bdd', {}), labels_zip))
    parsed_videos.extend(select_dancetrack_videos(cfg.get('dancetrack', {})))
    parsed_videos.extend(select_visdrone_videos(cfg.get('visdrone', {})))

    if not parsed_videos: print("\n❌ No videos selected!"); sys.exit(1)
    
    # 3. Apply Splits
    random.shuffle(parsed_videos)
    n_train = int(len(parsed_videos) * TRAIN_RATIO)
    n_val = int(len(parsed_videos) * VAL_RATIO)
    train_set, val_set, test_set = parsed_videos[:n_train], parsed_videos[n_train:n_train+n_val], parsed_videos[n_train+n_val:]
    
    video_to_split = {v['name']: s for s, lst in [("train", train_set), ("val", val_set), ("test", test_set)] for v in lst}
    video_source_map = {v['name']: v for v in parsed_videos}
    print(f"📊 Final Split: {len(train_set)} Train, {len(val_set)} Val, {len(test_set)} Test")

    # 4. Stream Images
    print(f"☁️  Streaming Frames to Universal MOT Structure...")
    download_queue = []
    
    # Collect Source URLS
    if cfg.get('bdd', {}).get('enabled', False):
        urls = cfg['bdd']['images_url'] if isinstance(cfg['bdd']['images_url'], list) else [cfg['bdd']['images_url']]
        for p in urls:
            url = get_s3_presigned_url(p) if p.startswith("s3://") else p
            if url: download_queue.append({"type": "bdd", "opener": RemoteZip if "http" in url else zipfile.ZipFile, "path": url})
            
    if cfg.get('visdrone', {}).get('enabled', False):
        download_queue.append({"type": "visdrone", "opener": zipfile.ZipFile, "path": cfg['visdrone']['images_zip']})

    if cfg.get('dancetrack', {}).get('enabled', False):
        urls = cfg['dancetrack']['images_url'] if isinstance(cfg['dancetrack']['images_url'], list) else [cfg['dancetrack']['images_url']]
        for p in urls:
            url = get_s3_presigned_url(p) if p.startswith("s3://") else p
            if url: download_queue.append({"type": "dancetrack", "opener": RemoteZip, "path": url})

    # Process Download Queue
    dl_count, cache_count = 0, 0
    for source in download_queue:
        try:
            if source['type'] == 'visdrone' and not Path(source['path']).exists(): continue
            with source['opener'](source['path']) as z:
                all_files = z.namelist()
                files_to_process = []
                
                # Check for files belonging to selected videos
                possible_videos = [v for v in video_source_map if video_source_map[v]['source_type'] == source['type']]
                
                for filename in all_files:
                    if not filename.endswith('.jpg'): continue
                    
                    found_video = None
                    path_parts = filename.split('/')
                    for v_name in possible_videos:
                        if v_name in path_parts:
                            found_video = v_name
                            break
                    
                    if found_video:
                        files_to_process.append((filename, video_to_split[found_video], found_video))
                
                if not files_to_process: continue
                
                for file_path, split, v_name in tqdm(files_to_process, desc=f"Extracting {source['type']}"):
                    try:
                        fname = os.path.basename(file_path)
                        digits = re.findall(r'\d+', fname)
                        if not digits: continue
                        frame_idx = int(digits[-1]) - 1 
                        
                        if frame_idx % FRAME_STEP != 0: continue 

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
                        tgt = target_dir / f"{(frame_idx // FRAME_STEP) + 1:08d}.jpg"
                        if not tgt.exists(): shutil.copy2(str(cached_file), str(tgt))

                    except Exception: pass
            
            for debris in ["bdd100k", "train", "val"]:
                if (data_dir / debris).exists(): shutil.rmtree(data_dir / debris)
                
        except Exception as e:
            print(f"❌ Error streaming from {source['type']}: {e}")
            if source['type'] == 'bdd': sys.exit(1)

    print(f"✅ Images: {dl_count} Downloaded, {cache_count} Cached.")
    
    print("📝 Generating Universal Metadata...")
    for split, data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        save_coco_format(split, data, out_dir)
        for vid in data: save_mot_gt_file(split, vid, out_dir)
        save_seqmap(split, data, out_dir)
    
    save_manifest({"train": train_set, "val": val_set, "test": test_set}, out_dir)
    print("-" * 50); print(open(out_dir / "manifest.txt").read()); print("-" * 50)
    print(f"🚀 Done! Universal Mini-BDD ready at: {out_dir}")

if __name__ == "__main__":
    try: build_mini_dataset()
    except KeyboardInterrupt: print("\n🛑 Interrupted."); sys.exit(0)