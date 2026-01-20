import json
import random
import sys
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

# --- Dependency Check ---
try:
    from remotezip import RemoteZip, RemoteIOError
    import tomli as toml
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    print("❌ Missing dependencies. Run: pip install remotezip tomli boto3")
    sys.exit(1)

# --- CONFIG ---
CONFIG_FILE = "config.toml"
MANIFEST_FILE = "manifest.json"
ENV_FILE = ".env"
SEED = 42

# How many videos to lock for evaluation?
LOCK_COUNTS = {
    "bdd":        {"val": 15, "test": 15}, 
    "dancetrack": {"val": 10, "test": 10}, 
    "visdrone":   {"val": 10, "test": 10}  
}

# --- HELPERS ---

def load_env_file():
    """Manually loads .env file if present to support local dev."""
    if os.path.exists(ENV_FILE):
        print(f"ℹ️  Loading environment variables from {ENV_FILE}...")
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key] = val

def check_aws_credentials():
    """Verifies that AWS credentials are present and valid."""
    session = boto3.Session()
    credentials = session.get_credentials()
    if not session.get_credentials():
        print("\n❌ CRITICAL: No AWS Credentials found!")
        print("   👉 Run 'aws configure' or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in .env")
        return False
    
    # Try a lightweight call to verify permissions (get caller identity)
    try:
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Credentials verified: {identity['Arn']}")
        return True
    except ClientError as e:
        print(f"\n❌ CRITICAL: AWS Credentials found but invalid (403/401).")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  Warning: Could not verify AWS identity: {e}")
        return True

def get_s3_url(url):
    """Converts s3:// to a presigned URL if needed."""
    if not url.startswith("s3://"): return url
    try:
        bucket, key = url.replace("s3://", "").split("/", 1)
        s3 = boto3.client('s3')
        return s3.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)
    except Exception as e:
        print(f"   ❌ S3 Generation Error: {e}")
        return None

def scan_bdd_labels(cfg):
    if not cfg.get('enabled', False): return []
    
    url = cfg.get('labels_url')
    if not url: return []
    
    print("🔍 Scanning BDD Labels...")
    video_set = set()
    signed_url = get_s3_url(url)
    
    if not signed_url: return []

    try:
        with RemoteZip(signed_url) as z:
            files = z.namelist()
            for f in tqdm(files, desc="BDD Labels"):
                if f.endswith(".json"):
                    video_name = os.path.basename(f).replace('.json', '')
                    if len(video_name) > 5: 
                        video_set.add(video_name)
    except Exception as e:
        print(f"   ⚠️  Error reading BDD Labels: {e}")
        
    return sorted(list(video_set))

def scan_zip_source(source_name, urls):
    """Generic scanner for remote zips."""
    video_set = set()
    errors = 0
    
    if isinstance(urls, str): urls = [urls]
    
    pbar = tqdm(urls, desc=f"{source_name} Zips")
    for url in pbar:
        signed_url = get_s3_url(url)
        if not signed_url: 
            errors += 1
            continue

        try:
            with RemoteZip(signed_url) as z:
                files = z.namelist()
                for f in files:
                    # BDD/DanceTrack heuristic: look for 'img1' folder
                    if "img1/" in f and f.endswith(".jpg"):
                        parts = f.split('/')
                        idx = parts.index('img1')
                        if idx > 0:
                            video_set.add(parts[idx-1])
        except RemoteIOError as e:
            if "403" in str(e):
                pbar.write(f"   ❌ ACCESS DENIED (403) for {url}")
                errors += 1
            else:
                pbar.write(f"   ⚠️  Read Error {url}: {e}")
                errors += 1
        except Exception as e:
            pbar.write(f"   ⚠️  Error {url}: {e}")
            errors += 1
            
    if errors == len(urls):
        print(f"\n❌ CRITICAL: Failed to access ALL {source_name} sources. Aborting.")
        return None
        
    return sorted(list(video_set))

def list_visdrone_videos(cfg):
    """Scans local VisDrone zip."""
    if not cfg.get('enabled', False): return []
    print("🔍 Scanning VisDrone source (Local)...")
    
    zip_path = Path(cfg.get('images_zip', ''))
    if not zip_path.exists():
        print(f"   ⚠️ VisDrone zip not found at {zip_path}")
        return []
    
    video_set = set()
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = z.namelist()
            for f in files:
                if f.endswith(".jpg") and "sequences/" in f:
                    parts = f.split('/')
                    for p in parts:
                        if p.startswith("uav"):
                            video_set.add(p)
                            break
    except Exception as e:
        print(f"   ⚠️ Error reading VisDrone zip: {e}")
        
    return sorted(list(video_set))

# --- MAIN ---

def main():
    # 0. Load Env & Config
    load_env_file()
    
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ {CONFIG_FILE} not found.")
        sys.exit(1)
        
    with open(CONFIG_FILE, "rb") as f:
        cfg = toml.load(f)

    # 1. Credential Check (Only if remote sources are enabled)
    remote_enabled = cfg.get('bdd', {}).get('enabled') or cfg.get('dancetrack', {}).get('enabled')
    if remote_enabled:
        if not check_aws_credentials():
            sys.exit(1)

    # 2. Collect Videos
    videos = {}
    
    # BDD
    if cfg.get('bdd', {}).get('enabled', False):
        res = scan_bdd_labels(cfg['bdd'])
        if res: videos["bdd"] = res

    # DanceTrack
    if cfg.get('dancetrack', {}).get('enabled', False):
        print("\n🔍 Scanning DanceTrack sources...")
        res = scan_zip_source("DanceTrack", cfg['dancetrack']['images_url'])
        if res is None: sys.exit(1)
        videos["dancetrack"] = res

    # VisDrone
    if cfg.get('visdrone', {}).get('enabled', False):
        videos["visdrone"] = list_visdrone_videos(cfg.get('visdrone', {}))

    # 3. Validation
    total_videos = sum(len(v) for v in videos.values())
    if total_videos == 0:
        print("\n❌ Error: No videos found in any source.")
        print("   Check your internet connection, AWS credentials, or config paths.")
        sys.exit(1)

    # 4. Select Splits
    manifest = {"val": [], "test": []}
    random.seed(SEED)

    print("\n🔐 Locking splits...")
    for source, vid_list in videos.items():
        if not vid_list: continue
            
        random.shuffle(vid_list)
        
        counts = LOCK_COUNTS.get(source, {"val": 0, "test": 0})
        n_val = counts['val']
        n_test = counts['test']
        
        # Handle small dataset edge case
        if len(vid_list) < (n_val + n_test):
            print(f"   ⚠️ Not enough videos in {source} (Found {len(vid_list)}). Using 50/50 split.")
            n_val = len(vid_list) // 2
            n_test = len(vid_list) - n_val
            
        val_vids = vid_list[:n_val]
        test_vids = vid_list[n_val:n_val+n_test]
        
        for v in val_vids: manifest["val"].append({"source": source, "name": v})
        for v in test_vids: manifest["test"].append({"source": source, "name": v})
            
        print(f"   - {source}: Locked {len(val_vids)} Val, {len(test_vids)} Test")

    # 5. Save
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\n✅ Manifest generated successfully at {MANIFEST_FILE}")
    print(f"   Total Val:  {len(manifest['val'])}")
    print(f"   Total Test: {len(manifest['test'])} (Test set will use Full FPS)")

if __name__ == "__main__":
    main()