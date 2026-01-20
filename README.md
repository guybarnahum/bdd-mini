# 🚗 BDD-Mini: Lightweight Dataset Builder

**BDD-Mini** is a specialized tool designed to create a "Mini-Dataset" for Multi-Object Tracking (MOT) training.

It supports building **Hybrid Datasets** by mixing ground-level autonomous driving footage (**BDD100K**), human motion data (**DanceTrack**), and aerial drone footage (**VisDrone**).

Instead of downloading massive datasets (32GB+ for BDD), this tool **streams** data from official mirrors, extracting only the specific video sequences and frames you need. It automatically formats everything into unified **COCO-Video** (for Transformers like MOTIP) and **MOTChallenge** (for TrackEval) formats.

https://github.com/user-attachments/assets/cd4d0d37-252e-4b0c-ba77-b6ec1234f0bb

## 🚀 Features

* **Hybrid Source Support:** Mix and match data from **BDD100K**, **DanceTrack** (streaming), and **VisDrone** (local zip).
* **Reproducible "Golden" Splits:** Uses a `manifest.json` system to lock specific videos into Validation and Test sets forever. This ensures Stage 1, Stage 2, and Stage 3 models are evaluated on the *exact same* footage, preventing data leakage.
* **Smart Streaming:** Connects to remote zip archives via HTTP Range Requests to download *only* the frames you need.
* **Direct S3 Support:** Can stream frames directly from S3 buckets using presigned URLs.
* **Resume Capability:** Images are cached locally in `data/image_cache`. If you interrupt the script (`Ctrl-C`), simply run it again to resume exactly where you left off.
* **Multi-Format Export:** Generates both **COCO-Video** JSONs (Train/Val/Test) and **MOTChallenge** (`gt.txt`) formats simultaneously.
* **Flexible Budgeting:** Define a `train_frame_budget` (e.g., 7000 frames) to limit training size while keeping validation sets fixed.
* **Visualization Tools:** Includes a renderer to generate MP4 movies with bounding boxes directly from your generated labels (COCO or MOT) using FFmpeg. Handles drone footage resolutions automatically.
* **Manifest Generation:** Creates a human-readable `manifest_log.txt` listing exactly which videos ended up in which split.


## 📂 Project Structure
```bash
bdd-mini/
├── config.toml          # ⚙️ Main configuration
├── setup.sh             # 🛠️ Installation script (creates venv)
├── manifest.py          # 🔐 Split Locker (Scans sources & generates manifest.json)
├── builder.py           # 🏗️ Dataset Builder (Reads manifest, streams frames & builds dataset)
├── render.py            # 🎬 Visualization tool (renders MP4s from labels)
├── cleanup.sh           # 🧹 Safer cleanup (protects image cache)
├── manifest.json        # 📜 The "Source of Truth" for Train/Val/Test splits
├── venv/                # 🐍 Virtual environment
├── data/                # 📦 Local cache (labels, image_cache & VisDrone zips)
└── output/              # 📤 Final Dataset Location
    └── dataset/
        ├── annotations/ # 📄 COCO Format (train.json, val.json)
        ├── mot_format/  # 📄 MOT Format (gt/gt.txt, seqinfo.ini)
        ├── val/         # 📄 MOT Format Val Set (Skipped frames for speed)
        ├── test/        # 📄 MOT Format Test Set (Full frames for accuracy)
        ├── train/       # 📄 MOT Format Train Set
        └── manifest_log.txt # 📋 detailed log of built videos
```

## 🛠️ Installation

1.  **Run the Setup Script:**
    This creates the Python virtual environment and installs dependencies (`remotezip`, `tqdm`, `opencv-python`, `boto3`).

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

2.  **Install FFmpeg (Optional but Recommended):**
    Required for `render.py` to generate visualization videos.
    * **Mac:** `brew install ffmpeg`
    * **Linux:** `sudo apt install ffmpeg`


## 🏃 Usage

### 1. Configure Sources (`config.toml`)
You have three options for accessing the massive BDD100K/DanceTrack data.

#### Option A: S3 Mirror (Recommended) 
Stream data directly from our S3 mirror (`s3://motip-datasets`). This is fast and requires no large downloads.
* **Prerequisite:** Ensure your machine has AWS credentials or an IAM Role with `AmazonS3ReadOnlyAccess`.

```toml
[bdd]
enabled = true
labels_url = "s3://motip-datasets/bdd-mini-data/box_track_labels_trainval.zip"
images_url = [
    "s3://motip-datasets/bdd-mini-data/images20-track-train-1.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-2.zip",
    # ... (See DATA.md for full list)
]
```

#### Option B: Zurich Mirror (Public HTTP)
Stream from the ETH Zurich public mirror. No credentials required, but can be unstable.

```toml
[bdd]
enabled = true
labels_url = "https://dl.cv.ethz.ch/bdd100k/data/box_track_labels_trainval.zip"
images_url = "https://dl.cv.ethz.ch/bdd100k/data/track_images_train.zip"
```

#### Option C: Local Files (Offline)
Download the files manually to `data/` if you have a slow connection or want to work offline.

```bash
cd data
wget http://128.32.162.150/bdd100k/mot20/images20-track-train-1.zip
# ... repeat for other parts
```

```toml
[bdd]
enabled = true
labels_url = "data/box_track_labels_trainval.zip"
images_url = [
    "data/images20-track-train-1.zip",
    "data/images20-track-train-2.zip",
    # ...
]
```

### 2. Prepare VisDrone Data (Optional)
If you enabled `[visdrone]`, you need the raw dataset file.

**Note:** Unlike BDD, VisDrone is a single monolithic 7.5GB file. Streaming it over the network is too slow, so you must download it to `data/` first.

#### Option A: S3 Mirror (Recommended for EC2)
Download the zip directly from our mirror. This takes seconds on an EC2 instance.

```bash
aws s3 cp s3://motip-datasets/bdd-mini-data/VisDrone2019-MOT-train.zip data/
```

#### Option B: Manual Download
1. Download **`VisDrone2019-MOT-train.zip`** (Images + Annotations) from the [VisDrone Website](http://aiskyeye.com/).
2. Place it in the `data/` folder.

**Configuration:**
Ensure `config.toml` points to the local file:
```toml
[visdrone]
# Point to the local file you just downloaded
images_zip = "data/VisDrone2019-MOT-train.zip"
labels_zip = "data/VisDrone2019-MOT-train.zip"
```

### 3. Lock the Splits (The "Golden Manifest")
**Important:** Before building, run the manifest generator. This scans your sources and locks specific videos into the Validation and Test sets.

```bash
source venv/bin/activate
python manifest.py
```
* **Output:** Creates `manifest.json`.
* **Why:** This ensures that even if you change your training budget later, your evaluation metrics remain comparable.

### 4. Build the Dataset
Run the builder. It reads `manifest.json`, downloads the mandatory Val/Test videos, and then fills the Training set up to your `train_frame_budget`.

```bash
python builder.py
```

* **Interrupting:** You can hit `Ctrl-C` at any time. Progress is saved in `data/image_cache`. Run the command again to resume instantly.

### 5. Visualize the Data
Verify your dataset by rendering a video with bounding boxes drawn from the generated labels.

#### Render a random video from the Training set
```bash
python render.py
```

#### Render a specific video (Check `manifest_log.txt` for names)
```bash
python render.py --video uav0000086_00000_v
```

#### Verify MOTChallenge export format specifically
```bash
python render.py --format mot
```

**Output:** Videos are saved to `output/rendered/`.

### 6. Clean Up
To remove generated outputs (e.g., to re-roll random videos) while **keeping the downloaded image cache**:

```bash
./cleanup.sh
```
*(The script will ask for confirmation before deleting the cache).*


## ⚙️ Configuration (`config.toml`)

Control every aspect of the dataset generation here.

```toml
[manifest]
path = "manifest.json"    # The file storing locked splits

[dataset]
seed = 42                 # Random seed for reproducibility
output_dir = "output/dataset"
frame_step = 5            # Sample 1 frame every N frames (5 = ~6FPS)

# Budgeting
# Caps the size of the training set. Val/Test sizes are fixed by the manifest.
train_frame_budget = 7000 

[bdd]
enabled = true
# See "Usage" section for S3 vs Local vs HTTP URL options
labels_url = "s3://motip-datasets/bdd-mini-data/box_track_labels_trainval.zip"
images_url = ["..."]

[dancetrack]
enabled = true
# ... URLs ...

[visdrone]
enabled = false           # Set to true after downloading zips
images_zip = "data/VisDrone2019-MOT-train.zip"
labels_zip = "data/VisDrone2019-MOT-train.zip"
```

## ❓ Troubleshooting

* **"S3 Download Failed / 403 Forbidden"**
Ensure your environment has AWS credentials configured (`aws configure` or `.env` file). You need read access to the bucket defined in `config.toml`.

* **"VisDrone zip files are missing!"**
You enabled `[visdrone]` but didn't installed the zip files in `data/`. See step 2 of Usage.

* **"Manifest file not found"**
You must run `python manifest.py` **once** before running `builder.py`.

* **"FFmpeg not found"**
If `render.py` fails, ensure ffmpeg is installed and in your system PATH.

* **"Broken pipe" / FFmpeg Error**
VisDrone videos often have odd resolutions (e.g., 1365px width). The included `render.py` attempts to automatically fixes this by resizing frames to even dimensions before rendering.

* **"Dataset is empty"**
Check `config.toml`. If `num_videos` is too high (e.g., >200), the script might struggle to find enough matching sequences in the specific label zip file provided.

---

## 📝 License
* **BDD100K:** Subject to the [BDD100K License](https://doc.bdd100k.com/license.html).
* **VisDrone:** Subject to the [VisDrone Dataset License](http://aiskyeye.com/).
* **DanceTrack:** Subject to the [DanceTrack License](https://github.com/DanceTrack/DanceTrack).
