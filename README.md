# 🚗 BDD-Mini: Lightweight Dataset Builder

**BDD-Mini** is a specialized tool designed to create a "Mini-Dataset" for Multi-Object Tracking (MOT) training.

It supports building **Hybrid Datasets** by mixing ground-level autonomous driving footage (**BDD100K**) with aerial drone footage (**VisDrone**).

Instead of downloading the massive 32GB+ BDD dataset, this tool **streams** BDD data from official mirrors, extracting only the specific video sequences and frames you need. It automatically formats everything (Ground + Drone) into unified **COCO-Video** (for Transformers like MOTIP) and **MOTChallenge** (for TrackEval) formats.

https://github.com/user-attachments/assets/cd4d0d37-252e-4b0c-ba77-b6ec1234f0bb

## 🚀 Features

* **Hybrid Source Support:** Mix and match data from **BDD100K** (streaming) and **VisDrone** (local zip).
* **Smart Streaming (BDD):** Connects to remote zip archives via HTTP Range Requests to download *only* the frames you need.
* **Robust Downloading (BDD):** Supports handling split zip files (multiple URLs) for large datasets.
* **Resume Capability:** Images are cached locally in `data/image_cache`. If you interrupt the script (`Ctrl-C`), simply run it again to resume exactly where you left off.
* **Multi-Format Export:** Generates both **COCO-Video** JSONs (Train/Val/Test) and **MOTChallenge** (`gt.txt`) formats simultaneously.
* **Configurable Splits:** Define your own Train/Val/Test ratios in `config.toml` (e.g., 70/15/15).
* **Visualization Tools:** Includes a renderer to generate MP4 movies with bounding boxes directly from your generated labels (COCO or MOT) using FFmpeg. Handles drone footage resolutions automatically.
* **Manifest Generation:** Creates a human-readable `manifest.txt` listing exactly which videos ended up in which split.


## 📂 Project Structure
```bash
bdd-mini/
├── config.toml          
├── setup.sh             # 🛠️ Installation script (creates venv)
├── builder.py           # 🏗️ Main script (streams, splits & builds dataset)
├── render.py            # 🎬 Visualization tool (renders MP4s from labels)
├── cleanup.sh           # 🧹 Safer cleanup (protects image cache)
├── venv/                # 🐍 Virtual environment
├── data/                # 📦 Local cache (labels, image_cache & VisDrone zips)
└── output/              # 📤 Final Dataset Location
    └── dataset/
        ├── annotations/ # 📄 COCO Format (train.json, val.json)
        ├── mot_format/  # 📄 MOT Format (gt/gt.txt, seqinfo.ini)
        ├── images/      # 🖼️ Images sorted by split (train/val/test)
        └── manifest.txt # 📋 List of all processed videos
```

## 🛠️ Installation

1.  **Run the Setup Script:**
    This creates the Python virtual environment and installs dependencies (`remotezip`, `tqdm`, `opencv-python`).

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
Enable or disable sources and set how many videos you want from each.

* **BDD100K:** Enabled by default (Streaming).
* **VisDrone:** Enabled by default (Requires manual download).

### 2. Prepare Data

#### BDD100K (EC2 / Server)
Due to unstable streaming mirrors, it is recommended to download the BDD zip files manually if you are on a fast server (like AWS EC2).
1. Download the split zip files (e.g., `images20-track-train-1.zip`, etc.) into `data/`.
2. Update `config.toml` to point to these local files in the `images_url` list.

#### VisDrone (Optional)
If you enabled `[visdrone]`:
1. Download **`VisDrone2019-MOT-train.zip`** (Images + Annotations).
2. Place it in the `data/` folder.
3. Update `config.toml` to point `images_zip` and `labels_zip` to this file.

### 3. Build the Dataset
Activate the environment and run the builder. It will download labels, select random videos (mixing BDD + VisDrone), and stream/extract frames.

```bash
source venv/bin/activate
python3 builder.py
```

* **Interrupting:** You can hit `Ctrl-C` at any time. Progress is saved in `data/image_cache`. Run the command again to resume instantly.

### 4. Visualize the Data
Verify your dataset by rendering a video with bounding boxes drawn from the generated labels.

#### Render a random video from the Training set
```bash
python3 render.py
```

#### Render a specific video (Check `manifest.txt` for names)
```bash
python3 render.py --video uav0000086_00000_v
```

#### Verify MOTChallenge export format specifically
```bash
python3 render.py --format mot
```

**Output:** Videos are saved to `output/rendered/`.

### 5. Clean Up
To remove generated outputs (e.g., to re-roll random videos) while **keeping the downloaded image cache**:

```bash
./cleanup.sh
```
*(The script will ask for confirmation before deleting the cache).*


## ⚙️ Configuration (`config.toml`)

Control every aspect of the dataset generation here.

```toml
[dataset]
seed = 42                 # Random seed for reproducibility
output_dir = "output/dataset"
frame_step = 5            # Sample 1 frame every N frames (5 = ~6FPS)

# Export Formats
# "coco" -> annotations/train.json (for MOTIP/MOTR)
# "mot"  -> mot_format/train/Video/gt/gt.txt (for TrackEval)
export_formats = ["coco", "mot"]

# Data Splits (Must sum to 1.0)
train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15

[bdd]
enabled = true
num_videos = 10
# Can be a single URL string or a list of file paths
images_url = [
    "data/images20-track-train-1.zip",
    "data/images20-track-train-2.zip",
    # ...
]
labels_url = "data/box_track_labels_trainval.zip"

[visdrone]
enabled = true           # Set to true after downloading zips
num_videos = 10
images_zip = "data/VisDrone_Data.zip"
labels_zip = "data/VisDrone_Data.zip"
```

## ❓ Troubleshooting

* **"VisDrone zip files are missing!"**
You enabled `[visdrone]` but didn't installed the zip files in `data/`. See step 2 of Usage.

* **"FFmpeg not found"**
If `render.py` fails, ensure ffmpeg is installed and in your system PATH.

* **"Broken pipe" / FFmpeg Error**
VisDrone videos often have odd resolutions (e.g., 1365px width). The included `render.py` attempts to automatically fixes this by resizing frames to even dimensions before rendering.

* **"Dataset is empty"**
Check `config.toml`. If `num_videos` is too high (e.g., >200), the script might struggle to find enough matching sequences in the specific label zip file provided.

* **"Streaming Error / Connection Reset"**
The script relies on the ETH Zurich mirror for BDD data. If unstable, try again later or check your internet connection. The Resume feature ensures you don't lose progress. Alternatively, download the zip files manually and point config to local paths.

---

## 📝 License
* **BDD100K:** Subject to the [BDD100K License](https://doc.bdd100k.com/license.html).
* **VisDrone:** Subject to the [VisDrone Dataset License](http://aiskyeye.com/).