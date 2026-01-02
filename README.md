# 🚗 BDD-Mini: Lightweight Dataset Builder

**BDD-Mini** is a specialized tool designed to create a "Mini-BDD100K" dataset for Multi-Object Tracking (MOT) training. 

Instead of downloading the massive 32GB+ dataset, this tool **streams** the data from official mirrors, extracting only the specific video sequences and frames you need. It automatically formats the data into the **COCO-Video** JSON structure required by modern trackers like MOTIP.

---

## 🚀 Features

* **Smart Streaming:** Connects to remote zip archives (via HTTP Range Requests) to download *only* the frames you need.
* **Space Efficient:** Turns a 32GB download into a ~200MB local dataset.
* **Auto-Formatting:** Converts BDD100K labels directly into COCO-Video format (Train/Val JSONs).
* **Configurable:** Customize the number of videos, frame sampling rate (FPS), and download mirrors via a simple TOML file.

---

## 📂 Project Structure

```text
bdd-mini/
├── config.toml         # ⚙️ Configuration (URLs, count, sampling)
├── setup.sh            # 🛠️ Installation script (creates venv)
├── builder.py          # 🏗️ Main script (streams & builds dataset)
├── cleanup.sh          # 🧹 Utility to clean data/output
├── venv/               # 🐍 Virtual environment (auto-created)
├── data/               # 📦 Cache for labels & temporary downloads
└── output/             # 📤 Final Dataset Location
    └── mini_bdd/
        ├── annotations/
        │   └── train.json
        └── train/
            ├── video1-frame001.jpg
            └── ...
```

## 🛠️ Installation
Run the Setup Script: This will create a Python virtual environment (venv) and install dependencies (remotezip, tqdm, etc.).

```bash
chmod +x setup.sh
./setup.sh
```

## Verify Configuration: 

Check config.toml to ensure the download URLs and settings match your needs.

## 🏃 Usage

1. Build the Dataset

To generate your dataset, simply activate the environment and run the builder:

```bash
source venv/bin/activate
python3 builder.py
```

What happens?

* Downloads the tracking labels (approx 114MB).
* Selects N random videos (defined in config).
* Streams specific frames for those videos from the 32GB remote zip.
* Generates output/mini_bdd/annotations/train.json.

2. Clean Up

To remove generated datasets or cache without deleting the code:
```bash 
./cleanup.sh
```