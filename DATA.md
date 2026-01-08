# 🗄️ BDD-Mini Data Mirror

**Bucket:** `s3://motip-datasets`  
**Region:** `us-east-1` (Adjust if different)  
**Purpose:** This bucket acts as the primary data mirror for the BDD-Mini project. It hosts the raw zip archives (BDD100K & VisDrone) required to build the dataset, as well as a cache of extracted images to speed up re-runs.

## 📦 Bucket Contents

The bucket mirrors the local `data/` directory structure into `bdd-mini-data/`

| File Name | Type | Size | Description |
| :--- | :--- | :--- | :--- |
| **`box_track_labels_trainval.zip`** | Zip | 114 MB | Official BDD100K Tracking Annotations (Train & Val). |
| **`VisDrone2019-MOT-train.zip`** | Zip | 7.5 GB | Official VisDrone Multi-Object Tracking Dataset (Images + Annotations). |
| **`images20-track-train-1.zip`** | Zip | 4.8 GB | BDD100K Training Images (Part 1/7). |
| **`images20-track-train-2.zip`** | Zip | 4.4 GB | BDD100K Training Images (Part 2/7). |
| **`images20-track-train-3.zip`** | Zip | 4.4 GB | BDD100K Training Images (Part 3/7). |
| **`images20-track-train-4.zip`** | Zip | 4.6 GB | BDD100K Training Images (Part 4/7). |
| **`images20-track-train-5.zip`** | Zip | 4.5 GB | BDD100K Training Images (Part 5/7). |
| **`images20-track-train-6.zip`** | Zip | 4.3 GB | BDD100K Training Images (Part 6/7). |
| **`images20-track-train-7.zip`** | Zip | 4.4 GB | BDD100K Training Images (Part 7/7). |
| **`images20-track-val-1.zip`** | Zip | 4.6 GB | BDD100K Validation Images. |
| **`images20-track-test-1.zip`** | Zip | 4.4 GB | BDD100K Test Images (Part 1/2). |
| **`images20-track-test-2.zip`** | Zip | 4.6 GB | BDD100K Test Images (Part 2/2). |
| **`image_cache/`** | Folder | -- | Extracted frames cached by `builder.py` to allow resuming work. |


## Config

In `config.toml` we have few options:

Remote access from s3 mirror without download (via RemoteZip opener):

```toml

images_url = [
    "s3://motip-datasets/bdd-mini-data/images20-track-train-1.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-2.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-3.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-4.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-5.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-6.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-train-7.zip",
    "s3://motip-datasets/bdd-mini-data/images20-track-val-1.zip"
]
labels_url = "s3://motip-datasets/bdd-mini-data/box_track_labels_trainval.zip"
```

Or Remote access from zurich mirror without download (via RemoteZip opener):

```toml
labels_url = "https://dl.cv.ethz.ch/bdd100k/data/box_track_labels_trainval.zip"
images_url = [ "https://dl.cv.ethz.ch/bdd100k/data/track_images_train.zip" ]
```

Or local access to zips downloaded from http://bdd-data.berkeley.edu/download.html

Those are very large files each around 4GB, works well on ec2 :

```bash
> cd data
> wget http://128.32.162.150/bdd100k/mot20/images20-track-train-1.zip
Connecting to 128.32.162.150:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 5172007826 (4.8G) [application/zip]
Saving to: ‘images20-track-train-1.zip’ 
images20-track-train-1.zip   100%[==============>]   4.82G  14.3MB/s    in 3m 41s  
2026-01-08 16:05:47 (22.3 MB/s) - ‘images20-track-train-1.zip’ saved [5172007826/5172007826]
```

In the config :

```toml
labels_url = "data/box_track_labels_trainval.zip"
images_url = [
    "data/images20-track-train-1.zip",
    "data/images20-track-train-2.zip",
    "data/images20-track-train-3.zip",
    "data/images20-track-train-4.zip",
    "data/images20-track-train-5.zip",
    "data/images20-track-train-6.zip",
    "data/images20-track-train-7.zip",
    "data/images20-track-val-1.zip"
]
```

## 🔁 Data Management

Use the AWS CLI to sync data between your EC2 instance (or local machine) and S3.

### Prerequisites
Ensure your EC2 instance has an IAM Role with `AmazonS3FullAccess` (or write access to this bucket).

### 1. Download Everything (Hydrate Environment)
Run this when setting up a fresh EC2 instance. It pulls all required zips and the cache.

```bash
# Go to project root
mkdir -p data

# Download bucket contents to local 'data' folder
aws s3 sync s3://motip-datasets/bdd-mini-data/ ./data
```

### 2. Upload / Backup (Save Progress)
Run this after downloading new zips or generating a large cache that you want to save.

```bash
# Sync local 'data' folder UP to S3
aws s3 sync ./data s3://motip-datasets/bdd-mini-data/
```

### 3. Upload a Single New File
If you just downloaded a new BDD zip manually:

```bash
aws s3 cp ./data/images20-track-train-1.zip s3://motip-datasets/bdd-mini-data/
```

### 4. Clean Up EC2 (Move to S3)
If you are running out of disk space on EC2, use `mv`. This uploads to S3 and **deletes** the local copy.

```bash
# WARNING: Deletes files from your server!
aws s3 mv ./data/images20-track-test-1.zip s3://motip-datasets/bdd-mini-data/
```
