import os
import json
import numpy as np
import open3d as o3d
import argparse
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest



# ------------------------------------------------------------
# CONFIGURABLE LOF SETTINGS
# ------------------------------------------------------------


DETECTOR_SETTINGS = {
    "low":       {"contamination": 0.0001},
    "high":      {"contamination": 0.00015},
}



# ------------------------------------------------------------
# LOF DETECTION
# ------------------------------------------------------------
def lof_detect(points, level):
    contamination = DETECTOR_SETTINGS[level]["contamination"]
    lof = LocalOutlierFactor(
        n_neighbors=30,
        contamination=contamination,
        novelty=False
    )
    y_pred = lof.fit_predict(points)
    return (y_pred == -1)

# ------------------------------------------------------------
# CHUNKING METHODS
# ------------------------------------------------------------
def iso_detect(points, level):
    contamination = DETECTOR_SETTINGS[level]["contamination"]
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto'
    )
    y_pred = iso.fit_predict(points)
    return (y_pred == -1)
def chunk_sequential(points, abn_mask, sample_size):
    N = len(points)
    chunks, labels, gtmask = [], [], []

    for start in range(0, N - sample_size + 1, sample_size):
        end = start + sample_size
        pts = points[start:end]
        mask = abn_mask[start:end]

        fk = mask.mean()
        label = 1 if fk >= 0.0025 else 0

        chunks.append(pts)
        labels.append(label)
        gtmask.append(mask.astype(np.uint8))

    return chunks, labels, gtmask





# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/radia123_/decomap/poly/ACQUISIZIONI/")
    parser.add_argument("--sampling", type=str, required=True,
                        choices=["sequential"])
    parser.add_argument("--detector", type=str,
                    choices=["lof", "iso"], required=True)
    parser.add_argument("--sample_size", type=int, choices=[9216,4096], required=True)
    parser.add_argument("--level", type=str,
                    choices=["low", "high"], required=True)
    args = parser.parse_args()


    point_cloud_dir = os.path.join(args.root, "point_cloud")

    # Output folder
    outdir = f"dataset_output/sampling={args.sampling}_size={args.sample_size}_detector={args.detector}_level={args.level}"
    os.makedirs(outdir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    all_chunks = []
    all_labels = []
    all_gtmask = []

    # Process each scan
    for sd in sorted(os.listdir(point_cloud_dir)):
        ply_path = os.path.join(point_cloud_dir, sd, "PointCloud_merged.ply")
        if not os.path.exists(ply_path):
            continue

        print(f"[INFO] Processing {sd}")

        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points).astype(np.float32)
        if len(points) < args.sample_size:
            continue

        # abnormal detection
        if args.detector == "lof":
            abn_mask = lof_detect(points, args.level)
        elif args.detector == "iso":
            abn_mask = iso_detect(points, args.level)

        # Choose chunking method
        if args.sampling == "sequential":
            chunks, labels, masks = chunk_sequential(points, abn_mask, args.sample_size)

     
        all_chunks.extend(chunks)
        all_labels.extend(labels)
        all_gtmask.extend(masks)

    # Convert to arrays
    data_arr = np.array(all_chunks, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.uint8)
    gtmask_arr = np.array(all_gtmask, dtype=np.uint8)

    # Save
    np.save(os.path.join(outdir, "global_data.npy"), data_arr)
    np.save(os.path.join(outdir, "global_labels.npy"), labels_arr)
    np.save(os.path.join(outdir, "global_gtmask.npy"), gtmask_arr)

    print("\n[DONE] Dataset created!")
    print("Output folder:", outdir)
    print("global_data shape:", data_arr.shape)
    print("global_labels shape:", labels_arr.shape)
    print("global_gtmask shape:", gtmask_arr.shape)


if __name__ == "__main__":
    main()
