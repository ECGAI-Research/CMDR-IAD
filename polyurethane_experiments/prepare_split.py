import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import argparse

def normalize_per_sample(X, eps=1e-12):
    # X shape: (num_samples, sample_size, 3)
    mins = X.min(axis=1, keepdims=True)
    maxs = X.max(axis=1, keepdims=True)
    denom = maxs - mins
    denom[denom == 0] = eps
    X_norm = 2 * (X - mins) / denom - 1
    return X_norm.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Folder that contains global_data.npy, global_labels.npy, global_gtmask.npy",
    )
    args = parser.parse_args()

    ds_dir = args.dataset_dir

    # ------------------- Load global dataset -------------------
    data = np.load(os.path.join(ds_dir, "global_data.npy"))         # (N_samples, sample_size, 3)
    labels = np.load(os.path.join(ds_dir, "global_labels.npy"))     # (N_samples,)
    gtmask = np.load(os.path.join(ds_dir, "global_gtmask.npy"))     # (N_samples, sample_size)

    print("📦 Loaded dataset from:", ds_dir)
    print("   Data shape:", data.shape)
    print("   Labels shape:", labels.shape)
    print("   GT mask shape:", gtmask.shape)
    print(f"   Normal samples:   {np.sum(labels == 0)}")
    print(f"   Abnormal samples: {np.sum(labels == 1)}")

    # ------------------- Split data -------------------
    normal_data = data[labels == 0]
    normal_mask = gtmask[labels == 0]

    X_train, X_test_normal, mask_train, mask_test_normal = train_test_split(
        normal_data, normal_mask, test_size=0.1, random_state=42
    )

    # Abnormal → test set
    X_test_abnormal = data[labels == 1]
    mask_test_abnormal = gtmask[labels == 1]
    y_test_abnormal = labels[labels == 1]

    # Combine test
    X_test = np.concatenate([X_test_normal, X_test_abnormal], axis=0)
    y_test = np.concatenate(
        [np.zeros(X_test_normal.shape[0]), y_test_abnormal],
        axis=0
    )
    mask_test = np.concatenate([mask_test_normal, mask_test_abnormal], axis=0)

    # Shuffle
    X_test, y_test, mask_test = shuffle(X_test, y_test, mask_test, random_state=42)

    # ------------------- Normalize -------------------
    X_train = normalize_per_sample(X_train)
    X_test = normalize_per_sample(X_test)

    # ------------------- Save splits (in same folder) -------------------
    np.save(os.path.join(ds_dir, "train.npy"), X_train)
    np.save(os.path.join(ds_dir, "test.npy"), X_test)
    np.save(os.path.join(ds_dir, "test_label.npy"), y_test)
    np.save(os.path.join(ds_dir, "train_gtmask.npy"), mask_train)
    np.save(os.path.join(ds_dir, "test_gtmask.npy"), mask_test)

    print("\n✅ Dataset prepared:")
    print("   Train shape:", X_train.shape)
    print("   Test shape: ", X_test.shape)
    print("   Test labels:", y_test.shape)
    print("   Train GT mask shape:", mask_train.shape)
    print("   Test GT mask shape: ", mask_test.shape)

    num_train_normals = X_train.shape[0]
    num_test_normals = np.sum(y_test == 0)
    num_test_abnormals = np.sum(y_test == 1)

    print("\n🧾 Dataset breakdown:")
    print(f"   ➤ Train set (Normal only): {num_train_normals}")
    print(f"   ➤ Test set total: {X_test.shape[0]}")
    print(f"       └─ Normals:   {num_test_normals}")
    print(f"       └─ Abnormals: {num_test_abnormals}")

if __name__ == "__main__":
    main()
