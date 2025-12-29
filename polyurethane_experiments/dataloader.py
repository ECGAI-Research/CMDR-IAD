import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def pointcloud_to_3ch_image(pc, target_height=64, target_width=64):
    """
    Convert an (N,3) point cloud to (3,H,W) tensor using interpolation.
    """
    if isinstance(pc, np.ndarray):
        pc = torch.tensor(pc, dtype=torch.float32)

    N = pc.shape[0]
    side = int(np.ceil(np.sqrt(N)))
    pad = side * side - N
    if pad > 0:
        pc = torch.cat([pc, torch.zeros(pad, 3)], dim=0)

    pc_grid = pc.reshape(side, side, 3)
    pc_tensor = pc_grid.permute(2, 0, 1).unsqueeze(0)
    pc_resized = F.interpolate(pc_tensor, size=(target_height, target_width), mode='nearest')
    return pc_resized.squeeze(0)  # (3,H,W)


# -------------------- TRAIN SET --------------------
class TrainSet(Dataset):
    def __init__(self, folder, img_size=64):
        self.train_data = np.load(os.path.join(folder, 'train.npy'))
        self.train_masks = None
        mask_path = os.path.join(folder, 'train_gtmask.npy')
        if os.path.exists(mask_path):
            self.train_masks = np.load(mask_path)
            print(f"✅ Loaded train GT masks: {self.train_masks.shape}")
        else:
            print("⚠️ train_gtmask.npy not found — masks disabled.")

        self.img_size = img_size

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index):
        pc = self.train_data[index]
        pc_img = pointcloud_to_3ch_image(pc, target_height=self.img_size, target_width=self.img_size)

        if self.train_masks is not None:
            mask = self.train_masks[index]  # (N,)
        else:
            mask = np.zeros(pc.shape[0], dtype=np.uint8)


        return pc_img, torch.tensor(mask, dtype=torch.uint8)


# -------------------- TEST SET --------------------
class TestSet(Dataset):
    def __init__(self, folder, img_size=64):
        self.test_data = np.load(os.path.join(folder, 'test.npy'))
        self.test_labels = np.load(os.path.join(folder, 'test_label.npy'))
        print("✅ Loaded test labels:", np.unique(self.test_labels, return_counts=True))

        mask_path = os.path.join(folder, 'test_gtmask.npy')
        if os.path.exists(mask_path):
            self.test_masks = np.load(mask_path)
            print(f"✅ Loaded test GT masks: {self.test_masks.shape}")
        else:
            self.test_masks = None
            print("⚠️ test_gtmask.npy not found — masks disabled.")

        self.img_size = img_size

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        pc = self.test_data[index]
        label = int(self.test_labels[index])
        pc_img = pointcloud_to_3ch_image(pc, target_height=self.img_size, target_width=self.img_size)

        if self.test_masks is not None:
            mask = self.test_masks[index]
        else:
            mask = np.zeros(pc.shape[0], dtype=np.uint8)

        return pc_img, label, torch.tensor(mask, dtype=torch.uint8)
