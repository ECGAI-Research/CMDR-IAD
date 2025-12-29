import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils.metrics_utils import calculate_au_pro

from dataloader import TestSet
from Featrec3d_models.PCFeatureEncoder_9216 import PointCloudFeatures
from Featrec3d_models.PCFeaturDecoder import  FeatureDecoder_9216

import time


# -------------------- Helper Functions --------------------
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -------------------- Testing Function --------------------
def test_featrec3d(args):
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = TestSet(folder=args.dataset_path, img_size=96)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2
    )


    # Load model + feature extractor
    feature_extractor = PointCloudFeatures()
    model = FeatureDecoder_9216(out_seq_len=9216, feature_dim=1152).to(device)

    print(f"Loading checkpoint from {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # Gaussian smoothing weights
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l ** 2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u ** 2)

    # Metrics storage
    gts, preds = [], []
    image_labels, pixel_labels = [], []
    image_scores, pixel_scores = [], []

    print("\n🚀 Running inference...")
    for pc_img, label, gt_mask in tqdm(test_loader, desc="Testing"):
        torch.cuda.synchronize()
        start_time = time.time()
        pc_img = pc_img.to(device)
        gt_mask = gt_mask.squeeze(0).numpy()

        with torch.no_grad():
            xyz_patch = feature_extractor.get_features_maps(pc_img).unsqueeze(0)
            pred = model(xyz_patch)
            xyz_patch = xyz_patch.squeeze(0)
            pred = pred.squeeze(0)

            # Cosine distance anomaly map
            mask = (xyz_patch.sum(dim=-1) == 0)
            cos_dist = (torch.nn.functional.normalize(pred, dim=1) -
                        torch.nn.functional.normalize(xyz_patch, dim=1)).pow(2).sum(1).sqrt()
            cos_dist[mask] = 0.0
            cos_map = cos_dist.reshape(96, 96)

            # Smoothing
            cos_map = cos_map.unsqueeze(0).unsqueeze(0)
            for _ in range(1):
                cos_map = torch.nn.functional.conv2d(cos_map, weight_l, padding=pad_l)
            for _ in range(0):
                cos_map = torch.nn.functional.conv2d(cos_map, weight_u, padding=pad_u)
            cos_map = cos_map.squeeze()

            # Normalize
            norm = torch.sqrt(cos_map[cos_map != 0].mean())
            norm_map = cos_map / (norm + 1e-6)

        # Flatten for metrics
        pred_flat = norm_map.cpu().numpy().flatten()
        gt_flat = gt_mask.flatten()

        gts.append(gt_flat)
        preds.append(pred_flat)
        image_labels.append(label.item())
        image_scores.append(pred_flat.max())
        pixel_labels.extend(gt_flat)
        pixel_scores.extend(pred_flat)




    # -------------------- Metrics --------------------
    print("\n📊 Calculating metrics...")

    # ✅ Reshape to 2D for PRO metric
    gts_2d, preds_2d = [], []
    for gt, pr in zip(gts, preds):
        side = int(np.ceil(np.sqrt(len(gt))))
        gt_reshaped = np.zeros((side, side))
        pr_reshaped = np.zeros((side, side))
        gt_reshaped.flat[:len(gt)] = gt
        pr_reshaped.flat[:len(pr)] = pr
        gts_2d.append(gt_reshaped)
        preds_2d.append(pr_reshaped)

    au_pros, _ = calculate_au_pro(gts_2d, preds_2d)
    pixel_auc = roc_auc_score(pixel_labels, pixel_scores)
    image_auc = roc_auc_score(image_labels, image_scores)

    print("\n✅ Results")
    print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC | I-AUROC")
    print(f"  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   {au_pros[3]:.3f}  |   {pixel_auc:.3f} |   {image_auc:.3f}")
    


# -------------------- Main --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,default='dataset_output/sampling=sequential_size=9216_detector=iso_level=low')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoint_poly/model_seq_9216_iso_low.pth')
    parser.add_argument('--result_path', type=str, default='./results_9216_iso_low')
    args = parser.parse_args()

    test_featrec3d(args)


