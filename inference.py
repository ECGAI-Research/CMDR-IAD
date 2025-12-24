# CMDR-IAD_inference.py
import torch
import numpy as np
import torch.nn.functional as F
import argparse

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from networks.features import MultimodalFeatures
from networks.dataset import get_data_loader
from networks.Map import FeatureMappigMLP
from networks.Dec2d import Decoder2D
from networks.Dec3d import Decoder3D

from utils.metrics_utils import calculate_au_pro


# --------------------------------------------------
def smooth_with_box_filters(fused, weight_l, pad_l, lower_iters=3):
    for _ in range(lower_iters):
        fused = F.conv2d(fused, weight_l, padding=pad_l)
    return fused


# --------------------------------------------------
def set_seeds(sid=42):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)

def reliability_gated_mapping_anomaly(d_map2d, d_map3d):
    """
    Reliability-gated cross-modal mapping anomaly.
    Captures spatially consistent appearance–geometry discrepancies.
    """

    # Joint cross-modal discrepancy
    joint = d_map2d * d_map3d

    # Local spatial consistency estimation
    local_std = F.avg_pool2d(
        (joint.unsqueeze(0).unsqueeze(0) - joint.mean()) ** 2,
        kernel_size=7,
        stride=1,
        padding=3
    ).sqrt().squeeze()

    # Reliability gating weight
    W_local = 4.0 + 2.0 * local_std / (local_std.mean() + 1e-8)

    # Reliability gate
    alpha = torch.sigmoid(W_local * joint)

    # Reliability-gated mapping anomaly
    reliability_gated_mapping_anomaly = alpha * joint

    return reliability_gated_mapping_anomaly
def euclid(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a - b).pow(2).sum(dim=-1).sqrt()
def confidence_weighted_reconstruction_anomaly(d_rec2d, d_rec3d, B=0.3):
    """
    Confidence-weighted reconstruction anomaly.
    Balances appearance and geometry reconstruction deviations.
    """

    # Confidence weights
    w2 = torch.exp(-B * d_rec2d)
    w3 = torch.exp(-B * d_rec3d)

    # Confidence-weighted reconstruction anomaly
    confidence_weighted_reconstruction_anomaly = (
        w2 * d_rec2d + w3 * d_rec3d
    ) / (w2 + w3 + 1e-8)

    return confidence_weighted_reconstruction_anomaly

# --------------------------------------------------
def infer_CFM_full(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = args.dataset_path
    checkpoint_folder = args.checkpoint_folder
    epochs_no = args.epochs_no
    batch_size = args.batch_size
    class_names = args.class_names

    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3

    all_aupro30, all_aupro10, all_aupro5, all_aupro1 = [], [], [], []
    all_pauroc, all_iauroc = [], []

    for class_name in class_names:
        print(f"\n🔍 Inference for class: {class_name}")

        test_loader = get_data_loader(
            "test",
            class_name,
            img_size=224,
            dataset_path=dataset_path
        )

        # ---- Models ----
        map2Dto3D = FeatureMappigMLP(768, 1152).to(device)
        map3Dto2D = FeatureMappigMLP(1152, 768).to(device)
        Dec2D = Decoder2D(dim=768).to(device)
        Dec3D = Decoder3D(out_seq_len=50176, feature_dim=1152).to(device)

        tag = f"{class_name}_independent_{epochs_no}ep_{batch_size}bs"
        ckdir = f"{checkpoint_folder}/{class_name}"

        map2Dto3D.load_state_dict(torch.load(f"{ckdir}/map2Dto3D_{tag}.pth", map_location=device))
        map3Dto2D.load_state_dict(torch.load(f"{ckdir}/map3Dto2D_{tag}.pth", map_location=device))
        Dec2D.load_state_dict(torch.load(f"{ckdir}/Dec2D_{tag}.pth", map_location=device))
        Dec3D.load_state_dict(torch.load(f"{ckdir}/Dec3D_{tag}.pth", map_location=device))

        for m in [Dec2D, map2Dto3D, map3Dto2D, Dec3D]:
            m.eval()

        feature_extractor = MultimodalFeatures()

        weight_l = torch.ones(1, 1, w_l, w_l, device=device) / (w_l ** 2)
        weight_u = torch.ones(1, 1, w_u, w_u, device=device) / (w_u ** 2)

        predictions, gts = [], []
        image_labels, pixel_labels = [], []
        image_preds, pixel_preds = [], []

        for (rgb, pc, depth), gt, label, rgb_path in tqdm(
            test_loader, desc=f"[{class_name}] Inference"
        ):
            rgb, pc = rgb.to(device), pc.to(device)

            with torch.no_grad():
                rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)

                if rgb_patch.dim() == 2:
                    rgb_patch = rgb_patch.unsqueeze(0)
                    xyz_patch = xyz_patch.unsqueeze(0)

                xyz_mask = (xyz_patch.sum(dim=-1) == 0)

                mapped_2d = map3Dto2D(xyz_patch)
                mapped_3d = map2Dto3D(rgb_patch)
                recon_2d = Dec2D(rgb_patch)
                recon_3d = Dec3D(xyz_patch)



                d_map2d = euclid(mapped_2d, rgb_patch)
                d_map3d = euclid(mapped_3d, xyz_patch)
                d_rec2d = euclid(recon_2d, rgb_patch)
                d_rec3d = euclid(recon_3d, xyz_patch)

                d_map2d[xyz_mask] = 0.0
                d_map3d[xyz_mask] = 0.0
                d_rec2d[xyz_mask] = 0.0
                d_rec3d[xyz_mask] = 0.0

                d_map2d = d_map2d.reshape(224,224)
                d_map3d = d_map3d.reshape(224,224)
                d_rec2d = d_rec2d.reshape(224,224)
                d_rec3d = d_rec3d.reshape(224,224)
                

                d_rec2d = d_rec2d.reshape(1, 1, 224, 224)
                d_rec2d = smooth_with_box_filters(d_rec2d, weight_l, pad_l, lower_iters=3)
                d_rec2d = smooth_with_box_filters(d_rec2d, weight_u, pad_u, lower_iters=2)
                d_rec2d = d_rec2d.reshape(224,224)

                A_map = reliability_gated_mapping_anomaly(d_map2d, d_map3d)
                A_rec = confidence_weighted_reconstruction_anomaly(d_rec2d, d_rec3d)

                # Final CMDR-IAD anomaly score (you control the fusion)
                fused = A_map * A_rec

                fused[xyz_mask.reshape(224,224)] = 0.0
                fused = fused.reshape(1, 1, 224, 224)
                fused = smooth_with_box_filters(fused, weight_l, pad_l, lower_iters=5)
                fused = smooth_with_box_filters(fused, weight_u, pad_u, lower_iters=3)
                fused = fused.reshape(224,224)

                
                gts.append(gt.squeeze().cpu().detach().numpy())
                predictions.append((fused / (fused[fused!=0].mean())).cpu().detach().numpy())

                image_labels.append(label)
                pixel_labels.extend(gt.flatten().cpu().detach().numpy())

                image_preds.append((fused / torch.sqrt(fused[fused!=0].mean())).cpu().detach().numpy().max())
                pixel_preds.extend((fused / torch.sqrt(fused.mean())).flatten().cpu().detach().numpy())
                
  
    
        # ---- Metrics ----
        au_pros, _ = calculate_au_pro(gts, predictions)
        
        pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
        image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

        
       

        print("\nAUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC | I-AUROC")
        print(f"  {au_pros[0]:.3f} | {au_pros[1]:.3f} | {au_pros[2]:.3f} | {au_pros[3]:.3f} | "
              f"{pixel_rocauc:.3f} | {image_rocauc:.3f}")

        # =========================
        # PRINT FPS + MEMORY
        # =========================
       
        # === accumulate metrics ===
        all_aupro30.append(au_pros[0])
        all_aupro10.append(au_pros[1])
        all_aupro5.append(au_pros[2])
        all_aupro1.append(au_pros[3])
        all_pauroc.append(pixel_rocauc)
        all_iauroc.append(image_rocauc)
        

    print("\n✅ All classes inference complete.")
    print("\n==================== FINAL MEAN RESULTS ====================\n")

    print(f"Mean AUPRO@30 : {np.mean(all_aupro30)* 100:.1f}")
    print(f"Mean AUPRO@10 : {np.mean(all_aupro10)* 100:.1f}")
    print(f"Mean AUPRO@5  : {np.mean(all_aupro5)* 100:.1f}")
    print(f"Mean AUPRO@1  : {np.mean(all_aupro1)* 100:.1f}")

    print(f"Mean Pixel AUROC : {np.mean(all_pauroc)* 100:.1f}")
    print(f"Mean Image AUROC : {np.mean(all_iauroc)* 100:.1f}")


    print("\n============================================================\n")



# --------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inference for CMDR-IAD (Cross-Modal Mapping and Dual-Branch Reconstruction)"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/radia123_/decomap/processing/mvtec_3d_anomaly_detection",
        help="Dataset path."
    )

    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="/home/radia123_/decomap/checkpoints/mm_Dec_Map",
        help="Folder containing trained checkpoints."
    )

    parser.add_argument(
        "--class_names",
        nargs="+",
        required=True,
        choices=[
            "bagel", "cable_gland", "carrot", "cookie", "dowel",
            "foam", "peach", "potato", "rope", "tire"
        ],
        help="Object categories for inference."
    )

    parser.add_argument(
        "--epochs_no",
        type=int,
        default=50,
        help="Epochs used during training (checkpoint naming)."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used during training (checkpoint naming)."
    )

    args = parser.parse_args()
    infer_CFM_full(args)
