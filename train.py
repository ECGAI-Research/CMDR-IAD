# CMDR-IAD_training.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import argparse


from networks.features import MultimodalFeatures
from networks.dataset import get_data_loader

from networks.Map import  FeatureMappigMLP
from networks.Dec2d import  Decoder2D
from networks.Dec3d import  Decoder3D


# -------------------------------
def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


# -------------------------------
def masked_similarity_loss(a, b, mask, mode="cos", chunk=8192, eps=1e-6):
    """Compute masked cosine or L2 loss (memory-friendly)"""
    B, N, C = a.shape
    a = a.reshape(B * N, C)
    b = b.reshape(B * N, C)
    m = mask.reshape(B * N)

    a = a[~m]
    b = b[~m]
    if a.numel() == 0:
        return a.new_tensor(0.0)

    if mode == "cos":
        sims = []
        for i in range(0, a.size(0), chunk):
            a_c = F.normalize(a[i:i+chunk], dim=-1, eps=eps)
            b_c = F.normalize(b[i:i+chunk], dim=-1, eps=eps)
            sims.append((a_c * b_c).sum(dim=-1))
            del a_c, b_c
        sim_all = torch.cat(sims, dim=0)
        return 1.0 - sim_all.mean()

    else:  # L2
        vals = []
        for i in range(0, a.size(0), chunk):
            diff = a[i:i+chunk] - b[i:i+chunk]
            vals.append((diff.pow(2).sum(dim=-1)).sqrt())
            del diff
        d_all = torch.cat(vals, dim=0)
        return d_all.mean()


# -------------------------------
def Train_CMDR_IAD(args):
    """
    Trains CMDR-IAD components independently for a single object category,
    following the protocol described in the paper.
    """

    class_name = args.class_name
    dataset_path = args.dataset_path
    checkpoint_savepath = args.checkpoint_savepath
    epochs = args.epochs_no
    batch_size = args.batch_size

    loss_mode = "cos"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n🚀 Training class: {class_name} on {device} (Independent training mode)")

    set_seeds()

    # ---- Data Loader ----
    train_loader = get_data_loader(
        "train", class_name=class_name, img_size=224,
        dataset_path=dataset_path, batch_size=batch_size, shuffle=True
    )

    # ---- Models ----
    Feat = MultimodalFeatures()
    map2Dto3D = FeatureMappigMLP(768, 1152).to(device)
    map3Dto2D = FeatureMappigMLP(1152, 768).to(device)
    Dec2D = Decoder2D(dim=768).to(device)
    Dec3D = Decoder3D(out_seq_len=50176, feature_dim=1152).to(device)

    # ---- Independent Optimizers ----
    opt_map2Dto3D = torch.optim.Adam(map2Dto3D.parameters())
    opt_map3Dto2D = torch.optim.Adam(map3Dto2D.parameters())
    opt_dec2D     = torch.optim.Adam(Dec2D.parameters())
    opt_dec3D     = torch.optim.Adam(Dec3D.parameters())

    # ---- Training Loop ----
    for epoch in trange(epochs, desc=f"[{class_name}] Training"):
        stats = {"map2d":0, "map3d":0, "rec2d":0, "rec3d":0, "n":0}

        for (rgb, pc, _), _ in tqdm(train_loader, desc="Batches"):
            rgb, pc = rgb.to(device), pc.to(device)

            with torch.no_grad():
                rp, xp = Feat.get_features_maps(rgb, pc)
                rgb_patch = rp.unsqueeze(0) if rp.dim()==2 else rp
                xyz_patch = xp.unsqueeze(0) if xp.dim()==2 else xp
                xyz_mask = (xyz_patch.sum(dim=-1) == 0)

            # ---- Forward ----
            mapped_3d = map2Dto3D(rgb_patch)
            mapped_2d = map3Dto2D(xyz_patch)
            recon_2d  = Dec2D(rgb_patch)
            recon_3d  = Dec3D(xyz_patch)

            # ---- Compute Losses ----
            L_map2D = masked_similarity_loss(mapped_2d, rgb_patch, xyz_mask, mode=loss_mode)
            L_map3D = masked_similarity_loss(mapped_3d, xyz_patch, xyz_mask, mode=loss_mode)
            L_rec2D = masked_similarity_loss(recon_2d, rgb_patch, xyz_mask, mode=loss_mode)
            L_rec3D = masked_similarity_loss(recon_3d, xyz_patch, xyz_mask, mode=loss_mode)

            # ---- Train Each Model Separately ----
            # 3D → 2D mapping
            if torch.isfinite(L_map2D):
                opt_map3Dto2D.zero_grad()
                L_map2D.backward()
                opt_map3Dto2D.step()

            # 2D → 3D mapping
            if torch.isfinite(L_map3D):
                opt_map2Dto3D.zero_grad()
                L_map3D.backward()
                opt_map2Dto3D.step()

            # 2D reconstruction
            if torch.isfinite(L_rec2D):
                opt_dec2D.zero_grad()
                L_rec2D.backward()
                opt_dec2D.step()

            # 3D reconstruction
            if torch.isfinite(L_rec3D):
                opt_dec3D.zero_grad()
                L_rec3D.backward()
                opt_dec3D.step()

            # ---- Stats ----
            stats["map2d"] += L_map2D.item()
            stats["map3d"] += L_map3D.item()
            stats["rec2d"] += L_rec2D.item()
            stats["rec3d"] += L_rec3D.item()
            stats["n"] += 1

        n = max(1, stats["n"])
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Map2D: {stats['map2d']/n:.4f} | "
              f"Map3D: {stats['map3d']/n:.4f} | "
              f"Rec2D: {stats['rec2d']/n:.4f} | "
              f"Rec3D: {stats['rec3d']/n:.4f}")

    # ---- Save Trained Models ----
    outdir = os.path.join(checkpoint_savepath, class_name)
    os.makedirs(outdir, exist_ok=True)
    tag = f"{class_name}_independent_{epochs}ep_{batch_size}bs"

    torch.save(map2Dto3D.state_dict(), os.path.join(outdir, f"map2Dto3D_{tag}.pth"))
    torch.save(map3Dto2D.state_dict(), os.path.join(outdir, f"map3Dto2D_{tag}.pth"))
    torch.save(Dec2D.state_dict(),     os.path.join(outdir, f"Dec2D_{tag}.pth"))
    torch.save(Dec3D.state_dict(),     os.path.join(outdir, f"Dec3D_{tag}.pth"))

    print(f"\n✅ Saved independent models for {class_name} in {outdir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train Cross-Modal Mapping and Dual-Branch Reconstruction for 2D–3D Multimodal Industrial Anomaly Detection'
    )

    parser.add_argument(
        '--dataset_path',
        default='/home/radia123_/decomap/processing/mvtec_3d_anomaly_detection',
        type=str,
        help='Dataset path.'
    )

    parser.add_argument(
        '--checkpoint_savepath',
        default='./checkpoints/CMDR_IAD_checkpoints',
        type=str,
        help='Where to save the model checkpoints.'
    )

    parser.add_argument(
        '--class_name',
        default=None,
        type=str,
        choices=[
            "bagel", "cable_gland", "carrot", "cookie", "dowel", "foam",
            "peach", "potato", "rope", "tire"
        ],
        help='Category name.'
    )

    parser.add_argument(
        '--epochs_no',
        default=50,
        type=int,
        help='Number of epochs to train.'
    )

    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch size.'
    )

    args = parser.parse_args()

    if args.class_name is None:
        raise ValueError("Please specify --class_name")

    Train_CMDR_IAD(args)

    print("\n Training finished successfully.")

