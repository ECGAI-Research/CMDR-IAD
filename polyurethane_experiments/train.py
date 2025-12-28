import os
import torch
import numpy as np
from itertools import chain
from tqdm import tqdm, trange
import argparse

# Import your TrainSet
from dataloader import TrainSet  

# Import model modules
from Featrec3d_models.PCFeatureEncoder_9216 import PointCloudFeatures
from Featrec3d_models.PCFeaturDecoder import FeatureDecoder_9216


# ==============================
# Utilities
# ==============================
def set_seeds(seed=115):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==============================
# Training Function
# ==============================
def train_featrec3d(args, save_model=True):
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = f"{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs"

    # ------------------------------
    # Load training dataset
    # ------------------------------
    train_dataset = TrainSet(folder=args.dataset_path, img_size=96)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"Loaded {len(train_dataset)} training samples.")

    # ------------------------------
    # Model setup
    # ------------------------------
    feature_extractor = PointCloudFeatures(image_size=96)
    model = FeatureDecoder_9216(out_seq_len=9216, feature_dim=1152).to(device)

    optimizer = torch.optim.AdamW(chain(model.parameters()))
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # ------------------------------
    # Training loop
    # ------------------------------
    for epoch in trange(args.epochs_no, desc=f"Training: {args.class_name}"):

        epoch_sim = []

        for pc, _ in tqdm(train_loader, desc="Processing batches"):
            pc = pc.to(device)
            model.train()

            # Feature extraction
            # Feature extraction
            if args.batch_size == 1:
                xyz_patch = feature_extractor.get_features_maps(pc)
            else:
                xyz_patch = torch.stack([
                    feature_extractor.get_features_maps(pc[i].unsqueeze(0))
                    for i in range(pc.size(0))
                ], dim=0)

            # Forward + loss
            feat_pred = model(xyz_patch)

            xyz_mask = (xyz_patch.sum(dim=-1) == 0)
            loss = 1 - metric(feat_pred[~xyz_mask], xyz_patch[~xyz_mask]).mean()
            epoch_sim.append(1 - loss.item())

            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ NaN or Inf detected — training stopped.")
                return

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{args.epochs_no} → Avg CosSim = {1-np.mean(epoch_sim):.4f}")

    # ------------------------------
    # Save checkpoint
    # ------------------------------
    if save_model:
        os.makedirs(args.checkpoint_savepath, exist_ok=True)

        # If user provided a name → use it
        if args.checkpoint_name:
            ckpt_name = (
                args.checkpoint_name if args.checkpoint_name.endswith(".pth")
                else args.checkpoint_name + ".pth"
            )
        else:
            # default naming
            ckpt_name = f"FeatRec3D_poly_{model_name}.pth"

        save_path = os.path.join(args.checkpoint_savepath, ckpt_name)
        torch.save(model.state_dict(), save_path)

        print(f"\n💾 Model saved to: {save_path}\n")


# ==============================
# Main — User Arguments
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Folder containing train.npy, test.npy, etc.")
    parser.add_argument("--checkpoint_savepath", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Name of the checkpoint file (without .pth)")
    parser.add_argument("--class_name", type=str, default="polyurethane_cuts")
    parser.add_argument("--epochs_no", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    train_featrec3d(args, save_model=True)

