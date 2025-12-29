import torch
import numpy as np


from utils.pointnet2_utils import interpolating_points
from Featrec3d_models.PCFeatrec3d_model import FeatureExtractors

dino_backbone_name = 'vit_base_patch8_224.dino'  # 224/8 -> 28 patches.
group_size = 32
num_group = 96

class PointCloudFeatures(torch.nn.Module):
    def __init__(self, image_size = 96):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # We will use only the feature extractor that handles point clouds.
        self.deep_feature_extractor = FeatureExtractors(device = self.device,
                                                       group_size = group_size, num_group = num_group)
        self.deep_feature_extractor.to(self.device)
        self.image_size = image_size

        # Resize operation will still be used for the 3D features.
        self.resize = torch.nn.AdaptiveAvgPool2d((96,96))
        
        self.average = torch.nn.AvgPool2d(kernel_size = 3, stride = 1) 

    def __call__(self, xyz):
        xyz = xyz.to(self.device)

        with torch.no_grad():
            # Only the point cloud features will be extracted here.
            xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(xyz)

        interpolated_feature_maps = interpolating_points(xyz, center.permute(0,2,1), xyz_feature_maps)

        # Return only the 3D feature maps
        return xyz_feature_maps, center, ori_idx, center_idx, interpolated_feature_maps

    def get_features_maps(self, pc):

      # Original shape
     #print("Original pc shape:", pc.shape)  # e.g., [1, 3, 32, 32]

    # Flatten the point cloud
     unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])
     #print("Flattened points shape:", unorganized_pc.shape)  # e.g., [1024, 3]

    # Find nonzero points
     nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)
     #print("Non-zero points count:", nonzero_indices.shape[0])  # actual number of points

    # Select nonzero points
     unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)
     #print("Final points fed to network:", unorganized_pc_no_zeros.shape)  # e.g., [1, 3, 900]

    # Extract features
     xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(unorganized_pc_no_zeros.contiguous())
    
    # Interpolation to full image
     xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                 dtype=xyz_feature_maps.dtype, device=self.device)
     xyz_patch_full[..., nonzero_indices] = interpolated_pc
     xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
     xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
     xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

     return xyz_patch


