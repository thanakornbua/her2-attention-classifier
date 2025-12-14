import numpy as np
import zarr
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)


class ZarrPatchDataset(Dataset):
    """
    Loads Macenko-normalized patches and labels from Zarr.

    Zarr structure expectation:
    - root/patches: (N, H, W, 3) uint8
    - root/labels: (N,) int64 (class indices)
    """

    def __init__(self, zarr_root_path: str, indices: np.ndarray | list[int] | None = None):
        super().__init__()
        self.store = zarr.open(zarr_root_path, mode="r")
        self.patches = self.store["patches"]
        self.labels = self.store["labels"]

        if indices is None:
            self.indices = np.arange(self.patches.shape[0])
        else:
            self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        img_uint8 = self.patches[i]  # (H, W, 3) uint8
        label = int(self.labels[i])

        # to float [0,1]
        img = torch.from_numpy(img_uint8).float() / 255.0  # (H, W, 3)
        # ImageNet normalization
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW
        img = img.permute(2, 0, 1).contiguous()

        # dummy localization target: [x, y, w, h] in normalized coords
        loc_target = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.long), loc_target
