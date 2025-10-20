class PatchDataset:
    def __init__(self, patch_paths, labels, transform=None):
        self.patch_paths = patch_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, idx):
        #TODO: Load patch image and apply transform if needed.
        pass