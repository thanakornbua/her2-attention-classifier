class MILDataset:
    # Loads bags of patches for Multiple Instance Learning at slide-level.
    def __init__(self, slide_patch_paths, labels, transform=None):
        self.slide_patch_paths = slide_patch_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.slide_patch_paths)

    def __getitem__(self, idx):
    # TODO: Load all patches in a slide and form a bag
        pass