from pathlib import Path
import torch
import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class CardiacDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
    
    @staticmethod
    def extract_files(root):
        files = []
        for subject in root.glob("*"):   
            slice_path = subject/"data"  
            for slice in slice_path.glob("*.npy"):
                files.append(slice)
        return files
    
    
    @staticmethod
    def change_img_to_label_path(path):
        parts = list(path.parts)
        parts[parts.index("data")] = "mask"
        return Path(*parts)

    def augment(self, slice, mask):
        random_seed = torch.randint(0, 1000000, (1,)).item()
        imgaug.seed(random_seed)
        mask = SegmentationMapsOnImage(mask, mask.shape)
        slice_aug, mask_aug = self.augment_params(image=slice, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        return slice_aug, mask_aug
    
    def __len__(self):
        return len(self.all_files)
    
    
    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        mask_path = self.change_img_to_label_path(file_path)
        slices = np.load(file_path).astype(np.float32)  
        mask = np.load(mask_path)
        
        if self.augment_params:
            slices, mask = self.augment(slices, mask)

        return np.expand_dims(slices, 0), np.expand_dims(mask, 0)
        
