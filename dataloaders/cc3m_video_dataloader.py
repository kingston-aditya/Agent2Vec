import random
import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os

import sys
root = os.getcwd()
while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, '.git')): root = os.path.dirname(root)
sys.path.insert(1, root)

from dataloaders.base_dataloader import VideoContrastiveDataset
import pdb

class CC3MPatchHFDataset(VideoContrastiveDataset):
    def __init__(self, data_path: str, target_size: tuple = (224, 224)):
        """
        Args:
            data_path: Path to local Hugging Face dataset directory or parquet files.
            target_size: Target (height, width) resolution for all frames.
        """
        self.target_size = target_size
        super().__init__(data_path)

    @staticmethod
    def load_dataset(data_path: str):
        """
        Loads the CC3M/Trinity dataset in standard map-style (non-streaming) mode.
        """
        # Load the dataset fully into memory/cache (streaming=False)
        # We specify the split directly to get a flat Dataset object rather than a DatasetDict
        dataset = load_dataset(data_path, split="train", streaming=False)
        
        # Standardize column names to match what your __getitem__ uses:
        # 1. Handle image pointers/paths
        if "image_path" not in dataset.column_names:
            if "image" in dataset.column_names:
                dataset = dataset.rename_column("image", "image_path")
            elif "jpg" in dataset.column_names:
                dataset = dataset.rename_column("jpg", "image")
                
        # 2. Handle caption/prompt strings
        if "prompt" not in dataset.column_names:
            if "caption" in dataset.column_names:
                dataset = dataset.rename_column("caption", "prompt")
            elif "txt" in dataset.column_names:
                dataset = dataset.rename_column("txt", "prompt")
                
        return dataset

    def create_patch_sequence(self, img: Image.Image) -> np.ndarray:
        """
        Resizes input PIL image, extracts 3x3 spatial patches, 
        and appends them to form a 10-frame video array [10, H, W, 3].
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Step 1: Base image
        main_img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        main_np = np.array(main_img)

        # Step 2: Compute patch dimensions on the original image
        orig_w, orig_h = img.size
        patch_w = orig_w // 3
        patch_h = orig_h // 3

        patches = []
        for i in range(3):
            for j in range(3):
                left = j * patch_w
                top = i * patch_h
                right = (j + 1) * patch_w if j < 2 else orig_w
                bottom = (i + 1) * patch_h if i < 2 else orig_h

                patch = img.crop((left, top, right, bottom))
                patch_resized = patch.resize(self.target_size, Image.Resampling.BILINEAR)
                patches.append(np.array(patch_resized))

        # Shape: [10, 224, 224, 3]
        video_sequence = np.stack([main_np] + patches, axis=0)
        return video_sequence

    def __getitem__(self, index: int):
        try:
            target_image = self.data[index]["image"]
            target_image = target_image.convert('RGB')

            if not target_image.mode == "RGB":
                target_image = target_image.convert("RGB")
        except Exception as e:
            new_index = random.randint(0, len(self.dataset)-1)
            return self.__getitem__(new_index)

        caption = str(self.data[index]["prompt"])

        # Build 10-frame patch video sequence: Shape [10, 224, 224, 3]
        video_pos = self.create_patch_sequence(target_image)

        return {
            "video_pos": video_pos,  # Shape: [10, 224, 224, 3]
            "prompts_neg": caption,  # Shape: [10, 224, 224, 3] (shuffled sequence)
            "prompts": self.__choose_one(random.randint(0, 5))
        }

# if __name__ == "__main__":
#     DATA_DIR = ""
#     dataset = CC3MPatchHFDataset(data_path = DATA_DIR)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
    
#     for i, batch in enumerate(dataloader):
#         pdb.set_trace()
#         break
